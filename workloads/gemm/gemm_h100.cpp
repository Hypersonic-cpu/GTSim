#include "arch_conf.h"
#include "../common/workload_options.h"
#include "../common/split_reduce.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace GTSim;

DAG* build_gemm_h100(int M = 1024, int N = 1024, int K = 1024,
                     const KernelOptions& options = {}, const SimulatorConfig& hardware = make_h100_sxm_config()) {
    options.validate(false, M, N, K, hardware);
    DAG* dag = new DAG();
    int node_id = 0;

    constexpr int TensorID_A { 0 };
    constexpr int TensorID_B { 1 };
    constexpr int TensorID_C { 2 };
    dag->tensor_registry.allocate_tensor(TensorID_A, "A", {static_cast<std::uint64_t>(M),
                                           static_cast<std::uint64_t>(K)},
                                 {static_cast<std::uint64_t>(K), 1}, DataType::FP16,
                                 MemoryTier::HBF);
    dag->tensor_registry.allocate_tensor(TensorID_B, "B", {static_cast<std::uint64_t>(K),
                                           static_cast<std::uint64_t>(N)},
                                 {static_cast<std::uint64_t>(N), 1}, DataType::FP16,
                                 MemoryTier::HBF);
    dag->tensor_registry.allocate_tensor(TensorID_C, "C", {static_cast<std::uint64_t>(M),
                                           static_cast<std::uint64_t>(N)},
                                 {static_cast<std::uint64_t>(N), 1}, DataType::FP16,
                                 MemoryTier::HBM);

    const int bm_tile = options.bm;
    const int bn_tile = options.bn;
    const int bk_tile = options.bk;

    const int num_tb_m = M / bm_tile;
    const int num_tb_n = N / bn_tile;
    const int output_tiles = num_tb_m * num_tb_n;
    const int total_tiles = output_tiles * options.split;
    std::vector<int> partials;
    std::vector<std::vector<int>> producer_stores(output_tiles);
    if (options.split > 1) {
        for (int split = 0; split < options.split; ++split) {
            int id = 3 + split;
            partials.push_back(id);
            dag->tensor_registry.allocate_tensor(id, "PartialC" + std::to_string(split),
                {static_cast<std::uint64_t>(M), static_cast<std::uint64_t>(N)},
                {static_cast<std::uint64_t>(N), 1}, DataType::FP32, MemoryTier::HBM);
        }
    }

    const int warps_per_tb = 12;
    const int sm_num = hardware.num_sms;
    const int tiles_k = K / (bk_tile * options.split);

    const int grid_x = std::min(sm_num, total_tiles);
    const int waves = (total_tiles + grid_x - 1) / grid_x;
    const int group = std::min(8, num_tb_n);

    std::vector<std::pair<int,int>> tile_order;
    if (options.order == "grouped") {
        for (int start = 0; start < num_tb_m; start += options.group_size)
            for (int n = 0; n < num_tb_n; ++n)
                for (int m = start; m < std::min(num_tb_m, start + options.group_size); ++m)
                    tile_order.emplace_back(m, n);
    } else if (options.order == "original") {
        // Original groups columns first. Clamp the final column group so
        // N/Bn need not be divisible by the grouping width.
        for (int start = 0; start < num_tb_n; start += group)
            for (int m = 0; m < num_tb_m; ++m)
                for (int n = start; n < std::min(num_tb_n, start + group); ++n)
                    tile_order.emplace_back(m,n);
    } else {
        for (int idx = 0; idx < output_tiles; ++idx) {
            if (options.order == "m-major") tile_order.emplace_back(idx % num_tb_m, idx / num_tb_m);
            else tile_order.emplace_back(idx / num_tb_n, idx % num_tb_n);
        }
    }
    std::unordered_map<std::string, DAGNode*> node_dict;

    auto add_node = [&](const std::string& name, const std::string& pipeline_type,
                        const std::string& op, int warp_id, int r_off, int c_off,
                        int r, int c, int register_change,
                        const std::vector<int>& depends_on = {}) -> DAGNode* {
        DAGNode* node = new DAGNode(node_id++, name, pipeline_type, op, warp_id,
                                    depends_on, register_change, Tile(r_off, c_off, r, c));
        dag->add_node(node);
        node_dict[name] = node;
        return node;
    };

    for (int block_idx = 0; block_idx < grid_x; ++block_idx) {
        int tb_id = block_idx;
        int sm_id = block_idx;
        int warp_base = tb_id * warps_per_tb;
        dag->cta_resources[tb_id] = options.resources(false);
        std::vector<std::vector<int>> buffer_consumers(options.stages);

            std::vector<int> prev_tile_post_store;
            std::vector<int> prev_tile_last_wgmma;
            std::vector<int> prev_tile_prev_wgmma;
            std::vector<int> prev_tile_last_tma;
            std::vector<int> prev_tile_store_issue;

        for (int w = 0; w < waves; ++w) {
            int tile_id = w * grid_x + block_idx;
            if (tile_id >= total_tiles) {
                break;
            }

            int output_idx = tile_id / options.split;
            int split_id = tile_id % options.split;
            int output_tensor = options.split > 1 ? partials[split_id] : TensorID_C;
            int k_base = split_id * (K / options.split);
            auto [tile_m, tile_n] = tile_order[output_idx];

            int base_tb_r = tile_m * bm_tile;
            int base_tb_c = tile_n * bn_tile;

            std::vector<std::vector<int>> wgmma_per_k(tiles_k);
            std::vector<std::vector<int>> tma_all_per_k(tiles_k);
            std::vector<int> last_wgmma_per_warp(bn_tile / 32, -1);

            // Loader warpgroup (warp 8..11) issues TMA loads
            for (int k = 0; k < tiles_k; ++k) {
                std::vector<int> tma_deps;
                const auto& previous_tma_issue =
                    k == 0 ? prev_tile_last_tma : tma_all_per_k[k - 1];
                if (options.gemm_baseline() && k == 0) {
                    // k and k-2 reuse the same ping-pong input buffer. The
                    // next output tile starts at buffer 0, so wait for the
                    // prior tile's final use of that buffer.
                    const bool reuse_last_k_buffer = (tiles_k % 2) != 0;
                    const auto& cross_wgmma = reuse_last_k_buffer
                                                   ? prev_tile_last_wgmma
                                                   : prev_tile_prev_wgmma;
                    if (!cross_wgmma.empty()) {
                        tma_deps.insert(tma_deps.end(),
                                        cross_wgmma.begin(),
                                        cross_wgmma.end());
                    }
                }
                if (options.gemm_baseline() && k > 1) {
                    tma_deps.insert(tma_deps.end(),
                                    wgmma_per_k[k - 2].begin(),
                                    wgmma_per_k[k - 2].end());
                }

                if (!options.gemm_baseline()) {
                    const auto& consumed = buffer_consumers[k % options.stages];
                    tma_deps.insert(tma_deps.end(), consumed.begin(), consumed.end());
                }
                int loader_warp = warp_base + 8;

                auto tma_a = add_node("TB" + std::to_string(tb_id) + "_Tile" +
                                      std::to_string(tile_id) + "_TMA_LoadA_k" +
                                      std::to_string(k),
                                      "LD", "cp.dram2sram_tma", loader_warp,
                                      base_tb_r, k_base + k * bk_tile, bm_tile, bk_tile, 0, tma_deps);
                tma_a->sm_id = sm_id;
                tma_a->thread_block_id = tb_id;
                tma_a->matrix_id = TensorID_A;
                tma_a->layout = Layout::RowMajor;
                tma_a->matrix_leading_dim = K;
                tma_all_per_k[k].push_back(tma_a->id);
                std::array<int, 4> tma_b_ids{};
                std::vector<int> tma_b_all;
                tma_b_all.reserve(4);
                for (int chunk = 0; chunk < bn_tile / 64; ++chunk) {
                    int off_c = base_tb_c + chunk * 64;
                    auto tma_b = add_node("TB" + std::to_string(tb_id) + "_Tile" +
                                          std::to_string(tile_id) + "_TMA_LoadB_k" +
                                          std::to_string(k) + "_c" +
                                          std::to_string(chunk),
                                          "LD", "cp.dram2sram_tma", loader_warp,
                                          k_base + k * bk_tile, off_c, bk_tile, 64, 0, tma_deps);
                    tma_b->sm_id = sm_id;
                    tma_b->thread_block_id = tb_id;
                    tma_b->matrix_id = TensorID_B;
                    tma_b->layout = Layout::RowMajor;
                    tma_b->matrix_leading_dim = N;
                    tma_b_ids[chunk] = tma_b->id;
                    tma_b_all.push_back(tma_b->id);
                    tma_all_per_k[k].push_back(tma_b->id);
                }
                if (!previous_tma_issue.empty()) {
                    // Keep TMA stages in issue order, but allow the previous
                    // stage's transfers to remain in flight.
                    for (int previous_tma : previous_tma_issue) {
                        dag->add_issue_dependency(previous_tma, tma_a->id);
                        for (int tma_b : tma_b_all) {
                            dag->add_issue_dependency(previous_tma, tma_b);
                        }
                    }
                }

                // Compute warps (0..7): two warp groups, 2 m-halves, 4 k-steps per k
                for (int warp_lane = 0; warp_lane < bn_tile / 32; ++warp_lane) {
                    int warp_id = warp_base + warp_lane;
                    int wg = warp_lane / 4;           // 0 or 1
                    int lane_in_wg = warp_lane % 4;   // 0..3
                    int n_chunk_base = wg * 2;        // 0 or 2

                    for (int m_half = 0; m_half < bm_tile / 64; ++m_half) {
                        int base_r = base_tb_r + m_half * 64 + lane_in_wg * 16;
                        for (int nc = 0; nc < 2; ++nc) {
                            int n_chunk = n_chunk_base + nc;
                            int base_c = base_tb_c + n_chunk * 64;
                            for (int k_step = 0; k_step < bk_tile / 16; ++k_step) {
                                int k_step_off = k_step * 16;
                                std::vector<int> deps = {tma_a->id};
                                deps.insert(deps.end(), tma_b_all.begin(), tma_b_all.end());
                                if (last_wgmma_per_warp[warp_lane] != -1) {
                                    deps.push_back(last_wgmma_per_warp[warp_lane]);
                                }

                                auto wgmma = add_node("TB" + std::to_string(tb_id) + "_Tile" +
                                                      std::to_string(tile_id) + "_WGMMA_w" +
                                                      std::to_string(warp_id) + "_k" +
                                                      std::to_string(k) + "_mh" +
                                                      std::to_string(m_half) + "_n" +
                                                      std::to_string(n_chunk) + "_ks" +
                                                      std::to_string(k_step),
                                                      "Tensor", "wgmma.m64n64k16", warp_id,
                                                      base_r, base_c, 16, 64, 0, deps);
                                wgmma->sm_id = sm_id;
                                wgmma->thread_block_id = tb_id;
                                wgmma->matrix_id = output_tensor;
                                wgmma->data_type = DataType::FP32;
                                if (k == 0 && !prev_tile_post_store.empty()) {
                                    wgmma->depends_on.insert(wgmma->depends_on.end(),
                                                             prev_tile_post_store.begin(),
                                                             prev_tile_post_store.end());
                                }

                                wgmma_per_k[k].push_back(wgmma->id);
                                last_wgmma_per_warp[warp_lane] = wgmma->id;
                                (void)k_step_off;
                            }
                        }
                    }
                }
                buffer_consumers[k % options.stages] = wgmma_per_k[k];
            }

            // Store to shared (swizzled) after all k
            std::vector<int> store_ids;
            store_ids.reserve(16);
            std::vector<int> last_round_wgmma_ids;
            if (!wgmma_per_k.empty()) {
                last_round_wgmma_ids = wgmma_per_k.back();
            }
            for (int warp_lane = 0; warp_lane < bn_tile / 32; ++warp_lane) {
                int warp_id = warp_base + warp_lane;
                int wg = warp_lane / 4;
                int lane_in_wg = warp_lane % 4;
                int n_chunk_base = wg * 128;

                int last_wgmma = last_wgmma_per_warp[warp_lane];
                if (last_wgmma == -1) {
                    continue;
                }

                for (int m_half = 0; m_half < bm_tile / 64; ++m_half) {
                    int base_r = base_tb_r + m_half * 64 + lane_in_wg * 16;
                    for (int nt = 0; nt < 8; ++nt) {
                        int base_c = base_tb_c + n_chunk_base + nt * 16;
                        std::vector<int> deps = {last_wgmma};
                        deps.insert(deps.end(),
                                    last_round_wgmma_ids.begin(),
                                    last_round_wgmma_ids.end());
                        auto store = add_node("TB" + std::to_string(tb_id) + "_Tile" +
                                              std::to_string(tile_id) + "_StSram_w" +
                                              std::to_string(warp_id) + "_mh" +
                                              std::to_string(m_half) + "_n" +
                                              std::to_string(nt),
                                              "ST", "st.reg2sram", warp_id,
                                              base_r, base_c, 16, 16, 0, deps);
                        store->sm_id = sm_id;
                        store->thread_block_id = tb_id;
                        store->matrix_id = output_tensor;
                        store->data_type = options.split > 1 ? DataType::FP32 : DataType::FP16;
                        store->layout = Layout::RowMajor;
                        store_ids.push_back(store->id);
                    }
                }
            }

            // TMA store C (4 chunks)
            std::vector<int> tma_store_ids;
            int loader_warp = warp_base + 8;
            for (int chunk = 0; chunk < bn_tile / 64; ++chunk) {
                int off_c = base_tb_c + chunk * 64;
                auto tma_c = add_node("TB" + std::to_string(tb_id) + "_Tile" +
                                      std::to_string(tile_id) + "_TMA_StoreC_c" +
                                      std::to_string(chunk),
                                      "ST", "cp.sram2dram_tma", loader_warp,
                                      base_tb_r, off_c, bm_tile, 64, 0, store_ids);
                tma_c->sm_id = sm_id;
                tma_c->thread_block_id = tb_id;
                tma_c->matrix_id = output_tensor;
                tma_c->data_type = options.split > 1 ? DataType::FP32 : DataType::FP16;
                tma_c->layout = Layout::RowMajor;
                tma_c->matrix_leading_dim = N;
                if (!tma_store_ids.empty()) {
                    dag->add_issue_dependency(tma_store_ids.back(), tma_c->id);
                }
                tma_store_ids.push_back(tma_c->id);
            }

            producer_stores[output_idx].insert(producer_stores[output_idx].end(), tma_store_ids.begin(), tma_store_ids.end());

            bool has_next_tile = (tile_id + grid_x) < total_tiles;
            prev_tile_post_store.clear();
            if (has_next_tile) {
                prev_tile_post_store = tma_store_ids;
            }

            prev_tile_last_wgmma.clear();
            if (!wgmma_per_k.empty()) {
                prev_tile_last_wgmma = wgmma_per_k.back();
            }
            prev_tile_prev_wgmma.clear();
            if (tiles_k >= 2) {
                prev_tile_prev_wgmma = wgmma_per_k[tiles_k - 2];
            }
            prev_tile_last_tma.clear();
            if (!tma_all_per_k.empty()) {
                prev_tile_last_tma = tma_all_per_k.back();
            }
            prev_tile_store_issue.clear();
            if (has_next_tile) {
                prev_tile_store_issue = tma_store_ids;
            }
        }
    }

    if (options.split > 1) {
        for (int i = 0; i < output_tiles; ++i) {
            auto [m,n] = tile_order[i];
            append_split_reduce(dag, partials, {}, producer_stores[i], TensorID_C,
                m * bm_tile, n * bn_tile, bm_tile, bn_tile, N,
                grid_x + i, i % sm_num);
        }
    }
    return dag;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: gtsim_gemm <architecture.yaml> [M N K] "
                     "[--memory-config memory.yaml]\n";
        return 1;
    }

    SimulatorConfig config;
    KernelOptions options;
    std::vector<std::string> workload_args;
    try {
        config = make_h100_sxm_variants(argv[1]);
        for (int index = 2; index < argc; ++index) {
            if (std::string(argv[index]) == "--memory-config") {
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "--memory-config requires a YAML path");
                }
                ArchConfYamlReader(argv[++index]).parse(config);
            } else if (std::string(argv[index]).starts_with("--")) {
                std::string flag = argv[index];
                if (index + 1 >= argc || !options.parse(flag, argv[++index]))
                    throw std::invalid_argument("unknown or missing kernel option: " + flag);
            } else {
                workload_args.emplace_back(argv[index]);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "Configuration error: " << error.what() << '\n';
        return 1;
    }

    int M = 1024;
    int N = 1024;
    int K = 1024;

    if (!workload_args.empty()) M = std::atoi(workload_args[0].c_str());
    if (workload_args.size() > 1) N = std::atoi(workload_args[1].c_str());
    if (workload_args.size() > 2) K = std::atoi(workload_args[2].c_str());

    std::cout << "=== H100 GEMM ===\n\n";

    DAG* dag = nullptr;
    try { dag = build_gemm_h100(M, N, K, options, config); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
    std::cout << "Generated DAG with " << dag->nodes.size() << " nodes\n\n";

    config.silence_mode = true;

    bool complete = false;
    try {
        Simulator simulator(dag, config);
        complete = simulator.run(100000000, true);
    } catch (const std::exception& error) {
        std::cerr << "Simulator error: " << error.what() << '\n';
        delete dag;
        return 3;
    }
    delete dag;
    return complete ? 0 : 3;
}
