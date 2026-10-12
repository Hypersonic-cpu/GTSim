#include "arch_conf.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace GTSim;

DAG* build_gemm_h100(int M = 1024, int N = 1024, int K = 1024) {
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

    const int bm_tile = 128;
    const int bn_tile = 256;
    const int bk_tile = 64;

    const int num_tb_m = M / bm_tile;
    const int num_tb_n = N / bn_tile;
    const int total_tiles = num_tb_m * num_tb_n;

    const int warps_per_tb = 12;
    const int sm_num = 132;
    const int tiles_k = K / bk_tile;

    const int grid_x = std::min(sm_num, total_tiles);
    const int waves = (total_tiles + grid_x - 1) / grid_x;
    const int group = std::min(8, num_tb_n);

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

            int tile_m = (tile_id / group) % num_tb_m;
            int tile_n = (tile_id % group) + (tile_id / group) / num_tb_m * group;
            if (tile_n >= num_tb_n) {
                continue;
            }

            int base_tb_r = tile_m * bm_tile;
            int base_tb_c = tile_n * bn_tile;

            std::vector<std::vector<int>> wgmma_per_k(tiles_k);
            std::vector<std::vector<int>> tma_all_per_k(tiles_k);
            std::vector<int> last_wgmma_per_warp(8, -1);

            // Loader warpgroup (warp 8..11) issues TMA loads
            for (int k = 0; k < tiles_k; ++k) {
                std::vector<int> tma_deps;
                if (k == 0) {
                    if (!prev_tile_last_tma.empty()) {
                        tma_deps.insert(tma_deps.end(),
                                        prev_tile_last_tma.begin(),
                                        prev_tile_last_tma.end());
                    }
                    const bool use_last = (tiles_k == 1) || ((tiles_k % 2) == 0);
                    const auto& cross_wgmma = use_last ? prev_tile_last_wgmma : prev_tile_prev_wgmma;
                    if (!cross_wgmma.empty()) {
                        tma_deps.insert(tma_deps.end(),
                                        cross_wgmma.begin(),
                                        cross_wgmma.end());
                    }
                }
                if (k > 0) {
                    tma_deps.insert(tma_deps.end(),
                                    tma_all_per_k[k - 1].begin(),
                                    tma_all_per_k[k - 1].end());
                }
                if (k > 1) {
                    tma_deps.insert(tma_deps.end(),
                                    wgmma_per_k[k - 2].begin(),
                                    wgmma_per_k[k - 2].end());
                }

                int loader_warp = warp_base + 8;

                auto tma_a = add_node("TB" + std::to_string(tb_id) + "_Tile" +
                                      std::to_string(tile_id) + "_TMA_LoadA_k" +
                                      std::to_string(k),
                                      "LD", "cp.dram2sram_tma", loader_warp,
                                      base_tb_r, k * bk_tile, bm_tile, bk_tile, 0, tma_deps);
                tma_a->sm_id = sm_id;
                tma_a->thread_block_id = tb_id;
                tma_a->matrix_id = TensorID_A;
                tma_a->layout = Layout::RowMajor;
                tma_a->matrix_leading_dim = K;
                tma_all_per_k[k].push_back(tma_a->id);
                std::array<int, 4> tma_b_ids{};
                std::vector<int> tma_b_all;
                tma_b_all.reserve(4);
                for (int chunk = 0; chunk < 4; ++chunk) {
                    int off_c = base_tb_c + chunk * 64;
                    auto tma_b = add_node("TB" + std::to_string(tb_id) + "_Tile" +
                                          std::to_string(tile_id) + "_TMA_LoadB_k" +
                                          std::to_string(k) + "_c" +
                                          std::to_string(chunk),
                                          "LD", "cp.dram2sram_tma", loader_warp,
                                          k * bk_tile, off_c, bk_tile, 64, 0, tma_deps);
                    tma_b->sm_id = sm_id;
                    tma_b->thread_block_id = tb_id;
                    tma_b->matrix_id = TensorID_B;
                    tma_b->layout = Layout::RowMajor;
                    tma_b->matrix_leading_dim = N;
                    tma_b_ids[chunk] = tma_b->id;
                    tma_b_all.push_back(tma_b->id);
                    tma_all_per_k[k].push_back(tma_b->id);
                }

                // Compute warps (0..7): two warp groups, 2 m-halves, 4 k-steps per k
                for (int warp_lane = 0; warp_lane < 8; ++warp_lane) {
                    int warp_id = warp_base + warp_lane;
                    int wg = warp_lane / 4;           // 0 or 1
                    int lane_in_wg = warp_lane % 4;   // 0..3
                    int n_chunk_base = wg * 2;        // 0 or 2

                    for (int m_half = 0; m_half < 2; ++m_half) {
                        int base_r = base_tb_r + m_half * 64 + lane_in_wg * 16;
                        for (int nc = 0; nc < 2; ++nc) {
                            int n_chunk = n_chunk_base + nc;
                            int base_c = base_tb_c + n_chunk * 64;
                            for (int k_step = 0; k_step < 4; ++k_step) {
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
                                wgmma->matrix_id = TensorID_C;
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
            }

            // Store to shared (swizzled) after all k
            std::vector<int> store_ids;
            store_ids.reserve(16);
            std::vector<int> last_round_wgmma_ids;
            if (!wgmma_per_k.empty()) {
                last_round_wgmma_ids = wgmma_per_k.back();
            }
            for (int warp_lane = 0; warp_lane < 8; ++warp_lane) {
                int warp_id = warp_base + warp_lane;
                int wg = warp_lane / 4;
                int lane_in_wg = warp_lane % 4;
                int n_chunk_base = wg * 128;

                int last_wgmma = last_wgmma_per_warp[warp_lane];
                if (last_wgmma == -1) {
                    continue;
                }

                for (int m_half = 0; m_half < 2; ++m_half) {
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
                        store->matrix_id = TensorID_C;
                        store->data_type = DataType::FP16;
                        store->layout = Layout::RowMajor;
                        store_ids.push_back(store->id);
                    }
                }
            }

            // TMA store C (4 chunks)
            std::vector<int> tma_store_ids;
            int loader_warp = warp_base + 8;
            for (int chunk = 0; chunk < 4; ++chunk) {
                int off_c = base_tb_c + chunk * 64;
                auto tma_c = add_node("TB" + std::to_string(tb_id) + "_Tile" +
                                      std::to_string(tile_id) + "_TMA_StoreC_c" +
                                      std::to_string(chunk),
                                      "ST", "cp.sram2dram_tma", loader_warp,
                                      base_tb_r, off_c, bm_tile, 64, 0, store_ids);
                tma_c->sm_id = sm_id;
                tma_c->thread_block_id = tb_id;
                tma_c->matrix_id = TensorID_C;
                tma_c->layout = Layout::RowMajor;
                tma_c->matrix_leading_dim = N;
                if (!tma_store_ids.empty()) {
                    dag->add_issue_dependency(tma_store_ids.back(), tma_c->id);
                }
                tma_store_ids.push_back(tma_c->id);
            }

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

    return dag;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: gtsim_gemm <architecture.yaml> [M N K] "
                     "[--memory-config memory.yaml]\n";
        return 1;
    }

    SimulatorConfig config;
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

    DAG* dag = build_gemm_h100(M, N, K);
    std::cout << "Generated DAG with " << dag->nodes.size() << " nodes\n\n";

    config.silence_mode = true;

    Simulator simulator(dag, config);
    simulator.run(100000000, true);

    delete dag;
    return 0;
}
