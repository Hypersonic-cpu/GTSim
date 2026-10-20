#include "arch_conf.h"
#include "tile_ir.h"
#include "memory.h"
#include <functional>
#include <iostream>
#include <sstream>

using namespace GTSim;

static void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    try {
        require(argc == 3, "expected copy and GEMM fixtures");
        auto ir = read_tile_ir(argv[1]);
        auto config = make_h100_sxm_config();
        auto dag = materialize_tile_ir(ir, config);
        require(dag->nodes.size() == 2 && dag->tensor_registry.size() == 2, "copy graph size");
        require(!dag->tensor_registry.contains_tier(MemoryTier::HBF), "HBM registry");
        require(dag->nodes[1]->depends_on == std::vector<int>{0}, "completion dependency");
        auto reject = [](const std::function<void()>& action, const std::string& label) {
            try { action(); }
            catch (const std::exception&) { return; }
            throw std::runtime_error("accepted invalid " + label);
        };
        auto bad_ir = [&](const std::function<void(TileIR&)>& change, const std::string& label) {
            auto bad = ir;
            change(bad);
            reject([&] { materialize_tile_ir(bad, config); }, label);
        };
        bad_ir([](auto& b) { b.tensors[0].tier = MemoryTier::HBF; }, "HBF tensor");
        bad_ir([](auto& b) { b.tensors[0].shape[0] = 0; }, "zero dimension");
        bad_ir([](auto& b) { b.tensors.push_back(b.tensors[0]); }, "duplicate tensor");
        bad_ir([](auto& b) { b.tensors[0].strides[1] = 2; }, "unsupported stride");
        bad_ir([](auto& b) { b.operations[0].region = Tile(0, 1, 1, 32); }, "out of bounds");
        bad_ir([](auto& b) { b.operations[0].tensor = 99; }, "tensor reference");
        bad_ir([](auto& b) { b.operations[0].id = 8; }, "sparse ID");
        bad_ir([](auto& b) { b.operations[0].op = "arbitrary.compute"; }, "operation");
        bad_ir([](auto& b) { b.operations[0].after_issue = {1}; }, "mixed dependency cycle");
        bad_ir([](auto& b) { b.operations[0].after_complete = {0}; }, "self dependency");
        bad_ir([](auto& b) { b.operations[0].after_complete = {99}; }, "dependency reference");
        bad_ir([](auto& b) { b.operations[0].cta = 1; }, "CTA reference");
        bad_ir([](auto& b) { b.operations[0].warp = 12; }, "warp range");
        bad_ir([](auto& b) { b.ctas[0].resources.shared_bytes = 999999; }, "shared capacity");
        bad_ir([](auto& b) { b.ctas[0].resources.register_bytes = 999999; }, "register capacity");
        bad_ir([](auto& b) { b.ctas[0].sm = 999999; }, "SM range");
        bad_ir([](auto& b) { b.operations[0].dtype = DataType::FP32; }, "dtype mismatch");
        auto non_hbm = config;
        non_hbm.memory.topology = MemoryTopology::FullHBF;
        reject([&] { materialize_tile_ir(ir, non_hbm); }, "HBF topology");
        for (const auto& text : {"gtsim.tile_ir 2", "gtsim.tile_ir 1\nunknown",
                               "gtsim.tile_ir 1\ntensor id=0 name=A shape=[1,32] strides=[32,1] dtype=fp16 tier=INVALID",
                               "gtsim.tile_ir 1\ntensor id=0 name=A shape=[1,32,] strides=[32,1] dtype=fp16 tier=HBM",
                               "gtsim.tile_ir 1\n%0 = mma name=x"}) {
            reject([&] { std::istringstream s(text); parse_tile_ir(s); }, "schema");
        }
        // Forward issue edges and repeated completion edges preserve their exact
        // multiplicity in the existing scoreboard.
        ir.operations[0].after_issue = {1};
        ir.operations[1].after_complete.clear();
        dag = materialize_tile_ir(ir, config);
        require(dag->nodes[0]->issue_depends_on == std::vector<int>{1}, "forward issue edge");
        ir.operations[0].after_issue.clear();
        ir.operations[1].after_complete = {0, 0};
        dag = materialize_tile_ir(ir, config);
        require(dag->nodes[1]->depends_on == std::vector<int>({0, 0}), "repeated completion edges");
        auto gemm = read_tile_ir(argv[2]);
        std::vector<int> leaders;
        for (const auto& op : gemm.operations)
            if (op.op == "wgmma.m64n64k16" && op.warp < 4) {
                if (leaders.size() == static_cast<std::size_t>(op.warp)) leaders.push_back(op.id);
            }
        require(leaders.size() == 4, "GEMM collective fixture");
        auto hybrid = gemm;
        hybrid.tensors[1].tier = MemoryTier::HBF;
        auto colocated = config;
        colocated.memory.topology = MemoryTopology::CoLocated;
        auto hybrid_dag = materialize_tile_ir(hybrid, colocated);
        const auto& b_tensor = hybrid_dag->tensor_registry.at(1);
        require(b_tensor.global_base_address == TensorRegistry::kHBFAddressBase, "HBF aperture");
        require(hybrid_dag->tensor_registry.at(0).global_base_address < TensorRegistry::kHBFAddressBase &&
                hybrid_dag->tensor_registry.at(2).global_base_address < TensorRegistry::kHBFAddressBase,
                "HBM aperture");
        auto baseline_dag = materialize_tile_ir(gemm, colocated);
        for (std::size_t i = 0; i < hybrid_dag->nodes.size(); ++i) {
            const auto* a = baseline_dag->nodes[i];
            const auto* b = hybrid_dag->nodes[i];
            require(a->name == b->name && a->op == b->op && a->matrix_id == b->matrix_id &&
                    a->warp_id == b->warp_id && a->sm_id == b->sm_id &&
                    a->depends_on == b->depends_on && a->issue_depends_on == b->issue_depends_on,
                    "placement changed computational DAG");
        }
        // Use a materialized TMA node's actual address through L2 and the
        // existing CoLocated controllers, with the documented default latency.
        MemorySystem memory(0, 128, 0, 4096, 128, 64, true,
                            colocated.memory, colocated.core_frequency_mhz,
                            hybrid_dag->tensor_registry);
        const auto found = std::find_if(hybrid_dag->nodes.begin(), hybrid_dag->nodes.end(), [](const auto* n) {
            return n->matrix_id == 1 && n->op == "cp.dram2sram_tma";
        });
        require(found != hybrid_dag->nodes.end(), "B TMA node");
        const auto keys = compute_dram_subop_lines(**found, b_tensor, 64, 0, 128);
        require(!keys.empty(), "B TMA cache lines");
        for (const auto& key : keys) {
            require(hybrid_dag->tensor_registry.resolve_tier(key.global_line_addr) == MemoryTier::HBF,
                    "TMA address resolves to HBF");
        }
        require(memory.l2()->enqueue_transaction_key(0, keys.front(), false), "TMA line enqueue");
        int completed_cycle = -1;
        for (int cycle = 0; cycle < 20000; ++cycle) {
            if (!memory.step(cycle).empty()) { completed_cycle = cycle; break; }
        }
        const auto stats = memory.statistics(std::max(0, completed_cycle));
        require(completed_cycle >= colocated.memory.hbf.read_latency_cycles &&
                stats.hbf.completed_requests == 1 && stats.hbf.read_bytes == 128 &&
                stats.hbm.request_count == 0 && stats.gpu_to_hbf.reverse_bytes == 128 &&
                stats.gpu_to_hbm.reverse_bytes == 0, "B bypassed HBF controller/link/latency");
        reject([&] { materialize_tile_ir(hybrid, config); }, "silent FullHBM remap");
        auto bad = gemm;
        bad.operations[leaders[3]].op = "st.reg2sram";
        bad.operations[leaders[3]].dtype = DataType::FP16;
        reject([&] { materialize_tile_ir(bad, config); }, "incomplete collective");
        bad = gemm;
        const auto& region = bad.operations[leaders[2]].region;
        bad.operations[leaders[2]].region = Tile(region.r_off, region.c_off + 64, region.r, region.c);
        reject([&] { materialize_tile_ir(bad, config); }, "collective region mismatch");
        bad = gemm;
        bad.operations[leaders[0]].after_complete.push_back(leaders[1]);
        reject([&] { materialize_tile_ir(bad, config); }, "collective sibling dependency");
        std::cout << "Tile IR boundary checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
