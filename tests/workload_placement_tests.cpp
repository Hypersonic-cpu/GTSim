#include "arch_conf.h"
#include "../workloads/common/placement.h"
#include <iostream>
#include <set>

using namespace GTSim;
DAG* build_gemm_h100(int, int, int, WorkloadMemoryOptions);
DAG* build_fa3_h100(int, int, int, int, int, WorkloadMemoryOptions);
namespace {
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> void rejects(F action, const char* message) {
    bool caught = false;
    try { action(); } catch (const std::exception&) { caught = true; }
    require(caught, message);
}

void segmented_allocation() {
    MemoryAddressLayout layout{true, 0, 8192, 8192, 12288, 0};
    TensorRegistry registry(layout);
    require(registry.allocate_tensor(0, "a", {3}, {1}, DataType::FP16).global_base_address == 0, "HBM base");
    require(registry.allocate_tensor(1, "b", {3}, {1}, DataType::FP16, MemoryTier::HBF, true, true).global_base_address == 8192,
            "high HBF segment allocation");
    require(registry.allocate_tensor(2, "c", {3}, {1}, DataType::FP16).global_base_address == 4096,
            "HBM bump cursor was moved by HBF allocation");
    require(registry.reserved_bytes(MemoryTier::HBM) == 8192 && registry.reserved_bytes(MemoryTier::HBF) == 4096,
            "alignment padding not included in capacity accounting");
    rejects([&] { registry.allocate_tensor(3, "overflow", {1}, {1}, DataType::FP16); }, "HBM capacity overflow accepted");
    rejects([&] { registry.validate_access(8192, 64, MemoryTier::HBM, false); }, "tier mismatch accepted");
    rejects([&] { registry.validate_access(8192, 64, MemoryTier::HBF, true); }, "read-only input overwrite accepted");
    rejects([&] { registry.validate_access(4000, 64, MemoryTier::HBM, false); }, "allocation hole addressed");
    rejects([&] { layout.resolve(8128, 128); }, "cross-device access accepted");
    TensorMetadata wrong{4, "wrong", 4096, {1}, {1}, DataType::FP16, MemoryTier::HBF};
    rejects([&] { registry.register_tensor(wrong); }, "tensor metadata disagrees with segment");
    MemoryAddressLayout bad{true, 0, 8192, 4096, 12288, 0};
    rejects([&] { TensorRegistry invalid(bad); }, "overlapping device segments accepted");
}

// Placement changes addresses, never the computation, dependencies or schedule.
void same_schedule(const DAG& left, const DAG& right) {
    require(left.nodes.size() == right.nodes.size(), "placement changed graph size");
    for (std::size_t i = 0; i < left.nodes.size(); ++i) {
        const auto& a = *left.nodes[i]; const auto& b = *right.nodes[i];
        require(a.id == b.id && a.op_type == b.op_type && a.warp_id == b.warp_id &&
                a.sm_id == b.sm_id && a.thread_block_id == b.thread_block_id &&
                a.matrix_id == b.matrix_id && a.depends_on == b.depends_on &&
                a.issue_depends_on == b.issue_depends_on && a.tile.ndims == b.tile.ndims &&
                a.data_type == b.data_type && a.layout == b.layout &&
                a.register_change == b.register_change, "placement changed operation or dependencies");
        for (int d = 0; d < Tile::kMaxDims; ++d)
            require(a.tile.dims[d] == b.tile.dims[d] && a.tile.offs[d] == b.tile.offs[d] &&
                    a.matrix_strides[d] == b.matrix_strides[d], "placement changed tile/strides");
    }
}

void check_dram_extents(const DAG& graph) {
    for (auto* node : graph.nodes) {
        const bool read = node->op_type == OpType::CP_DRAM2SRAM_TMA;
        const bool write = node->op_type == OpType::CP_SRAM2DRAM_TMA;
        if (!read && !write) continue;
        const auto& tensor = graph.tensor_registry.at(node->matrix_id);
        for (int i = 0; i < tile_total_elements(node->tile); ++i) {
            const auto address = compute_global_element_addr(*node, tensor, i);
            require(tensor.contains(address), "TMA tile exceeds its tensor's shape/strides");
        }
        if (write) require(tensor.initial_memory_tier == MemoryTier::HBM, "output was placed in flash");
    }
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected mixed memory config");
        segmented_allocation();
        SimulatorConfig config;
        ArchConfYamlReader(argv[1]).parse(config);
        auto options = workload_memory_options(config);
        options.placement = TensorPlacementProfile::GEMMBInHBF;
        std::unique_ptr<DAG> gemm(build_gemm_h100(128, 256, 64, options));
        require(gemm->tensor_registry.at(0).initial_memory_tier == MemoryTier::HBM &&
                gemm->tensor_registry.at(1).initial_memory_tier == MemoryTier::HBF &&
                gemm->tensor_registry.at(2).initial_memory_tier == MemoryTier::HBM,
                "GEMM A/B/C placement is wrong");
        require(gemm->tensor_registry.at(1).preloaded && gemm->tensor_registry.at(1).read_only,
                "HBF B was not declared preloaded and immutable");
        check_dram_extents(*gemm);
        auto baseline = options; baseline.placement = TensorPlacementProfile::AllHBM;
        std::unique_ptr<DAG> baseline_gemm(build_gemm_h100(128, 256, 64, baseline));
        same_schedule(*gemm, *baseline_gemm);
        // More tiles than H100 SMs exercises persistent multi-wave dependencies.
        std::unique_ptr<DAG> multi(build_gemm_h100(2048, 4096, 64, options));
        std::unique_ptr<DAG> baseline_multi(build_gemm_h100(2048, 4096, 64, baseline));
        same_schedule(*multi, *baseline_multi);

        options.placement = TensorPlacementProfile::FA3QInHBF;
        constexpr int batch = 2, heads_q = 4, heads_kv = 2;
        std::unique_ptr<DAG> fa3(build_fa3_h100(128, 256, heads_q, heads_kv, batch, options));
        require(fa3->tensor_registry.size() == batch * (2 * heads_q + 2 * heads_kv),
                "batched/GQA tensor IDs collide");
        std::set<std::uint64_t> bases;
        for (const auto& [id, tensor] : fa3->tensor_registry.entries()) {
            (void)id;
            require(bases.insert(tensor.global_base_address).second, "batched tensors alias addresses");
            const bool q = tensor.name.starts_with("Q_");
            require((tensor.initial_memory_tier == MemoryTier::HBF) == q, "FA3 fabricated weight/KV/output placement in HBF");
            if (q) require(tensor.preloaded && tensor.read_only, "synthetic Q must be preloaded per invocation");
        }
        std::map<int, std::set<int>> kv_by_cta;
        for (auto* node : fa3->nodes) {
            if (node->op_type != OpType::CP_DRAM2SRAM_TMA) continue;
            const auto& tensor = fa3->tensor_registry.at(node->matrix_id);
            if (tensor.name.starts_with("K_") || tensor.name.starts_with("V_"))
                kv_by_cta[node->thread_block_id].insert(tensor.id);
        }
        require(kv_by_cta[0] == kv_by_cta[1] && kv_by_cta[2] == kv_by_cta[3] &&
                kv_by_cta[0] != kv_by_cta[2] && kv_by_cta[0] != kv_by_cta[4],
                "GQA K/V sharing or batch separation is wrong");
        check_dram_extents(*fa3);
        std::unique_ptr<DAG> baseline_fa3(build_fa3_h100(128, 256, heads_q, heads_kv, batch, baseline));
        same_schedule(*fa3, *baseline_fa3);
        std::cout << "Segment allocation, GEMM placement, batched/GQA IDs, K/V sharing and TMA strides passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
