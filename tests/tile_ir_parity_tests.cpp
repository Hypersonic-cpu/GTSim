// Reference builder is used only by this test oracle, never by the IR frontend.
#define main gemm_reference_cli_main
#include "../workloads/gemm/gemm_h100.cpp"
#undef main
#include "tile_ir.h"
#include <map>
#include <memory>

static void require(bool value, const std::string& why) {
    if (!value) throw std::runtime_error(why);
}

int main(int argc, char** argv) {
    try {
        require(argc == 5 || argc == 6 || argc == 9, "expected fixture M N K [SMs [BM BN BK]]");
        auto config = make_h100_sxm_config();
        if (argc >= 6) config.num_sms = std::stoi(argv[5]);
        KernelOptions options;
        if (argc == 9) { options.bm = std::stoi(argv[6]); options.bn = std::stoi(argv[7]); options.bk = std::stoi(argv[8]); }
        auto ir = materialize_tile_ir(read_tile_ir(argv[1]), config);
        std::unique_ptr<DAG> reference(build_gemm_h100(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), options, config));
        require(ir->nodes.size() == reference->nodes.size(), "node count differs");
        require(ir->tensor_registry.size() == reference->tensor_registry.size(), "tensor count differs");
        for (int id : {0, 1, 2}) {
            const auto& a = ir->tensor_registry.at(id);
            const auto& b = reference->tensor_registry.at(id);
            require(a.id == b.id && a.name == b.name && a.shape == b.shape && a.strides == b.strides &&
                    a.dtype == b.dtype && a.global_base_address == b.global_base_address, "tensor metadata differs");
            require(a.initial_memory_tier == MemoryTier::HBM, "IR tensor not explicitly HBM");
        }
        require(ir->cta_resources.size() == reference->cta_resources.size(), "CTA count differs");
        for (const auto& [id, a] : ir->cta_resources) {
            const auto& b = reference->cta_resources.at(id);
            require(a.shared_bytes == b.shared_bytes && a.register_bytes == b.register_bytes && a.warps == b.warps,
                    "CTA resources differ");
        }
        std::map<std::string, int> counts;
        std::size_t complete_edges = 0, issue_edges = 0;
        for (std::size_t i = 0; i < ir->nodes.size(); ++i) {
            const auto* a = ir->nodes[i];
            const auto* b = reference->nodes[i];
            std::string where = " at %" + std::to_string(i);
            require(a->id == b->id && a->name == b->name && a->op == b->op &&
                    a->op_type == b->op_type && a->pipeline_type == b->pipeline_type, "operation differs" + where);
            require(a->warp_id == b->warp_id && a->sm_id == b->sm_id && a->thread_block_id == b->thread_block_id,
                    "execution identity differs" + where);
            require(a->matrix_id == b->matrix_id && a->matrix_leading_dim == b->matrix_leading_dim &&
                    a->data_type == b->data_type && a->layout == b->layout &&
                    a->tile.ndims == b->tile.ndims && a->tile.r_off == b->tile.r_off && a->tile.c_off == b->tile.c_off &&
                    a->tile.r == b->tile.r && a->tile.c == b->tile.c, "access region differs" + where);
            for (int d = 0; d < Tile::kMaxDims; ++d)
                require(a->matrix_strides[d] == b->matrix_strides[d] && a->tile.dims[d] == b->tile.dims[d] &&
                        a->tile.offs[d] == b->tile.offs[d], "strides/dimensions differ" + where);
            require(a->depends_on == b->depends_on, "completion dependencies differ" + where);
            require(a->issue_depends_on == b->issue_depends_on, "issue dependencies differ" + where);
            require(a->register_change == b->register_change && a->setup_latency == b->setup_latency &&
                    a->shared_bank_conflict_factor == b->shared_bank_conflict_factor &&
                    a->tma_setup_latency == b->tma_setup_latency && a->tma_issue_interval == b->tma_issue_interval &&
                    a->tma_issue_rate_bytes_per_cycle == b->tma_issue_rate_bytes_per_cycle &&
                    a->memory_coalesce_bytes == b->memory_coalesce_bytes &&
                    a->memory_access_granularity_bytes == b->memory_access_granularity_bytes &&
                    a->issue_group_kind == b->issue_group_kind && a->issue_group_size == b->issue_group_size &&
                    a->issue_group_leader_warp == b->issue_group_leader_warp &&
                    a->issue_group_member_index == b->issue_group_member_index, "execution metadata differs" + where);
            ++counts[a->op];
            complete_edges += a->depends_on.size();
            issue_edges += a->issue_depends_on.size();
        }
        std::cout << "Exact graph parity: " << ir->nodes.size() << " nodes, " << complete_edges
                  << " completion edges, " << issue_edges << " issue edges\n";
        for (const auto& [op, count] : counts) std::cout << op << ": " << count << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
