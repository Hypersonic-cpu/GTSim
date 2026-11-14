#include "execution_ir/kernel_graph_builder.h"

#include <algorithm>
#include <cassert>
#include <limits>

namespace GTSim::ExecutionIR {

KernelGraphBuilder::KernelGraphBuilder(const TensorRegistry& global_tensors,
                                       const SimulatorConfig& config)
    : global_tensors_(global_tensors),
      config_(config),
      sm_next_subpartition_(std::max(0, config.num_sms), 0) {}

DataType KernelGraphBuilder::parse_dtype(const std::string& value) {
    if (value == "fp16") return DataType::FP16;
    if (value == "bf16") return DataType::BF16;
    if (value == "fp32") return DataType::FP32;
    if (value == "int32") return DataType::INT32;
    assert(false && "unsupported execution dtype");
    return DataType::FP16;
}

const std::unordered_map<std::string_view, std::string_view>&
KernelGraphBuilder::supported_ops() {
    static const std::unordered_map<std::string_view, std::string_view> values =
        {{"cp.dram2sram_tma", "LD"},
        {"cp.sram2dram_tma", "ST"},
        {"st.reg2sram", "ST"},
        {"wgmma.m64n64k16", "Tensor"},
        {"mma.sync.m16n8k16", "Tensor"},
        {"local_max", "SIMD"},
        {"reduce_max", "SIMD"},
        {"merge_max", "SIMD"},
        {"local_sum", "SIMD"},
        {"reduce_sum", "SIMD"},
        {"merge_sum", "SIMD"},
        {"logsum", "SIMD"},
        {"acc_o_scale", "SIMD"},
        {"acc_s_cast", "SIMD"},
        {"shfl_max", "SHFL"},
        {"shfl_sum", "SHFL"},
        {"scores_scale", "SFU"},
        {"exp2", "SFU"},
        {"acc_o_div", "SFU"},
        {"simd.add", "SIMD"},
        {"simd.sub", "SIMD"},
        {"simd.mul", "SIMD"},
        {"simd.fma", "SIMD"},
        {"simd.max", "SIMD"},
        {"simd.min", "SIMD"},
        {"simd.cast", "SIMD"},
        {"simd.fill", "SIMD"},
        {"simd.identity", "SIMD"},
        {"sfu.exp", "SFU"},
        {"sfu.log", "SFU"},
        {"sfu.sqrt", "SFU"},
        {"sfu.rsqrt", "SFU"},
        {"sfu.divide", "SFU"},
        {"sfu.sin", "SFU"},
        {"sfu.cos", "SFU"},
        {"shfl", "SHFL"}};
    return values;
}

void KernelGraphBuilder::require_started() const {
    assert(dag_);
}

void KernelGraphBuilder::clear_bindings() noexcept {
    tensor_ids_.clear();
    cta_sms_.clear();
    cta_warps_.clear();
    warp_bases_.clear();
    prefetch_windows_.clear();
#ifndef NDEBUG
    output_areas_.clear();
#endif
}

void KernelGraphBuilder::begin(KernelHeader header) {
    assert(!dag_);
    assert(!header.launch_id.empty() && !header.stream_id.empty());
    assert(!header.tensor_bindings.empty());
    std::vector<int> ids(header.tensor_bindings.size(), -1);
    for (const auto& [local, global] : header.tensor_bindings) {
        assert(local >= 0 &&
               static_cast<std::size_t>(local) < ids.size());
        (void)global_tensors_.at(global);
        auto& bound = ids[static_cast<std::size_t>(local)];
        assert(bound < 0);
        bound = global;
    }
#ifndef NDEBUG
    for (const int id : ids) assert(id >= 0);
#endif

    header_ = std::move(header);
    tensor_ids_ = std::move(ids);
    dag_ = std::make_unique<DAG>();
    dag_->use_tensor_registry(global_tensors_);
    cta_sms_.clear();
    cta_warps_.clear();
    warp_bases_.clear();
    sm_next_subpartition_.assign(std::max(0, config_.num_sms), 0);
    next_warp_id_ = 0;
    prefetch_windows_.clear();
#ifndef NDEBUG
    output_areas_.assign(tensor_ids_.size(), 0);
#endif
}

void KernelGraphBuilder::add_cta(const CTA& cta) {
    require_started();
    assert(cta.id == static_cast<int>(cta_sms_.size()));
    assert(cta.sm >= 0 && cta.sm < config_.num_sms);
    assert(cta.warps >= 1 && cta.warps <= config_.max_warps_per_sm &&
           cta.warps <= 32);
    assert(cta.shared_bytes > 0 &&
           cta.shared_bytes <= config_.shared_memory_per_cta_bytes);
    assert(cta.register_bytes > 0 &&
           cta.register_bytes <= config_.register_file_per_sm_bytes);
    const auto desired =
        static_cast<std::uint64_t>(sm_next_subpartition_[cta.sm] % 4);
    next_warp_id_ += static_cast<int>(
        (desired + 4 - static_cast<std::uint64_t>(next_warp_id_) % 4) % 4);
    assert(next_warp_id_ <= std::numeric_limits<int>::max() - cta.warps);
    cta_sms_.push_back(cta.sm);
    cta_warps_.push_back(cta.warps);
    warp_bases_.push_back(next_warp_id_);
    next_warp_id_ += cta.warps;
    sm_next_subpartition_[cta.sm] =
        (sm_next_subpartition_[cta.sm] + cta.warps) % 4;
    dag_->cta_resources[cta.id] = {cta.shared_bytes, cta.register_bytes,
                                   cta.warps};
}

void KernelGraphBuilder::add_instruction(const Instruction& op) {
    require_started();
    assert(op.id == static_cast<int>(dag_->nodes.size()));
    assert(!op.name.empty());
    assert(op.cta >= 0 && static_cast<std::size_t>(op.cta) < cta_sms_.size());
#ifndef NDEBUG
    const auto cta = static_cast<std::size_t>(op.cta);
    assert(op.warp >= warp_bases_[cta] &&
           op.warp < warp_bases_[cta] + cta_warps_[cta]);
#endif
#ifndef NDEBUG
    const auto supported = supported_ops().find(op.op);
    assert(supported != supported_ops().end() && supported->second == op.pipeline);
#endif
    const auto dtype = parse_dtype(op.dtype);
    assert(op.tile[0] >= 0 && op.tile[1] >= 0 && op.tile[2] > 0 &&
           op.tile[3] > 0);

#ifndef NDEBUG
    for (const auto* list : {&op.after_issue, &op.after_complete})
        for (const int dep : *list)
            assert(dep >= 0 && dep < op.id);
#endif

    assert(op.tensor >= 0 &&
           static_cast<std::size_t>(op.tensor) < tensor_ids_.size());
    const auto local_tensor = static_cast<std::size_t>(op.tensor);
#ifndef NDEBUG
    const bool global =
        op.op == "cp.dram2sram_tma" || op.op == "cp.sram2dram_tma";
    if (global) {
        const int global_id = tensor_ids_[local_tensor];
        const auto& tensor = global_tensors_.at(global_id);
        assert(tensor.shape.size() == 2);
        assert(std::uint64_t(op.tile[0]) + op.tile[2] <= tensor.shape[0] &&
               std::uint64_t(op.tile[1]) + op.tile[3] <= tensor.shape[1]);
        assert(op.leading_dim == static_cast<int>(tensor.strides[0]));
        assert(dtype == tensor.dtype);
        assert(op.op != "cp.sram2dram_tma" || !tensor.read_only);
        if (op.op == "cp.sram2dram_tma") {
            output_areas_[static_cast<std::size_t>(op.tensor)] +=
                static_cast<std::uint64_t>(op.tile[2]) * op.tile[3];
        }
    }
#endif

    auto* node = new DAGNode(
        op.id, op.name, op.pipeline, op.op, op.warp, op.after_complete, 0,
        Tile(op.tile[0], op.tile[1], op.tile[2], op.tile[3]), dtype);
    node->issue_depends_on = op.after_issue;
    node->sm_id = cta_sms_[static_cast<std::size_t>(op.cta)];
    node->thread_block_id = op.cta;
    node->matrix_id = tensor_ids_[local_tensor];
    node->data_type = dtype;
    node->matrix_leading_dim = op.leading_dim;
    dag_->add_node(node);
}

void KernelGraphBuilder::add_prefetch_window(PrefetchWindow window) {
    require_started();
    assert(window.buffer.count >= 2 && window.buffer.slot >= 0 &&
           window.buffer.slot < window.buffer.count &&
           !window.consumers.empty());
    prefetch_windows_.push_back(std::move(window));
}

BuiltKernel KernelGraphBuilder::finish(const KernelEnd& end) {
    require_started();
#ifndef NDEBUG
    assert(end.op_count == dag_->nodes.size());
    assert(end.cta_count == dag_->cta_resources.size());
    assert(!dag_->nodes.empty() && !dag_->cta_resources.empty());
    assert(!end.outputs.empty());
    std::vector<std::uint8_t> output_flags(tensor_ids_.size(), 0);
    for (const int output : end.outputs) {
        assert(output >= 0 &&
               static_cast<std::size_t>(output) < tensor_ids_.size());
        auto& flags = output_flags[static_cast<std::size_t>(output)];
        assert(!(flags & 1));
        flags |= 1;
    }
    for (const int output : end.partial_outputs) {
        assert(output >= 0 &&
               static_cast<std::size_t>(output) < tensor_ids_.size());
        auto& flags = output_flags[static_cast<std::size_t>(output)];
        assert((flags & 1) && !(flags & 2));
        flags |= 2;
    }
    for (std::size_t output = 0; output < output_flags.size(); ++output) {
        const auto flags = output_flags[output];
        if (!(flags & 1)) {
            assert(output_areas_[output] == 0);
            continue;
        }
        const auto& tensor = global_tensors_.at(tensor_ids_[output]);
        assert(tensor.shape.size() == 2 && tensor.strides.size() == 2);
        const auto area = output_areas_[output];
        const auto full_area = tensor.shape[0] * tensor.shape[1];
        assert(area > 0 && area <= full_area &&
               ((flags & 2) || area == full_area));
    }

    for (const auto& window : prefetch_windows_) {
        for (const int consumer : window.consumers) {
            assert(consumer >= 0 &&
                   consumer < static_cast<int>(dag_->nodes.size()));
            assert(dag_->nodes[consumer]->op == "cp.dram2sram_tma");
        }
        for (const auto& range : window.ranges) {
            assert(range.tensor >= 0 &&
                   static_cast<std::size_t>(range.tensor) < tensor_ids_.size() &&
                   range.bytes != 0);
            const auto& tensor =
                global_tensors_.at(tensor_ids_[static_cast<std::size_t>(range.tensor)]);
            assert(range.offset <= tensor.byte_size() &&
                   range.bytes <= tensor.byte_size() - range.offset);
        }
    }
#else
    (void)end;
#endif

    BuiltKernel result{std::move(header_), std::move(dag_),
                       std::move(prefetch_windows_)};
    clear_bindings();
    return result;
}

void KernelGraphBuilder::abort() noexcept {
    dag_.reset();
    clear_bindings();
}

} // namespace GTSim::ExecutionIR
