#include "execution_ir/kernel_graph_builder.h"

#include <algorithm>
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
    throw std::invalid_argument("unsupported execution dtype " + value);
}

const std::map<std::string, std::string>& KernelGraphBuilder::supported_ops() {
    static const std::map<std::string, std::string> values = {
        {"cp.dram2sram_tma", "LD"},
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
    if (!dag_) throw std::logic_error("kernel graph builder is not active");
}

void KernelGraphBuilder::begin(KernelHeader header) {
    if (dag_) throw std::logic_error("concurrent kernel graph construction");
    if (header.launch_id.empty() || header.stream_id.empty())
        throw std::invalid_argument("invalid kernel header");
    std::map<int, int> ids;
    for (const auto& [local, global] : header.tensor_bindings) {
        if (local < 0 || !ids.emplace(local, global).second)
            throw std::invalid_argument("duplicate kernel tensor binding");
        (void)global_tensors_.at(global);
    }
    if (ids.empty())
        throw std::invalid_argument("kernel has no tensor bindings");

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
    read_allocations_.clear();
    write_allocations_.clear();
    output_areas_.clear();
    prefetch_consumers_.clear();
}

void KernelGraphBuilder::add_cta(const CTA& cta) {
    require_started();
    if (cta.id < 0 || cta.sm < 0 || cta.sm >= config_.num_sms ||
        cta.warps < 1 || cta.warps > config_.max_warps_per_sm ||
        cta.warps > 32 || cta.shared_bytes <= 0 ||
        cta.shared_bytes > config_.shared_memory_per_cta_bytes ||
        cta.register_bytes <= 0 ||
        cta.register_bytes > config_.register_file_per_sm_bytes ||
        cta_sms_.contains(cta.id))
        throw std::invalid_argument("invalid CTA resources or mapping");
    const auto desired =
        static_cast<std::uint64_t>(sm_next_subpartition_[cta.sm] % 4);
    next_warp_id_ += static_cast<int>(
        (desired + 4 - static_cast<std::uint64_t>(next_warp_id_) % 4) % 4);
    if (next_warp_id_ > std::numeric_limits<int>::max() - cta.warps)
        throw std::overflow_error("execution warp numbering exceeds int32");
    cta_sms_[cta.id] = cta.sm;
    cta_warps_[cta.id] = cta.warps;
    warp_bases_[cta.id] = next_warp_id_;
    next_warp_id_ += cta.warps;
    sm_next_subpartition_[cta.sm] =
        (sm_next_subpartition_[cta.sm] + cta.warps) % 4;
    dag_->cta_resources[cta.id] = {cta.shared_bytes, cta.register_bytes,
                                   cta.warps};
}

void KernelGraphBuilder::add_instruction(const Instruction& op) {
    require_started();
    const int expected = static_cast<int>(dag_->nodes.size());
    const auto supported = supported_ops().find(op.op);
    if (op.id != expected || op.name.empty() || !cta_sms_.contains(op.cta) ||
        op.warp < warp_bases_.at(op.cta) ||
        op.warp >= warp_bases_.at(op.cta) + cta_warps_.at(op.cta) ||
        supported == supported_ops().end() ||
        supported->second != op.pipeline ||
        (op.dtype != "fp16" && op.dtype != "bf16" && op.dtype != "fp32" &&
         op.dtype != "int32") ||
        op.tile[0] < 0 || op.tile[1] < 0 || op.tile[2] <= 0 || op.tile[3] <= 0)
        throw std::invalid_argument("invalid native instruction " + op.name);

    for (const auto* list : {&op.after_issue, &op.after_complete})
        for (const int dep : *list)
            if (dep < 0 || dep >= op.id)
                throw std::invalid_argument(
                    "dangling/forward instruction dependency");

    const bool global =
        op.op == "cp.dram2sram_tma" || op.op == "cp.sram2dram_tma";
    if (global) {
        if (!tensor_ids_.contains(op.tensor))
            throw std::invalid_argument("unbound memory tensor");
        const auto& tensor = global_tensors_.at(tensor_ids_.at(op.tensor));
        if (tensor.shape.size() != 2 ||
            std::uint64_t(op.tile[0]) + op.tile[2] > tensor.shape[0] ||
            std::uint64_t(op.tile[1]) + op.tile[3] > tensor.shape[1] ||
            op.leading_dim != static_cast<int>(tensor.strides[0]) ||
            parse_dtype(op.dtype) != tensor.dtype ||
            (op.op == "cp.sram2dram_tma" && tensor.read_only))
            throw std::invalid_argument(
                "invalid memory region/visibility binding");
        const int global_id = tensor_ids_.at(op.tensor);
        if (op.op == "cp.dram2sram_tma")
            read_allocations_.insert(global_id);
        else {
            write_allocations_.insert(global_id);
            output_areas_[op.tensor] +=
                static_cast<std::uint64_t>(op.tile[2]) * op.tile[3];
        }
    }

    auto* node = new DAGNode(
        op.id, op.name, op.pipeline, op.op, op.warp, op.after_complete, 0,
        Tile(op.tile[0], op.tile[1], op.tile[2], op.tile[3]),
        parse_dtype(op.dtype));
    node->issue_depends_on = op.after_issue;
    node->sm_id = cta_sms_.at(op.cta);
    node->thread_block_id = op.cta;
    node->matrix_id =
        tensor_ids_.contains(op.tensor) ? tensor_ids_.at(op.tensor) : op.tensor;
    node->data_type = parse_dtype(op.dtype);
    node->matrix_leading_dim = op.leading_dim;
    dag_->add_node(node);
}

void KernelGraphBuilder::add_prefetch_window(PrefetchWindow window) {
    require_started();
    if (window.buffer.count < 2 || window.buffer.slot < 0 ||
        window.buffer.slot >= window.buffer.count || window.consumers.empty())
        throw std::invalid_argument("invalid streaming prefetch window");
    prefetch_windows_.push_back(std::move(window));
}

BuiltKernel KernelGraphBuilder::finish(const KernelEnd& end) {
    require_started();
    if (end.op_count != dag_->nodes.size() ||
        end.cta_count != dag_->cta_resources.size() || dag_->nodes.empty() ||
        dag_->cta_resources.empty())
        throw std::invalid_argument(
            "kernel end count does not match streamed graph");

    std::set<int> outputs(end.outputs.begin(), end.outputs.end());
    std::set<int> partial(end.partial_outputs.begin(),
                          end.partial_outputs.end());
    if (outputs.empty() || outputs.size() != end.outputs.size() ||
        partial.size() != end.partial_outputs.size() ||
        !std::all_of(partial.begin(), partial.end(),
                     [&](int id) { return outputs.contains(id); }))
        throw std::invalid_argument("invalid output contract");
    for (const int output : outputs) {
        if (!tensor_ids_.contains(output))
            throw std::invalid_argument("unknown output tensor");
        const auto& tensor = global_tensors_.at(tensor_ids_.at(output));
        if (tensor.shape.size() != 2 || tensor.strides.size() != 2)
            throw std::invalid_argument(
                "output contract requires a 2-D tensor");
        const auto area = output_areas_[output];
        const auto full_area = tensor.shape[0] * tensor.shape[1];
        if (area == 0 || area > full_area ||
            (!partial.contains(output) && area != full_area))
            throw std::invalid_argument("incomplete output coverage");
    }
    for (const auto& [id, area] : output_areas_)
        if (!outputs.contains(id) || area == 0)
            throw std::invalid_argument("undeclared output store");

    for (const auto& window : prefetch_windows_) {
        for (const int consumer : window.consumers) {
            if (consumer < 0 ||
                consumer >= static_cast<int>(dag_->nodes.size()) ||
                dag_->nodes[consumer]->op != "cp.dram2sram_tma" ||
                !prefetch_consumers_.insert(consumer).second)
                throw std::invalid_argument("invalid prefetch consumer");
        }
        for (const auto& range : window.ranges) {
            if (!tensor_ids_.contains(range.tensor) || range.bytes == 0)
                throw std::invalid_argument("invalid prefetch range");
            const auto& tensor =
                global_tensors_.at(tensor_ids_.at(range.tensor));
            if (range.offset > tensor.byte_size() ||
                range.bytes > tensor.byte_size() - range.offset)
                throw std::invalid_argument("prefetch range exceeds tensor");
        }
    }
    for (const int id : write_allocations_)
        if (read_allocations_.contains(id))
            throw std::invalid_argument(
                "input/output allocation alias is unsupported");

    BuiltKernel result{std::move(header_), std::move(dag_),
                       std::move(prefetch_windows_)};
    tensor_ids_.clear();
    cta_sms_.clear();
    cta_warps_.clear();
    warp_bases_.clear();
    read_allocations_.clear();
    write_allocations_.clear();
    output_areas_.clear();
    prefetch_consumers_.clear();
    return result;
}

void KernelGraphBuilder::abort() noexcept {
    dag_.reset();
    tensor_ids_.clear();
    cta_sms_.clear();
    cta_warps_.clear();
    warp_bases_.clear();
    prefetch_windows_.clear();
    read_allocations_.clear();
    write_allocations_.clear();
    output_areas_.clear();
    prefetch_consumers_.clear();
}

} // namespace GTSim::ExecutionIR
