#include "execution_ir/kernel_graph_builder.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <regex>

namespace GTSim::ExecutionIR {

KernelGraphBuilder::KernelGraphBuilder(const TensorRegistry& global_tensors,
                                       const SimulatorConfig& config)
    : global_tensors_(global_tensors), config_(config),
      sm_next_subpartition_(std::max(0, config.num_sms), 0) {}

DataType KernelGraphBuilder::parse_dtype(const std::string& value) {
    if (value == "fp16") return DataType::FP16;
    if (value == "bf16") return DataType::BF16;
    if (value == "fp32") return DataType::FP32;
    if (value == "int32") return DataType::INT32;
    throw std::invalid_argument("unsupported execution dtype " + value);
}

const std::map<std::string, std::string>&
KernelGraphBuilder::supported_ops() {
    static const std::map<std::string, std::string> values = {
        {"cp.dram2sram_tma", "LD"}, {"cp.sram2dram_tma", "ST"},
        {"st.reg2sram", "ST"},       {"wgmma.m64n64k16", "Tensor"},
        {"mma.sync.m16n8k16", "Tensor"},
        {"local_max", "SIMD"},       {"reduce_max", "SIMD"},
        {"merge_max", "SIMD"},       {"local_sum", "SIMD"},
        {"reduce_sum", "SIMD"},      {"merge_sum", "SIMD"},
        {"logsum", "SIMD"},          {"acc_o_scale", "SIMD"},
        {"acc_s_cast", "SIMD"},      {"shfl_max", "SHFL"},
        {"shfl_sum", "SHFL"},        {"scores_scale", "SFU"},
        {"exp2", "SFU"},              {"acc_o_div", "SFU"},
        {"simd.add", "SIMD"},        {"simd.sub", "SIMD"},
        {"simd.mul", "SIMD"},         {"simd.fma", "SIMD"},
        {"simd.max", "SIMD"},         {"simd.min", "SIMD"},
        {"simd.cast", "SIMD"},        {"simd.fill", "SIMD"},
        {"simd.identity", "SIMD"},    {"sfu.exp", "SFU"},
        {"sfu.log", "SFU"},            {"sfu.sqrt", "SFU"},
        {"sfu.rsqrt", "SFU"},          {"sfu.divide", "SFU"},
        {"sfu.sin", "SFU"},            {"sfu.cos", "SFU"},
        {"shfl", "SHFL"}};
    return values;
}

void KernelGraphBuilder::require_started() const {
    if (!dag_) throw std::logic_error("kernel graph builder is not active");
}

void KernelGraphBuilder::begin(KernelHeader header) {
    if (dag_) throw std::logic_error("concurrent kernel graph construction");
    if (header.launch_id.empty() || header.stream_id.empty() ||
        header.signal_token.empty())
        throw std::invalid_argument("invalid kernel header");
    std::map<int, int> ids;
    for (const auto& [local, global] : header.tensor_bindings) {
        if (!ids.emplace(local, global).second || local < 0)
            throw std::invalid_argument("duplicate kernel tensor binding");
        (void)global_tensors_.at(global);
    }
    if (ids.empty()) throw std::invalid_argument("kernel has no tensor bindings");

    header_ = std::move(header);
    tensor_ids_ = std::move(ids);
    dag_ = std::make_unique<DAG>();
    dag_->use_tensor_registry(global_tensors_);
    cta_sms_.clear();
    cta_warps_.clear();
    warp_bases_.clear();
    sm_next_subpartition_.assign(std::max(0, config_.num_sms), 0);
    next_warp_id_ = 0;
    deps_.clear();
    children_.clear();
    prefetch_windows_.clear();
    instruction_names_.clear();
    read_allocations_.clear();
    write_allocations_.clear();
    collectives_.clear();
    output_regions_.clear();
    opcodes_.clear();
    prefetch_consumers_.clear();
    outputs_.clear();
}

void KernelGraphBuilder::add_cta(const CTA& cta) {
    require_started();
    if (cta.id < 0 || cta.sm < 0 || cta.sm >= config_.num_sms || cta.warps < 1 ||
        cta.warps > config_.max_warps_per_sm || cta.warps > 32 ||
        cta.shared_bytes <= 0 ||
        cta.shared_bytes > config_.shared_memory_per_cta_bytes ||
        cta.register_bytes <= 0 ||
        cta.register_bytes > config_.register_file_per_sm_bytes ||
        cta_sms_.contains(cta.id))
        throw std::invalid_argument("invalid CTA resources or mapping");
    const auto desired = static_cast<std::uint64_t>(
        sm_next_subpartition_[cta.sm] % 4);
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
    dag_->cta_resources[cta.id] =
        {cta.shared_bytes, cta.register_bytes, cta.warps};
}

void KernelGraphBuilder::add_instruction(const Instruction& op) {
    require_started();
    const int expected = static_cast<int>(dag_->nodes.size());
    if (op.id != expected || op.name.empty() ||
        !instruction_names_.insert(op.name).second ||
        !cta_sms_.contains(op.cta) ||
        op.warp < warp_bases_.at(op.cta) ||
        op.warp >= warp_bases_.at(op.cta) + cta_warps_.at(op.cta) ||
        !supported_ops().contains(op.op) || supported_ops().at(op.op) != op.pipeline ||
        (op.dtype != "fp16" && op.dtype != "bf16" && op.dtype != "fp32" &&
         op.dtype != "int32") ||
        op.tile[0] < 0 || op.tile[1] < 0 || op.tile[2] <= 0 || op.tile[3] <= 0)
        throw std::invalid_argument("invalid native instruction " + op.name);

    auto& deps = deps_.emplace_back();
    children_.emplace_back();
    for (const auto* list : {&op.after_issue, &op.after_complete}) {
        for (const int dep : *list) {
            if (dep < 0 || dep >= op.id)
                throw std::invalid_argument("dangling/forward instruction dependency");
            deps.insert(dep);
        }
    }

    const bool global = op.op == "cp.dram2sram_tma" ||
                        op.op == "cp.sram2dram_tma";
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
            throw std::invalid_argument("invalid memory region/visibility binding");
        if (op.op == "cp.dram2sram_tma")
            read_allocations_.insert(tensor_ids_.at(op.tensor));
        else {
            write_allocations_.insert(tensor_ids_.at(op.tensor));
            output_regions_[op.tensor].push_back(op.tile);
        }
    }

    if (op.op == "wgmma.m64n64k16") {
        static const std::regex warp_pattern("_w[0-9]+");
        collectives_[{op.cta, op.warp / 4,
                      std::regex_replace(op.name, warp_pattern, "_w")}]
            .push_back(op.warp % 4);
    }
    auto* node = new DAGNode(op.id, op.name, op.pipeline, op.op, op.warp,
                             op.after_complete, 0,
                             Tile(op.tile[0], op.tile[1], op.tile[2], op.tile[3]),
                             parse_dtype(op.dtype));
    node->issue_depends_on = op.after_issue;
    node->sm_id = cta_sms_.at(op.cta);
    node->thread_block_id = op.cta;
    node->matrix_id = tensor_ids_.contains(op.tensor) ? tensor_ids_.at(op.tensor)
                                                      : op.tensor;
    node->data_type = parse_dtype(op.dtype);
    node->matrix_leading_dim = op.leading_dim;
    dag_->add_node(node);
    opcodes_.emplace(op.id, op.op);
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
    if (end.op_count != dag_->nodes.size() || end.cta_count != dag_->cta_resources.size() ||
        dag_->nodes.empty() || dag_->cta_resources.empty())
        throw std::invalid_argument("kernel end count does not match streamed graph");

    std::set<int> outputs(end.outputs.begin(), end.outputs.end());
    std::set<int> partial(end.partial_outputs.begin(), end.partial_outputs.end());
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
            throw std::invalid_argument("output contract requires a 2-D tensor");
        std::uint64_t area = 0;
        const auto regions = output_regions_.find(output);
        if (regions != output_regions_.end()) {
            for (const auto& tile : regions->second) {
                area += std::uint64_t(tile[2]) * tile[3];
                if (std::uint64_t(tile[0]) + tile[2] > tensor.shape[0] ||
                    std::uint64_t(tile[1]) + tile[3] > tensor.shape[1])
                    throw std::invalid_argument("output region out of bounds");
            }
        }
        const auto full_area = tensor.shape[0] * tensor.shape[1];
        if (area == 0 || area > full_area ||
            (!partial.contains(output) && area != full_area))
            throw std::invalid_argument("incomplete output coverage");
    }
    for (const auto& [id, regions] : output_regions_) {
        if (!outputs.contains(id))
            throw std::invalid_argument("undeclared output store");
        for (std::size_t i = 0; i < regions.size(); ++i)
            for (std::size_t j = i + 1; j < regions.size(); ++j) {
                const auto& a = regions[i];
                const auto& b = regions[j];
                if (std::max(a[0], b[0]) <
                        std::min(std::int64_t(a[0]) + a[2],
                                 std::int64_t(b[0]) + b[2]) &&
                    std::max(a[1], b[1]) <
                        std::min(std::int64_t(a[1]) + a[3],
                                 std::int64_t(b[1]) + b[3]))
                    throw std::invalid_argument("overlapping output stores");
            }
    }
    for (const auto& window : prefetch_windows_) {
        for (const int consumer : window.consumers) {
            if (!opcodes_.contains(consumer) ||
                opcodes_.at(consumer) != "cp.dram2sram_tma" ||
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
            throw std::invalid_argument("input/output allocation alias is unsupported");

    for (auto& [unused, lanes] : collectives_) {
        (void)unused;
        std::sort(lanes.begin(), lanes.end());
        if (lanes != std::vector<int>{0, 1, 2, 3})
            throw std::invalid_argument("incomplete WGMMA collective");
    }
    for (std::size_t i = 0; i < deps_.size(); ++i)
        for (const int dep : deps_[i]) children_[dep].push_back(static_cast<int>(i));
    auto pending = deps_;
    std::queue<int> ready;
    for (std::size_t i = 0; i < pending.size(); ++i)
        if (pending[i].empty()) ready.push(static_cast<int>(i));
    std::size_t visited = 0;
    while (!ready.empty()) {
        const int current = ready.front();
        ready.pop();
        ++visited;
        for (const int child : children_[current])
            if (pending[child].erase(current) && pending[child].empty())
                ready.push(child);
    }
    if (visited != dag_->nodes.size())
        throw std::invalid_argument("kernel dependency cycle");
    dag_->build_dependency_graph();

    BuiltKernel result{std::move(header_), std::move(dag_),
                       std::move(prefetch_windows_)};
    tensor_ids_.clear();
    cta_sms_.clear();
    cta_warps_.clear();
    warp_bases_.clear();
    deps_.clear();
    children_.clear();
    instruction_names_.clear();
    read_allocations_.clear();
    write_allocations_.clear();
    collectives_.clear();
    output_regions_.clear();
    outputs_.clear();
    opcodes_.clear();
    prefetch_consumers_.clear();
    return result;
}

void KernelGraphBuilder::abort() noexcept {
    dag_.reset();
    tensor_ids_.clear();
    cta_sms_.clear();
    cta_warps_.clear();
    warp_bases_.clear();
    deps_.clear();
    children_.clear();
    prefetch_windows_.clear();
    instruction_names_.clear();
    read_allocations_.clear();
    write_allocations_.clear();
    collectives_.clear();
    output_regions_.clear();
    outputs_.clear();
    opcodes_.clear();
    prefetch_consumers_.clear();
}
} // namespace GTSim::ExecutionIR
