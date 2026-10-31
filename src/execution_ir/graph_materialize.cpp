#include "execution_ir/kernel_graph.h"
#include <set>
#include <queue>
#include <regex>

namespace GTSim::ExecutionIR {
std::unique_ptr<DAG>
materialize(const Kernel& k, const SimulatorConfig& config,
            const TensorRegistry& registry,
            const std::vector<std::pair<int, int>>& bindings) {
    if (k.ops.empty() || k.ctas.empty() || k.tensors.empty())
        throw std::invalid_argument("empty execution kernel");
    auto dag = std::make_unique<DAG>();
    dag->tensor_registry = registry;
    std::map<int, int> ids(bindings.begin(), bindings.end());
    if (ids.size() != bindings.size() || ids.size() != k.tensors.size())
        throw std::invalid_argument("incomplete/duplicate tensor bindings");
    std::set<int> local_ids;
    std::set<std::string> local_names;
    for (const auto& t : k.tensors) {
        if (!local_ids.insert(t.id).second ||
            !local_names.insert(t.name).second)
            throw std::invalid_argument("duplicate kernel tensor");
        if (!ids.contains(t.id))
            throw std::invalid_argument("unbound kernel tensor");
        const auto& global = registry.at(ids.at(t.id));
        if (t.shape != global.shape || t.strides != global.strides ||
            t.dtype != "fp16" || t.tier != "HBM")
            throw std::invalid_argument("incompatible tensor binding");
    }
    for (const auto& c : k.ctas) {
        if (c.id < 0 || c.sm < 0 || c.sm >= config.num_sms || c.warps != 12 ||
            c.shared_bytes <= 0 ||
            c.shared_bytes > config.shared_memory_per_cta_bytes ||
            c.register_bytes <= 0 ||
            c.register_bytes > config.register_file_per_sm_bytes ||
            dag->cta_resources.contains(c.id))
            throw std::invalid_argument("invalid CTA resources or mapping");
        dag->cta_resources[c.id] = {c.shared_bytes, c.register_bytes, c.warps};
    }
    const std::map<std::string, std::string> supported = {
        {"cp.dram2sram_tma", "LD"},
        {"cp.sram2dram_tma", "ST"},
        {"st.reg2sram", "ST"},
        {"wgmma.m64n64k16", "Tensor"},
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
        {"acc_o_div", "SFU"}};
    std::map<int, int> sms;
    for (const auto& c : k.ctas) sms[c.id] = c.sm;
    std::vector<std::set<int>> deps(k.ops.size());
    std::vector<std::vector<int>> children(k.ops.size());
    std::set<int> output_ids(k.outputs.begin(), k.outputs.end());
    if (output_ids.empty() || output_ids.size() != k.outputs.size())
        throw std::invalid_argument("missing/duplicate output contract");
    for (int output : output_ids) {
        if (!ids.contains(output))
            throw std::invalid_argument("unknown output tensor");
        const auto& tensor = registry.at(ids.at(output));
        std::vector<std::array<int, 4>> regions;
        std::uint64_t area = 0;
        for (const auto& o : k.ops)
            if (o.op == "cp.sram2dram_tma" && o.tensor == output) {
                if (o.tile[2] <= 0 || o.tile[3] <= 0)
                    throw std::invalid_argument("invalid output region");
                area += std::uint64_t(o.tile[2]) * o.tile[3];
                regions.push_back(o.tile);
            }
        if (area != tensor.shape[0] * tensor.shape[1])
            throw std::invalid_argument("incomplete output coverage");
        for (std::size_t i = 0; i < regions.size(); ++i)
            for (std::size_t j = i + 1; j < regions.size(); ++j) {
                const auto& a = regions[i];
                const auto& b = regions[j];
                if (std::max(a[0], b[0]) <
                        std::min(std::int64_t(a[0]) + a[2],
                                 std::int64_t(b[0]) + b[2]) &&
                    std::max(a[1], b[1]) < std::min(std::int64_t(a[1]) + a[3],
                                                    std::int64_t(b[1]) + b[3]))
                    throw std::invalid_argument("overlapping output stores");
            }
    }
    const std::regex warp_pattern("_w[0-9]+");
    std::map<std::tuple<int, int, std::string>, std::vector<int>> collectives;
    std::set<std::string> instruction_names;
    std::set<int> read_allocations, write_allocations;
    for (const auto& o : k.ops) {
        if (!instruction_names.insert(o.name).second)
            throw std::invalid_argument("duplicate instruction name");
        if (o.op == "wgmma.m64n64k16")
            collectives[{o.cta, o.warp / 4,
                         std::regex_replace(o.name, warp_pattern, "_w")}]
                .push_back(o.warp % 4);
        if (o.id != static_cast<int>(dag->nodes.size()) ||
            !sms.contains(o.cta) || o.warp < o.cta * 12 ||
            o.warp >= o.cta * 12 + 12 || !supported.contains(o.op) ||
            supported.at(o.op) != o.pipeline ||
            (o.dtype != "fp16" && o.dtype != "fp32") || o.tile[0] < 0 ||
            o.tile[1] < 0 || o.tile[2] <= 0 || o.tile[3] <= 0)
            throw std::invalid_argument("invalid native instruction " + o.name);
        for (const auto* list : {&o.after_issue, &o.after_complete})
            for (int d : *list) {
                if (d < 0 || d >= static_cast<int>(k.ops.size()) || d == o.id)
                    throw std::invalid_argument("dangling dependency");
                deps[o.id].insert(d);
            }
        const bool global =
            o.op == "cp.dram2sram_tma" || o.op == "cp.sram2dram_tma";
        if (o.op == "cp.sram2dram_tma" && !output_ids.contains(o.tensor))
            throw std::invalid_argument("undeclared output store");
        if (global) {
            if (!ids.contains(o.tensor))
                throw std::invalid_argument("unbound memory tensor");
            if (o.op == "cp.dram2sram_tma")
                read_allocations.insert(ids.at(o.tensor));
            else
                write_allocations.insert(ids.at(o.tensor));
            const auto& t = registry.at(ids.at(o.tensor));
            if (t.shape.size() != 2 ||
                std::uint64_t(o.tile[0]) + o.tile[2] > t.shape[0] ||
                std::uint64_t(o.tile[1]) + o.tile[3] > t.shape[1] ||
                o.leading_dim != static_cast<int>(t.strides[0]) ||
                (o.op == "cp.sram2dram_tma" && t.read_only))
                throw std::invalid_argument(
                    "invalid memory region/visibility binding");
        }
        auto* n = new DAGNode(o.id, o.name, o.pipeline, o.op, o.warp,
                              o.after_complete, 0,
                              Tile(o.tile[0], o.tile[1], o.tile[2], o.tile[3]));
        n->issue_depends_on = o.after_issue;
        n->sm_id = sms.at(o.cta);
        n->thread_block_id = o.cta;
        n->matrix_id = ids.contains(o.tensor) ? ids.at(o.tensor) : o.tensor;
        n->data_type = o.dtype == "fp32" ? DataType::FP32 : DataType::FP16;
        n->matrix_leading_dim = o.leading_dim;
        dag->add_node(n);
    }
    for (int id : write_allocations)
        if (read_allocations.contains(id))
            throw std::invalid_argument(
                "input/output allocation alias is unsupported");
    for (auto& [key, lanes] : collectives) {
        (void)key;
        std::sort(lanes.begin(), lanes.end());
        if (lanes != std::vector<int>{0, 1, 2, 3})
            throw std::invalid_argument("incomplete WGMMA collective");
    }
    std::queue<int> ready;
    for (std::size_t i = 0; i < deps.size(); ++i) {
        if (deps[i].empty()) ready.push(static_cast<int>(i));
        for (int d : deps[i]) children[d].push_back(static_cast<int>(i));
    }
    std::size_t visited = 0;
    while (!ready.empty()) {
        int n = ready.front();
        ready.pop();
        ++visited;
        for (int child : children[n]) {
            deps[child].erase(n);
            if (deps[child].empty()) ready.push(child);
        }
    }
    if (visited != k.ops.size())
        throw std::invalid_argument("kernel dependency cycle");
    return dag;
}
} // namespace GTSim::ExecutionIR
