#include "tile_ir.h"
#include "simulator.h"
#include <limits>
#include <map>
#include <queue>
#include <sstream>
#include <unordered_set>

namespace GTSim {
std::string describe_tile_ir_hardware(const SimulatorConfig& config) {
    std::ostringstream output;
    output << "{\"num_sms\":" << config.num_sms
           << ",\"shared_memory_per_cta_bytes\":" << config.shared_memory_per_cta_bytes
           << ",\"shared_memory_per_sm_bytes\":" << config.shared_memory_per_sm_bytes
           << ",\"register_file_per_sm_bytes\":" << config.register_file_per_sm_bytes
           << ",\"max_warps_per_sm\":" << config.max_warps_per_sm
           << ",\"max_concurrent_blocks_per_sm\":" << config.max_concurrent_blocks_per_sm
           << ",\"topology\":\"" << memory_topology_name(config.memory.topology) << "\"}";
    return output.str();
}

std::unique_ptr<DAG> materialize_tile_ir(const TileIR& ir, const SimulatorConfig& config) {
    if (ir.tensors.empty() || ir.ctas.empty() || ir.operations.empty())
        throw std::invalid_argument("Tile IR requires tensors, CTAs, and operations");
    auto dag = std::make_unique<DAG>();
    std::unordered_set<std::string> names;
    for (const auto& t : ir.tensors) {
        if (t.id < 0 || t.name.empty() || !names.insert(t.name).second || t.shape.size() != 2 ||
            t.strides.size() != 2 || t.shape[0] == 0 || t.shape[1] == 0 ||
            t.strides[1] != 1 || t.strides[0] < t.shape[1] ||
            t.strides[0] > std::numeric_limits<int>::max() ||
            (t.dtype != DataType::FP16 && t.dtype != DataType::FP32))
            throw std::invalid_argument("unsupported tensor metadata");
        if ((config.memory.topology == MemoryTopology::FullHBM && t.tier == MemoryTier::HBF) ||
            (config.memory.topology == MemoryTopology::FullHBF && t.tier == MemoryTier::HBM))
            throw std::invalid_argument("tensor device is unavailable in selected topology");
        dag->tensor_registry.allocate_device_tensor(t.id, t.name, t.shape, t.strides, t.dtype, t.tier);
    }
    std::unordered_map<int, TileIRCTA> ctas;
    for (const auto& c : ir.ctas) {
        const auto& r = c.resources;
        if (c.id < 0 || c.id > (std::numeric_limits<int>::max() - 11) / 12 ||
            !ctas.emplace(c.id, c).second || c.sm < 0 || c.sm >= config.num_sms ||
            r.warps != 12 || r.shared_bytes < 0 || r.register_bytes <= 0 ||
            r.shared_bytes > config.shared_memory_per_cta_bytes ||
            r.shared_bytes > config.shared_memory_per_sm_bytes ||
            r.register_bytes > config.register_file_per_sm_bytes ||
            r.register_bytes > r.warps * 32 * 255 * 4 || r.warps > config.max_warps_per_sm)
            throw std::invalid_argument("invalid CTA mapping or resource requirements");
        dag->cta_resources[c.id] = r;
    }
    const std::unordered_map<std::string, std::string> pipelines{
        {"ld.dram2reg", "LD"}, {"st.reg2dram", "ST"},
        {"cp.dram2sram_tma", "LD"}, {"cp.sram2dram_tma", "ST"},
        {"st.reg2sram", "ST"}, {"wgmma.m64n64k16", "Tensor"}};
    names.clear();
    std::unordered_set<int> used_ctas;
    std::map<std::pair<int, int>, std::array<std::vector<int>, 4>> collectives;
    for (const auto& op : ir.operations) {
        if (op.id != static_cast<int>(dag->nodes.size()) || !names.insert(op.name).second || op.name.empty())
            throw std::invalid_argument("operation IDs must be dense, ordered, and uniquely named");
        if (!pipelines.contains(op.op)) throw std::invalid_argument("unsupported operation '" + op.op + "'");
        if (!ctas.contains(op.cta) || op.warp < 0 || op.warp >= 12)
            throw std::invalid_argument("invalid CTA/warp reference");
        used_ctas.insert(op.cta);
        const auto& tensor = dag->tensor_registry.at(op.tensor);
        const auto& tile = op.region;
        if (tile.ndims != 2 || tile.dims[0] != tile.r || tile.dims[1] != tile.c ||
            tile.offs[0] != tile.r_off || tile.offs[1] != tile.c_off ||
            tile.r_off < 0 || tile.c_off < 0 || tile.r <= 0 || tile.c <= 0 ||
            tile.r > std::numeric_limits<int>::max() / tile.c ||
            static_cast<std::uint64_t>(tile.r_off) + tile.r > tensor.shape[0] ||
            static_cast<std::uint64_t>(tile.c_off) + tile.c > tensor.shape[1])
            throw std::invalid_argument("out-of-bounds access region at %" + std::to_string(op.id));
        bool wgmma = op.op == "wgmma.m64n64k16";
        if ((wgmma && (op.dtype != DataType::FP32 || tensor.dtype != DataType::FP16 ||
                       tile.r != 16 || tile.c != 64 || op.warp >= 8)) ||
            (!wgmma && op.dtype != tensor.dtype))
            throw std::invalid_argument("operation dtype or WGMMA region is unsupported");
        if (wgmma) {
            collectives[{op.cta, op.warp / 4}][op.warp % 4].push_back(op.id);
        }
        auto node = std::make_unique<DAGNode>(op.id, op.name, pipelines.at(op.op), op.op,
            op.cta * 12 + op.warp, op.after_complete, 0, tile, op.dtype);
        node->sm_id = ctas.at(op.cta).sm;
        node->thread_block_id = op.cta;
        node->matrix_id = op.tensor;
        // The reference uses a leading dimension only for global accesses.
        if (op.op != "st.reg2sram" && !wgmma)
            node->matrix_leading_dim = static_cast<int>(tensor.strides[0]);
        node->issue_depends_on = op.after_issue;
        for (const auto* edges : {&op.after_complete, &op.after_issue}) {
            for (int parent : *edges) {
                if (parent < 0 || parent >= static_cast<int>(ir.operations.size()) || parent == op.id)
                    throw std::invalid_argument("invalid dependency reference at %" + std::to_string(op.id));
            }
        }
        dag->add_node(node.release());
    }
    if (used_ctas.size() != ctas.size()) throw std::invalid_argument("CTA has no operations");
    // A collective's nth operation must be the same instruction in each warp.
    // Collapse its four fragments when checking for cycles: an acyclic scalar
    // graph could otherwise deadlock by waiting on a sibling fragment.
    std::vector<int> representative(ir.operations.size());
    for (std::size_t i = 0; i < representative.size(); ++i) representative[i] = static_cast<int>(i);
    std::size_t vertices = ir.operations.size();
    for (const auto& [key, lanes] : collectives) {
        (void)key;
        if (lanes[0].empty() || lanes[1].size() != lanes[0].size() ||
            lanes[2].size() != lanes[0].size() || lanes[3].size() != lanes[0].size())
            throw std::invalid_argument("incomplete four-warp WGMMA collective");
        for (std::size_t index = 0; index < lanes[0].size(); ++index) {
            const auto& leader = ir.operations[lanes[0][index]];
            for (int lane = 1; lane < 4; ++lane) {
                const auto& member = ir.operations[lanes[lane][index]];
                if (member.tensor != leader.tensor || member.region.c_off != leader.region.c_off ||
                    member.region.r_off != leader.region.r_off + lane * 16)
                    throw std::invalid_argument("mismatched WGMMA collective regions/order");
                representative[member.id] = leader.id;
                --vertices;
            }
        }
    }
    std::vector<int> pending(ir.operations.size(), 0);
    std::vector<std::vector<int>> children(ir.operations.size());
    for (const auto& op : ir.operations) {
        for (const auto* edges : {&op.after_complete, &op.after_issue}) {
            for (int parent : *edges) {
                int from = representative[parent], to = representative[op.id];
                if (from == to) throw std::invalid_argument("dependency within a WGMMA collective");
                children[from].push_back(to);
                ++pending[to];
            }
        }
    }
    std::queue<int> ready;
    for (std::size_t i = 0; i < pending.size(); ++i)
        if (representative[i] == static_cast<int>(i) && pending[i] == 0) ready.push(static_cast<int>(i));
    std::size_t visited = 0;
    while (!ready.empty()) {
        int id = ready.front();
        ready.pop();
        ++visited;
        for (int child : children[id]) if (--pending[child] == 0) ready.push(child);
    }
    if (visited != vertices) throw std::invalid_argument("dependency cycle in Tile IR, including collective waits");
    return dag;
}
} // namespace GTSim
