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
           << ",\"core_frequency_mhz\":" << config.core_frequency_mhz
           << ",\"simd_width\":" << config.simd_width
           << ",\"simd_throughput\":" << config.simd_throughput
           << ",\"topology\":\"" << memory_topology_name(config.memory.topology) << "\""
           << ",\"hbm_capacity_bytes\":" << config.memory.hbm.capacity_bytes
           << ",\"hbf_capacity_bytes\":" << config.memory.hbf.capacity_bytes
           << ",\"hbm_read_bw_GBps\":" << config.memory.hbm.read_bw_GBps
           << ",\"hbf_read_bw_GBps\":" << config.memory.hbf.read_bw_GBps
           << "}";
    return output.str();
}

std::unique_ptr<DAG> materialize_tile_ir(const TileIR& ir, const SimulatorConfig& config) {
    if (ir.version != 1 && ir.version != 2) throw std::invalid_argument("unsupported Tile IR version");
    if (ir.tensors.empty() || ir.ctas.empty() || ir.operations.empty())
        throw std::invalid_argument("Tile IR requires tensors, CTAs, and operations");
    auto dag = std::make_unique<DAG>();
    std::unordered_set<std::string> names;
    std::uint64_t allocated[2]{};
    for (const auto& t : ir.tensors) {
        if (t.id < 0 || t.name.empty() || !names.insert(t.name).second || t.shape.size() != 2 ||
            t.strides.size() != 2 || t.shape[0] == 0 || t.shape[1] == 0 ||
            t.strides[1] != 1 || t.strides[0] < t.shape[1] ||
            t.strides[0] > std::numeric_limits<int>::max() ||
            (t.dtype != DataType::FP16 && t.dtype != DataType::FP32 &&
             !(ir.version == 2 && t.dtype == DataType::BF16)))
            throw std::invalid_argument("unsupported tensor metadata");
        if ((config.memory.topology == MemoryTopology::FullHBM && t.tier == MemoryTier::HBF) ||
            (config.memory.topology == MemoryTopology::FullHBF && t.tier == MemoryTier::HBM))
            throw std::invalid_argument("tensor device is unavailable in selected topology");
        dag->tensor_registry.allocate_device_tensor(t.id, t.name, t.shape, t.strides, t.dtype, t.tier);
        const int device = t.tier == MemoryTier::HBM ? 0 : 1;
        const auto bytes = dag->tensor_registry.at(t.id).byte_size();
        if (bytes > std::numeric_limits<std::uint64_t>::max() - allocated[device])
            throw std::overflow_error("device allocation size overflows");
        allocated[device] += bytes;
        const auto capacity = device == 0 ? config.memory.hbm.capacity_bytes : config.memory.hbf.capacity_bytes;
        if (capacity && allocated[device] > capacity)
            throw std::invalid_argument("tensor allocations exceed configured device capacity");
    }
    const int warp_stride = ir.version == 1 ? 12 : 32;
    std::unordered_map<int, TileIRCTA> ctas;
    std::unordered_map<int, int> warp_bases;
    std::vector<int> sm_next_subpartition(config.num_sms, 0);
    std::uint64_t next_warp_id = 0;
    for (const auto& c : ir.ctas) {
        const auto& r = c.resources;
        if (c.id < 0 || c.id > (std::numeric_limits<int>::max() - (warp_stride - 1)) / warp_stride ||
            !ctas.emplace(c.id, c).second || c.sm < 0 || c.sm >= config.num_sms ||
            (ir.version == 1 ? r.warps != 12 : r.warps < 1 || r.warps > 32) ||
            r.shared_bytes < 0 || r.register_bytes <= 0 ||
            r.shared_bytes > config.shared_memory_per_cta_bytes ||
            r.shared_bytes > config.shared_memory_per_sm_bytes ||
            r.register_bytes > config.register_file_per_sm_bytes ||
            r.register_bytes > r.warps * 32 * 255 * 4 || r.warps > config.max_warps_per_sm)
            throw std::invalid_argument("invalid CTA mapping or resource requirements");
        dag->cta_resources[c.id] = r;
        if (ir.version == 1) {
            warp_bases[c.id] = c.id * 12;
        } else {
            // Warp IDs determine subpartition via modulo four. Number each
            // SM's resident/queued CTA warps round-robin, rather than pinning
            // every one-warp CTA to SP0 through a fixed global CTA stride.
            const auto desired = static_cast<std::uint64_t>(sm_next_subpartition[c.sm] % 4);
            next_warp_id += (desired + 4 - next_warp_id % 4) % 4;
            if (next_warp_id + r.warps > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
                throw std::overflow_error("warp numbering exceeds int32");
            warp_bases[c.id] = static_cast<int>(next_warp_id);
            next_warp_id += r.warps;
            sm_next_subpartition[c.sm] = (sm_next_subpartition[c.sm] + r.warps) % 4;
        }
    }
    std::unordered_map<std::string, std::string> pipelines{
        {"ld.dram2reg", "LD"}, {"st.reg2dram", "ST"},
        {"cp.dram2sram_tma", "LD"}, {"cp.sram2dram_tma", "ST"},
        {"st.reg2sram", "ST"}, {"wgmma.m64n64k16", "Tensor"}};
    if (ir.version == 2) {
        pipelines.erase("wgmma.m64n64k16");
        pipelines.erase("cp.dram2sram_tma");
        pipelines.erase("cp.sram2dram_tma");
        pipelines.insert({{"cp.dram2sram", "LD"}, {"ld.sram2reg", "LD"},
            {"mma.sync.m16n8k16", "Tensor"}, {"barrier.registers", "SIMD"},
            {"simd.add", "SIMD"}, {"simd.sub", "SIMD"}, {"simd.mul", "SIMD"},
            {"simd.fma", "SIMD"}, {"simd.max", "SIMD"}, {"simd.min", "SIMD"},
            {"simd.cast", "SIMD"}, {"simd.fill", "SIMD"}, {"simd.identity", "SIMD"},
            {"sfu.exp", "SFU"}, {"sfu.log", "SFU"}, {"sfu.sqrt", "SFU"},
            {"sfu.rsqrt", "SFU"}, {"sfu.divide", "SFU"}, {"shfl", "SHFL"}});
    }
    names.clear();
    std::unordered_set<int> used_ctas;
    std::map<std::pair<int, int>, std::array<std::vector<int>, 4>> collectives;
    for (const auto& op : ir.operations) {
        if (op.id != static_cast<int>(dag->nodes.size()) || !names.insert(op.name).second || op.name.empty())
            throw std::invalid_argument("operation IDs must be dense, ordered, and uniquely named");
        if (!pipelines.contains(op.op)) throw std::invalid_argument("unsupported operation '" + op.op + "'");
        if (!ctas.contains(op.cta) || op.warp < 0 || op.warp >= ctas.at(op.cta).resources.warps)
            throw std::invalid_argument("invalid CTA/warp reference");
        used_ctas.insert(op.cta);
        const auto& tensor = dag->tensor_registry.at(op.tensor);
        const auto& tile = op.region;
        const bool global = op.op == "ld.dram2reg" || op.op == "st.reg2dram" ||
            op.op == "cp.dram2sram" || op.op == "cp.dram2sram_tma" || op.op == "cp.sram2dram_tma";
        if (tile.ndims != 2 || tile.dims[0] != tile.r || tile.dims[1] != tile.c ||
            tile.offs[0] != tile.r_off || tile.offs[1] != tile.c_off ||
            tile.r_off < 0 || tile.c_off < 0 || tile.r <= 0 || tile.c <= 0 ||
            tile.r > std::numeric_limits<int>::max() / tile.c ||
            ((ir.version == 1 || global) &&
             (static_cast<std::uint64_t>(tile.r_off) + tile.r > tensor.shape[0] ||
              static_cast<std::uint64_t>(tile.c_off) + tile.c > tensor.shape[1])))
            throw std::invalid_argument("out-of-bounds access region at %" + std::to_string(op.id));
        bool wgmma = op.op == "wgmma.m64n64k16";
        if ((wgmma && (op.dtype != DataType::FP32 || tensor.dtype != DataType::FP16 ||
                       tile.r != 16 || tile.c != 64 || op.warp >= 8)) ||
            (!wgmma && (ir.version == 1 || global) && op.dtype != tensor.dtype))
            throw std::invalid_argument("operation dtype or WGMMA region is unsupported");
        if (ir.version == 2) {
            if (op.granularity_bytes != 4 && op.granularity_bytes != 8 && op.granularity_bytes != 16)
                throw std::invalid_argument("unsupported memory instruction vector width");
            if (op.op == "mma.sync.m16n8k16" &&
                (op.dtype != DataType::FP32 || tile.r != 16 || tile.c != 8))
                throw std::invalid_argument("invalid Ampere warp MMA fragment");
            if (!global && (op.op == "ld.sram2reg" || op.op == "st.reg2sram") &&
                tile.element_count() * bytes_per_element(op.dtype) > ctas.at(op.cta).resources.shared_bytes)
                throw std::invalid_argument("shared instruction region exceeds CTA allocation");
        }
        if (wgmma) {
            collectives[{op.cta, op.warp / 4}][op.warp % 4].push_back(op.id);
        }
        auto node = std::make_unique<DAGNode>(op.id, op.name, pipelines.at(op.op), op.op,
            warp_bases.at(op.cta) + op.warp, op.after_complete, 0, tile, op.dtype);
        node->sm_id = ctas.at(op.cta).sm;
        node->thread_block_id = op.cta;
        node->matrix_id = ir.version == 2 && !global ? -1 : op.tensor;
        node->memory_access_granularity_bytes = op.granularity_bytes;
        node->layout = op.layout;
        // The reference uses a leading dimension only for global accesses.
        if (ir.version == 1 ? op.op != "st.reg2sram" && !wgmma : global)
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
