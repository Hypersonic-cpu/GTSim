#ifndef DAG_NODE_H
#define DAG_NODE_H

#include <array>
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

namespace GTSim {

// Operation type enum for fast dispatch (avoids string operations in hot path)
enum class OpType {
    LD_SRAM2REG,    // Load from SRAM to register
    LD_SRAM2REG_DSM, // Load from remote SRAM (DSM) to register
    LD_DRAM2REG,    // Load from DRAM to register
    ST_REG2SRAM,    // Store from register to SRAM
    ST_REG2SRAM_DSM, // Store from register to remote SRAM (DSM)
    ST_REG2DRAM,    // Store from register to DRAM
    CP_DRAM2SRAM,   // Copy from DRAM to SRAM
    CP_SRAM2DRAM,   // Copy from SRAM to DRAM
    CP_DRAM2SRAM_TMA, // TMA copy from DRAM to SRAM
    CP_SRAM2DRAM_TMA, // TMA copy from SRAM to DRAM
    CP_SRAM2TMEM,     // Copy from SRAM to Tensor Memory
    MMA,            // Matrix multiply-accumulate
    WGMMA,          // Warp-group MMA (H100)
    TCGEN05_MMA,    // UMMA tcgen05 MMA (Blackwell)
    TCGEN05_ALLOC,  // Tensor Memory allocation
    TCGEN05_COMMIT, // Tensor Memory commit
    TCGEN05_WAIT,   // Tensor Memory wait
    TCGEN05_LD,     // Tensor Memory load
    TCGEN05_ST,     // Tensor Memory store
    BARRIER,        // Synchronization barrier
    COMPUTE,        // General compute operation
    OTHER           // Other operations
};

enum class IssueGroupKind {
    NONE,       // Single-warp issue
    WARPGROUP4, // 4 consecutive warps issue together
    CTA_PAIR    // 2-CTA coupled issue (for UMMA expansion)
};

// Tile information for a DAG node
struct Tile {
    static constexpr int kMaxDims = 4;

    int r_off;  // Row offset
    int c_off;  // Column offset
    int r;      // Row size
    int c;      // Column size

    int ndims;  // Number of dimensions (1-4); 2 keeps legacy behavior
    int dims[kMaxDims];
    int offs[kMaxDims];

    Tile(int r_off = 0, int c_off = 0, int r = 0, int c = 0)
        : r_off(r_off), c_off(c_off), r(r), c(c), ndims(2) {
        dims[0] = r;
        dims[1] = c;
        dims[2] = 1;
        dims[3] = 1;
        offs[0] = r_off;
        offs[1] = c_off;
        offs[2] = 0;
        offs[3] = 0;
    }

    void set_dims(std::initializer_list<int> dims_list,
                  std::initializer_list<int> offs_list = {}) {
        int d = 0;
        for (int v : dims_list) {
            if (d >= kMaxDims) break;
            dims[d++] = v;
        }
        if (d == 0) {
            ndims = 2;
            dims[0] = r;
            dims[1] = c;
            dims[2] = 1;
            dims[3] = 1;
        } else {
            ndims = d;
            for (int i = d; i < kMaxDims; ++i) {
                dims[i] = 1;
            }
        }

        int o = 0;
        for (int v : offs_list) {
            if (o >= kMaxDims) break;
            offs[o++] = v;
        }
        if (o == 0) {
            offs[0] = r_off;
            offs[1] = c_off;
            offs[2] = 0;
            offs[3] = 0;
        } else {
            for (int i = o; i < kMaxDims; ++i) {
                offs[i] = 0;
            }
        }

        r = dims[0];
        c = (ndims > 1) ? dims[1] : 1;
        r_off = offs[0];
        c_off = (ndims > 1) ? offs[1] : 0;
    }

    int dim(int idx) const {
        return dims[idx];
    }

    int off(int idx) const {
        return offs[idx];
    }

    int element_count() const {
        if (ndims <= 2) {
            return r * c;
        }
        long long total = 1;
        for (int i = 0; i < ndims; ++i) {
            total *= static_cast<long long>(dims[i]);
        }
        return static_cast<int>(total);
    }

    bool empty() const {
        if (ndims <= 2) {
            return r <= 0 || c <= 0;
        }
        for (int i = 0; i < ndims; ++i) {
            if (dims[i] <= 0) return true;
        }
        return false;
    }
};

// Memory layout for tile data
enum class Layout {
    RowMajor,
    ColMajor,
    Swizzled
};

// Data type for tile elements
enum class DataType {
    FP16,
    BF16,
    FP8,
    FP32,
    INT8
};

inline int bytes_per_element(DataType type) {
    switch (type) {
        case DataType::FP16:
        case DataType::BF16:
            return 2;
        case DataType::FP8:
            return 1;
        case DataType::FP32:
            return 4;
        case DataType::INT8:
            return 1;
        default:
            return 2;
    }
}

enum class MemoryTier : std::uint8_t {
    HBM,
    HBF
};

struct TensorMetadata {
    int id{};
    std::string name;
    std::uint64_t global_base_address{};
    std::vector<std::uint64_t> shape;
    std::vector<std::uint64_t> strides;
    DataType dtype{DataType::FP16};
    MemoryTier initial_memory_tier{MemoryTier::HBM};

    std::uint64_t byte_size() const {
        if (shape.size() != strides.size()) {
            throw std::invalid_argument("tensor shape and strides must have the same rank");
        }
        if (shape.empty()) {
            return 0;
        }

        std::uint64_t last_element = 0;
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] == 0) {
                return 0;
            }
            const std::uint64_t extent = shape[i] - 1;
            if (strides[i] > 0 &&
                extent > (std::numeric_limits<std::uint64_t>::max() - last_element) /
                             strides[i]) {
                throw std::overflow_error("tensor stride span overflows uint64_t");
            }
            last_element += extent * strides[i];
        }
        if (last_element == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("tensor element count overflows uint64_t");
        }
        const std::uint64_t element_count = last_element + 1;
        const std::uint64_t elem_bytes =
            static_cast<std::uint64_t>(bytes_per_element(dtype));
        if (element_count > std::numeric_limits<std::uint64_t>::max() / elem_bytes) {
            throw std::overflow_error("tensor byte size overflows uint64_t");
        }
        return element_count * elem_bytes;
    }

    bool contains(std::uint64_t address) const {
        const std::uint64_t size = byte_size();
        return size > 0 && address >= global_base_address &&
               address - global_base_address < size;
    }
};

// Workload tensors occupy non-overlapping global address ranges. 4 KiB bases
// keep cache-line alignment stable for the simulator's current power-of-two
// cache-line sizes while leaving address assignment out of Tile/DAGNode.
class TensorRegistry {
public:
    static constexpr std::uint64_t kAddressAlignment = 4096;
    static constexpr std::uint64_t kFirstGlobalAddress = 1ULL << 32;

    TensorRegistry() : next_global_address_(kFirstGlobalAddress) {}

    const TensorMetadata& allocate_tensor(int id, std::string name,
                                          std::vector<std::uint64_t> shape,
                                          std::vector<std::uint64_t> strides,
                                          DataType dtype,
                                          MemoryTier tier = MemoryTier::HBM) {
        const std::uint64_t base = align_up(next_global_address_, kAddressAlignment);
        TensorMetadata tensor{id, std::move(name), base, std::move(shape),
                              std::move(strides), dtype, tier};
        return register_tensor(std::move(tensor));
    }

    const TensorMetadata& register_tensor(TensorMetadata tensor) {
        const auto existing = tensors_.find(tensor.id);
        if (existing != tensors_.end()) {
            throw std::invalid_argument("duplicate tensor id " + std::to_string(tensor.id));
        }
        const std::uint64_t size = tensor.byte_size();
        const std::uint64_t reserved_size = std::max<std::uint64_t>(size, 1);
        if (tensor.global_base_address >
            std::numeric_limits<std::uint64_t>::max() - reserved_size) {
            throw std::overflow_error("global tensor address range overflows uint64_t");
        }
        for (const auto& [unused_id, existing_tensor] : tensors_) {
            (void)unused_id;
            const std::uint64_t existing_size = existing_tensor.byte_size();
            if (size > 0 && existing_size > 0 &&
                tensor.global_base_address < existing_tensor.global_base_address + existing_size &&
                existing_tensor.global_base_address < tensor.global_base_address + size) {
                throw std::invalid_argument("overlapping global tensor address ranges");
            }
        }
        next_global_address_ = std::max(
            next_global_address_,
            align_up(tensor.global_base_address + reserved_size, kAddressAlignment));
        return tensors_.emplace(tensor.id, std::move(tensor)).first->second;
    }

    const TensorMetadata& at(int id) const {
        const auto it = tensors_.find(id);
        if (it == tensors_.end()) {
            throw std::out_of_range("tensor id " + std::to_string(id) + " is not registered");
        }
        return it->second;
    }

    MemoryTier resolve_tier(std::uint64_t address) const {
        for (const auto& [unused_id, tensor] : tensors_) {
            (void)unused_id;
            if (tensor.contains(address)) {
                return tensor.initial_memory_tier;
            }
        }
        throw std::out_of_range("global address is outside registered tensor ranges");
    }

    std::size_t size() const { return tensors_.size(); }

private:
    std::unordered_map<int, TensorMetadata> tensors_;
    std::uint64_t next_global_address_;

    static std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
        const std::uint64_t remainder = value % alignment;
        if (remainder == 0) {
            return value;
        }
        const std::uint64_t increment = alignment - remainder;
        if (value > std::numeric_limits<std::uint64_t>::max() - increment) {
            throw std::overflow_error("global tensor address alignment overflows uint64_t");
        }
        return value + increment;
    }
};

// DAG node representing a single operation
class DAGNode {
public:
    int id;                          // Unified identifier
    std::string name;                // Human-readable name
    std::string pipeline_type;       // "Tensor", "SIMD", "SFU", "LS"
    std::string op;                  // Operation (e.g., "ld.sram2reg", "mma")
    OpType op_type;                  // Parsed operation type for fast dispatch
    int warp_id;                     // Which warp (0-15) executes this
    Tile tile;                       // Tile size information
    Layout layout;                   // Tile memory layout
    DataType data_type;              // Tile element type
    int matrix_id;                   // Matrix identifier for cache indexing
    int matrix_leading_dim;          // Leading dimension (0 means derive from tile)
    int matrix_strides[Tile::kMaxDims]; // Per-dim strides (0 means infer)
    std::vector<int> depends_on;     // Dependencies (node IDs)
    std::vector<DAGNode*> children;  // Downstream dependent nodes
    std::vector<int> issue_depends_on;     // Issue-order dependencies (node IDs)
    std::vector<DAGNode*> issue_children;  // Downstream nodes released on issue completion
    int register_change;             // Register allocation change in bytes

    // Execution tracking
    int start;                       // Start cycle (-1 if not started)
    int end;                         // End cycle (-1 if not finished)
    bool finished;                   // Completion flag
    int pending_transactions;        // Remaining sub-ops before completion
    int total_transactions;          // Total sub-ops for this node
    int next_transaction_index;      // Next transaction index to enqueue
    int remaining_deps;              // Remaining unresolved dependencies
    int ready_cycle;                 // Earliest cycle this node can issue
    int issue_cycle;                 // Cycle when this node is issued
    bool issue_done;                 // All sub-ops issued
    bool issue_deps_resolved;         // Issue dependencies already applied
    int tma_issue_complete_cycle;    // Cycle when all TMA transactions issued

    // Bank conflicts
    int shared_bank_conflict_factor; // Additional cycles (0-31) for bank conflicts
    int setup_latency;              // Extra setup latency between sub-ops (override)
    int memory_coalesce_bytes;      // Global memory coalesce size (bytes)
    int memory_access_granularity_bytes; // Memory access granularity per thread (bytes)
    int tma_setup_latency;          // Override TMA setup latency (cycles), -1 for default
    int tma_issue_interval;         // Override TMA issue interval (cycles), -1 for default
    int tma_issue_rate_bytes_per_cycle; // Override TMA issue rate, -1 for default

    // Group-issue metadata (WGMMA/UMMA extensions)
    IssueGroupKind issue_group_kind; // Scheduling coupling kind
    int issue_group_size;            // Number of warps that co-issue (0 = infer by op type)
    int issue_group_leader_warp;     // Explicit leader warp id (-1 = infer)
    int issue_group_member_index;    // Optional member index in issue group

    // CTA/cluster coupling metadata for future 2-CTA UMMA modeling
    int cta_group_id;                // CTA group identifier (-1 = not grouped)
    int cta_group_size;              // Number of CTAs in group (1/2/...)
    int cta_group_rank;              // Rank of this CTA inside cta_group
    int peer_cta_id;                 // Paired CTA id for 2-CTA modes (-1 = none)
    int cluster_id;                  // Cluster identifier (-1 = unassigned)

    // Tensor Memory metadata (Blackwell UMMA path)
    int tmem_cols;                   // Number of TMEM columns reserved/used
    int tmem_addr;                   // Abstract TMEM address/handle

    // Thread block mapping
    int sm_id;                       // SM this node is assigned to
    int thread_block_id;             // Thread block ID
    int target_sm_id;                // Target SM for remote SRAM access (-1 = local)

    // Parse operation string to enum (called once during construction)
    static OpType parse_op_type(const std::string& op) {
        // Check load operations
        if (op.find("ld.") == 0) {
            if (op.find("sram2reg_dsm") != std::string::npos) return OpType::LD_SRAM2REG_DSM;
            if (op.find("sram2reg") != std::string::npos) return OpType::LD_SRAM2REG;
            if (op.find("dram2reg") != std::string::npos) return OpType::LD_DRAM2REG;
        }
        // Check store operations
        else if (op.find("st.") == 0) {
            if (op.find("reg2sram_dsm") != std::string::npos) return OpType::ST_REG2SRAM_DSM;
            if (op.find("reg2sram") != std::string::npos) return OpType::ST_REG2SRAM;
            if (op.find("reg2dram") != std::string::npos) return OpType::ST_REG2DRAM;
        }
        // Check copy operations
        else if (op.find("cp.") == 0) {
            if (op.find("dram2sram_tma") != std::string::npos) return OpType::CP_DRAM2SRAM_TMA;
            if (op.find("sram2dram_tma") != std::string::npos) return OpType::CP_SRAM2DRAM_TMA;
            if (op.find("sram2tmem") != std::string::npos) return OpType::CP_SRAM2TMEM;
            if (op.find("dram2sram") != std::string::npos) return OpType::CP_DRAM2SRAM;
            if (op.find("sram2dram") != std::string::npos) return OpType::CP_SRAM2DRAM;
        }
        // Check WGMMA
        else if (op == "wgmma" || op.find("wgmma.") == 0) {
            return OpType::WGMMA;
        }
        // Check tcgen05 operations
        else if (op.find("tcgen05.") == 0) {
            if (op.find(".mma") != std::string::npos) return OpType::TCGEN05_MMA;
            if (op.find(".alloc") != std::string::npos) return OpType::TCGEN05_ALLOC;
            if (op.find(".commit") != std::string::npos) return OpType::TCGEN05_COMMIT;
            if (op.find(".wait") != std::string::npos) return OpType::TCGEN05_WAIT;
            if (op.find(".ld") != std::string::npos) return OpType::TCGEN05_LD;
            if (op.find(".st") != std::string::npos) return OpType::TCGEN05_ST;
        }
        // Check barrier
        else if (op == "barrier" || op.find("barrier.") == 0) {
            return OpType::BARRIER;
        }
        // Check MMA
        else if (op == "mma" || op.find("mma.") == 0) {
            return OpType::MMA;
        }

        // Default: generic compute or other
        return OpType::COMPUTE;
    }

    DAGNode(int id, const std::string& name, const std::string& pipeline_type,
            const std::string& op, int warp_id, const std::vector<int>& depends_on,
            int register_change, const Tile& tile, DataType dtype = DataType::FP16)
        : id(id), name(name), pipeline_type(pipeline_type), op(op),
          op_type(parse_op_type(op)), warp_id(warp_id), tile(tile),
          layout(Layout::RowMajor), data_type(dtype),
          matrix_id(0), matrix_leading_dim(0),
          depends_on(depends_on),
          register_change(register_change), start(-1), end(-1), finished(false),
          pending_transactions(0), total_transactions(0), next_transaction_index(0),
          remaining_deps(static_cast<int>(depends_on.size())), ready_cycle(0),
          issue_cycle(-1), issue_done(false), issue_deps_resolved(false),
          tma_issue_complete_cycle(-1),
          shared_bank_conflict_factor(0),
          setup_latency(0),
          memory_coalesce_bytes(128),
          memory_access_granularity_bytes(4),
          tma_setup_latency(-1),
          tma_issue_interval(-1),
          tma_issue_rate_bytes_per_cycle(-1),
          issue_group_kind(IssueGroupKind::NONE),
          issue_group_size(0),
          issue_group_leader_warp(-1),
          issue_group_member_index(-1),
          cta_group_id(-1),
          cta_group_size(1),
          cta_group_rank(0),
          peer_cta_id(-1),
          cluster_id(-1),
          tmem_cols(0),
          tmem_addr(-1),
          sm_id(-1), thread_block_id(-1), target_sm_id(-1) {
        for (int i = 0; i < Tile::kMaxDims; ++i) {
            matrix_strides[i] = 0;
        }
    }

    void set_matrix_strides(std::initializer_list<int> strides) {
        int i = 0;
        for (int v : strides) {
            if (i >= Tile::kMaxDims) break;
            matrix_strides[i++] = v;
        }
        for (; i < Tile::kMaxDims; ++i) {
            matrix_strides[i] = 0;
        }
    }

    int element_size_bytes() const {
        return bytes_per_element(data_type);
    }
};

// DAG containing all nodes for a workload
class DAG {
public:
    TensorRegistry tensor_registry;
    std::vector<DAGNode*> nodes;
    std::unordered_map<int, DAGNode*> nodes_dict;
    std::unordered_map<std::string, DAGNode*> nodes_by_name;

    DAG() = default;

    void add_node(DAGNode* node) {
        nodes.push_back(node);
        nodes_dict[node->id] = node;
        nodes_by_name[node->name] = node;
    }

    DAGNode* add_node_nd(int id, const std::string& name, const std::string& pipeline_type,
                         const std::string& op, int warp_id,
                         const std::vector<int>& depends_on, int register_change,
                         std::initializer_list<int> dims,
                         std::initializer_list<int> offs = {},
                         DataType dtype = DataType::FP16) {
        Tile tile;
        tile.set_dims(dims, offs);
        DAGNode* node = new DAGNode(id, name, pipeline_type, op, warp_id,
                                    depends_on, register_change, tile, dtype);
        add_node(node);
        return node;
    }

    void add_issue_dependency(int from_id, int to_id) {
        auto it = nodes_dict.find(to_id);
        if (it != nodes_dict.end()) {
            it->second->issue_depends_on.push_back(from_id);
        }
    }

    void build_dependency_graph() {
        for (auto* node : nodes) {
            node->children.clear();
            node->issue_children.clear();
            node->remaining_deps = static_cast<int>(node->depends_on.size() +
                                                   node->issue_depends_on.size());
            node->issue_done = false;
            node->issue_deps_resolved = false;
        }

        for (auto* node : nodes) {
            for (int dep_id : node->depends_on) {
                auto it = nodes_dict.find(dep_id);
                if (it != nodes_dict.end()) {
                    it->second->children.push_back(node);
                }
            }
            for (int dep_id : node->issue_depends_on) {
                auto it = nodes_dict.find(dep_id);
                if (it != nodes_dict.end()) {
                    it->second->issue_children.push_back(node);
                }
            }
        }
    }

    DAGNode* get_node_by_id(int id) {
        auto it = nodes_dict.find(id);
        return (it != nodes_dict.end()) ? it->second : nullptr;
    }

    DAGNode* get_node_by_name(const std::string& name) {
        auto it = nodes_by_name.find(name);
        return (it != nodes_by_name.end()) ? it->second : nullptr;
    }

    ~DAG() {
        for (auto* node : nodes) {
            delete node;
        }
    }
};

} // namespace GTSim

#endif // DAG_NODE_H
