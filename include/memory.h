#ifndef MEMORY_H
#define MEMORY_H

#include "dag_node.h"
#include "mem_conf.h"
#include "memory_backend.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iostream>
#include <array>
#include <list>
#include <limits>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace GTSim {

// Memory subsystem (SRAM) with queues and bandwidth modeling
class Memory {
public:
    int memory_latency;
    int memory_bandwidth;
    int memory_queue_depth;

    // Read queue (for ld.sram2reg and cp.dram2sram)
    std::vector<std::tuple<std::vector<int>, int, int>> memory_queue;  // (node_ids, bytes, complete_cycle)
    int memory_occupied_until;

    // Write queue (for st.reg2sram - separate port)
    std::vector<std::tuple<std::vector<int>, int, int>> memory_write_queue;
    int memory_write_queue_depth;
    int memory_write_bandwidth;
    int memory_write_occupied_until;

    // Bank conflict statistics
    int total_bank_conflict_cycles;
    int num_loads_with_conflicts;

    Memory(int latency, int bandwidth, int depth)
        : memory_latency(latency), memory_bandwidth(bandwidth),
          memory_queue_depth(depth), memory_occupied_until(0),
          memory_write_queue_depth(depth), memory_write_bandwidth(bandwidth),
          memory_write_occupied_until(0), total_bank_conflict_cycles(0),
          num_loads_with_conflicts(0) {}

    // Check if queue is full
    bool is_full(bool is_write = false) const {
        if (is_write) {
            return memory_write_queue.size() >= static_cast<size_t>(memory_write_queue_depth);
        } else {
            return memory_queue.size() >= static_cast<size_t>(memory_queue_depth);
        }
    }

    // Enqueue a memory request
    bool enqueue(int node_id, int bytes_val, int current_cycle,
                 int bank_conflict_factor = 0, bool is_write = false) {
        if (is_write) {
            return enqueue_write({node_id}, bytes_val, current_cycle, bank_conflict_factor);
        }
        // Track bank conflict statistics (only for reads)
        if (bank_conflict_factor > 0) {
            total_bank_conflict_cycles += bank_conflict_factor;
            num_loads_with_conflicts++;
        }
        return enqueue_read({node_id}, bytes_val, current_cycle, bank_conflict_factor);
    }

    // Process queues and return completed node IDs
    std::vector<int> step(int current_cycle) {
        std::vector<int> completed;

        // Compact in place: retain request order without allocating and copying
        // every pending request's node-ID vector on every cycle.
        const auto drain = [&](auto& queue) {
            queue.erase(std::remove_if(queue.begin(), queue.end(), [&](const auto& entry) {
                const auto& [node_ids, bytes_val, complete_cycle] = entry;
                (void)bytes_val;
                if (current_cycle < complete_cycle) return false;
                completed.insert(completed.end(), node_ids.begin(), node_ids.end());
                return true;
            }), queue.end());
        };
        drain(memory_queue);
        drain(memory_write_queue);

        return completed;
    }

private:
    // Enqueue read request
    bool enqueue_read(const std::vector<int>& node_ids, int bytes_val,
                      int current_cycle, int bank_conflict_factor = 0) {
        if (is_full(false)) return false;

        // Calculate timing: T_sum = T_frontend + T_queuing + T_data + T_bank_conflict
        int T_frontend = memory_latency;
        int T_queuing = 0;

        if (current_cycle + T_frontend < memory_occupied_until) {
            T_queuing = memory_occupied_until - (current_cycle + T_frontend);
        }

        int T_data = static_cast<int>(std::ceil(static_cast<double>(bytes_val) / memory_bandwidth));
        int T_bank_conflict = bank_conflict_factor;

        int T_sum = T_frontend + T_queuing + T_data + T_bank_conflict;
        int complete_cycle = current_cycle + T_sum;

        memory_occupied_until = current_cycle + T_frontend + T_queuing + T_data + T_bank_conflict;
        memory_queue.push_back({node_ids, bytes_val, complete_cycle});

        return true;
    }

    // Enqueue write request (no bank conflicts)
    bool enqueue_write(const std::vector<int>& node_ids, int bytes_val,
                       int current_cycle, int bank_conflict_factor = 0) {
        if (is_full(true)) return false;

        int T_frontend = memory_latency;
        int T_queuing = 0;

        if (current_cycle + T_frontend < memory_write_occupied_until) {
            T_queuing = memory_write_occupied_until - (current_cycle + T_frontend);
        }

        int T_data = static_cast<int>(std::ceil(static_cast<double>(bytes_val) / memory_write_bandwidth));
        int T_bank_conflict = bank_conflict_factor;
        int T_sum = T_frontend + T_queuing + T_data + T_bank_conflict;
        int complete_cycle = current_cycle + T_sum;

        memory_write_occupied_until = current_cycle + T_frontend + T_queuing + T_data + T_bank_conflict;
        memory_write_queue.push_back({node_ids, bytes_val, complete_cycle});

        return true;
    }
};

inline int tile_ndims(const Tile& tile) {
    return tile.ndims > 0 ? tile.ndims : 2;
}

inline bool layout_is_row_major(const DAGNode& node) {
    return node.layout == Layout::RowMajor;
}

inline int tile_total_elements(const Tile& tile) {
    if (tile.empty()) {
        return 0;
    }
    return tile.element_count();
}

inline void compute_matrix_strides(const DAGNode& node, int ndims, int* strides) {
    for (int i = 0; i < Tile::kMaxDims; ++i) {
        strides[i] = 0;
    }

    bool have_all = true;
    for (int i = 0; i < ndims; ++i) {
        if (node.matrix_strides[i] <= 0) {
            have_all = false;
            break;
        }
    }

    if (have_all) {
        for (int i = 0; i < ndims; ++i) {
            strides[i] = node.matrix_strides[i];
        }
        return;
    }

    if (ndims == 2 && node.matrix_leading_dim > 0) {
        if (layout_is_row_major(node)) {
            strides[0] = node.matrix_leading_dim;
            strides[1] = 1;
        } else {
            strides[0] = 1;
            strides[1] = node.matrix_leading_dim;
        }
        return;
    }

    if (layout_is_row_major(node)) {
        strides[ndims - 1] = 1;
        for (int i = ndims - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * node.tile.dim(i + 1);
        }
    } else {
        strides[0] = 1;
        for (int i = 1; i < ndims; ++i) {
            strides[i] = strides[i - 1] * node.tile.dim(i - 1);
        }
    }
}

inline bool has_explicit_strides(const DAGNode& node, int ndims) {
    for (int i = 0; i < ndims; ++i) {
        if (node.matrix_strides[i] <= 0) return false;
    }
    return ndims > 0;
}

inline void compute_dim_order(const DAGNode& node, const int* strides, int ndims, int* order) {
    for (int i = 0; i < ndims; ++i) {
        order[i] = i;
    }
    if (has_explicit_strides(node, ndims)) {
        // Smallest stride is the fastest-changing dimension.
        std::stable_sort(order, order + ndims, [&](int a, int b) {
            int sa = strides[a];
            int sb = strides[b];
            if (sa <= 0 && sb <= 0) return a < b;
            if (sa <= 0) return false;
            if (sb <= 0) return true;
            if (sa != sb) return sa < sb;
            return a < b;
        });
    } else {
        // Fall back to layout-based order.
        if (layout_is_row_major(node)) {
            std::reverse(order, order + ndims);
        }
    }
}

inline void compute_sram_address_params(const DAGNode& node, int& elem_bytes,
                                        int* strides, int& ndims) {
    elem_bytes = node.element_size_bytes();
    ndims = tile_ndims(node.tile);
    compute_matrix_strides(node, ndims, strides);
}

inline std::uint64_t compute_sram_element_addr(const DAGNode& node, int linear_index,
                                               int elem_bytes, const int* strides,
                                               int ndims) {
    std::uint64_t elem_index = 0;
    int remaining = linear_index;
    int order[Tile::kMaxDims];
    compute_dim_order(node, strides, ndims, order);
    for (int oi = 0; oi < ndims; ++oi) {
        int i = order[oi];
        int dim = node.tile.dim(i);
        int idx = (dim > 0) ? (remaining % dim) : 0;
        remaining = (dim > 0) ? (remaining / dim) : 0;
        elem_index += static_cast<std::uint64_t>(node.tile.off(i) + idx) *
                      static_cast<std::uint64_t>(strides[i]);
    }
    return elem_index * static_cast<std::uint64_t>(elem_bytes);
}

inline int get_access_granularity_bytes(const DAGNode& node) {
    if (node.memory_access_granularity_bytes > 0) {
        return node.memory_access_granularity_bytes;
    }
    return 4;
}

inline int compute_sram_subops(const DAGNode& node, int access_granularity_bytes,
                               std::vector<int>* subop_bytes = nullptr,
                               std::vector<int>* subop_conflicts = nullptr) {
    constexpr int kBankWidthBytes = 4;
    constexpr int kBankCount = 32;

    if (subop_bytes) {
        subop_bytes->clear();
    }
    if (subop_conflicts) {
        subop_conflicts->clear();
    }

    if (node.tile.empty()) {
        if (subop_bytes) {
            subop_bytes->push_back(0);
        }
        if (subop_conflicts) {
            subop_conflicts->push_back(0);
        }
        return 1;
    }

    int elem_bytes = 0;
    int ndims = 0;
    int strides[Tile::kMaxDims];
    compute_sram_address_params(node, elem_bytes, strides, ndims);
    int total_elements = tile_total_elements(node.tile);

    int access_bytes = access_granularity_bytes > 0 ? access_granularity_bytes : 4;
    int elems_per_thread = std::max(1, (access_bytes + elem_bytes - 1) / elem_bytes);
    int elems_per_subop = 32 * elems_per_thread;
    int subops = (total_elements + elems_per_subop - 1) / elems_per_subop;

    if (!subop_bytes && !subop_conflicts) {
        return std::max(1, subops);
    }

    for (int subop = 0; subop < subops; ++subop) {
        int base_idx = subop * elems_per_subop;
        int remaining = total_elements - base_idx;
        int elems_in_subop = std::min(remaining, elems_per_subop);
        if (subop_bytes) {
            subop_bytes->push_back(elems_in_subop * elem_bytes);
        }
        if (subop_conflicts) {
            int conflicts = 0;
            std::array<bool, kBankCount> bank_used{};
            bank_used.fill(false);
            std::unordered_map<std::uint64_t, int> word_counts;

            if (node.layout == Layout::Swizzled) {
                conflicts = 0;
            } else {
                for (int t = 0; t < 32; ++t) {
                    for (int e = 0; e < elems_per_thread; ++e) {
                        int linear = base_idx + t * elems_per_thread + e;
                        if (linear >= total_elements) {
                            break;
                        }
                        std::uint64_t addr =
                            compute_sram_element_addr(node, linear, elem_bytes, strides, ndims);
                        if (elem_bytes < kBankWidthBytes) {
                            std::uint64_t word_addr = addr / kBankWidthBytes;
                            int capacity = kBankWidthBytes / elem_bytes;
                            if (word_counts[word_addr] >= capacity) {
                                conflicts++;
                            }
                            word_counts[word_addr] += 1;
                        } else {
                            int words = (elem_bytes + kBankWidthBytes - 1) / kBankWidthBytes;
                            for (int w = 0; w < words; ++w) {
                                int bank = static_cast<int>(((addr + static_cast<std::uint64_t>(w) *
                                                             kBankWidthBytes) /
                                                             kBankWidthBytes) % kBankCount);
                                if (bank_used[bank]) {
                                    conflicts++;
                                } else {
                                    bank_used[bank] = true;
                                }
                            }
                        }
                    }
                }
            }
            subop_conflicts->push_back(conflicts);
        }
    }

    return std::max(1, subops);
}

inline int compute_sram_subop_bytes(const DAGNode& node, int access_granularity_bytes,
                                    int subop_index) {
    if (subop_index < 0) {
        return 0;
    }
    std::vector<int> subop_bytes;
    compute_sram_subops(node, access_granularity_bytes, &subop_bytes);
    if (subop_index >= static_cast<int>(subop_bytes.size())) {
        return 0;
    }
    return subop_bytes[static_cast<size_t>(subop_index)];
}

inline int compute_sram_subop_conflict(const DAGNode& node, int access_granularity_bytes,
                                       int subop_index) {
    if (subop_index < 0) {
        return 0;
    }
    if (node.layout == Layout::Swizzled) {
        return 0;
    }
    std::vector<int> subop_conflicts;
    compute_sram_subops(node, access_granularity_bytes, nullptr, &subop_conflicts);
    if (subop_index >= static_cast<int>(subop_conflicts.size())) {
        return 0;
    }
    return subop_conflicts[static_cast<size_t>(subop_index)];
}

struct CacheLineKey {
    int matrix_id;
    std::uint64_t line_addr;  // Tensor-relative line address retained for diagnostics.
    std::uint64_t global_line_addr;

    bool operator==(const CacheLineKey& other) const {
        return global_line_addr == other.global_line_addr;
    }
};

struct CacheLineKeyHash {
    std::size_t operator()(const CacheLineKey& key) const {
        return std::hash<std::uint64_t>{}(key.global_line_addr);
    }
};

struct L2Access {
    int node_id;
    CacheLineKey line;
    int size_bytes;
    MemoryOperation operation;
    MemoryTier tier;
};

inline int compute_dram_subops(const DAGNode& node, int access_granularity_bytes) {
    if (node.tile.empty()) {
        return 1;
    }
    int elem_bytes = node.element_size_bytes();
    int access_bytes = access_granularity_bytes > 0 ? access_granularity_bytes : 4;
    int elems_per_thread = std::max(1, (access_bytes + elem_bytes - 1) / elem_bytes);
    int elems_per_subop = 32 * elems_per_thread;
    int total_elements = tile_total_elements(node.tile);
    int subops = (total_elements + elems_per_subop - 1) / elems_per_subop;
    return std::max(1, subops);
}

inline int compute_dram_subops(const DAGNode& node, int access_granularity_bytes,
                               int elem_bytes) {
    if (node.tile.empty()) {
        return 1;
    }
    int access_bytes = access_granularity_bytes > 0 ? access_granularity_bytes : 4;
    int elems_per_thread = std::max(1, (access_bytes + elem_bytes - 1) / elem_bytes);
    int elems_per_subop = 32 * elems_per_thread;
    int total_elements = tile_total_elements(node.tile);
    int subops = (total_elements + elems_per_subop - 1) / elems_per_subop;
    return std::max(1, subops);
}

inline std::uint64_t compute_global_element_addr(const DAGNode& node,
                                                 const TensorMetadata& tensor,
                                                 int linear_index) {
    const int ndims = tile_ndims(node.tile);
    if (ndims <= 0 || ndims > Tile::kMaxDims ||
        static_cast<std::size_t>(ndims) != tensor.shape.size() ||
        tensor.strides.size() != tensor.shape.size()) {
        throw std::invalid_argument("tensor rank does not match tile rank for node '" +
                                    node.name + "'");
    }
    if (linear_index < 0 || linear_index >= tile_total_elements(node.tile)) {
        throw std::out_of_range("tile element index is out of range for node '" +
                                node.name + "'");
    }

    int order[Tile::kMaxDims];
    for (int i = 0; i < ndims; ++i) {
        order[i] = i;
    }
    std::stable_sort(order, order + ndims, [&](int a, int b) {
        const auto stride_a = tensor.strides[static_cast<std::size_t>(a)];
        const auto stride_b = tensor.strides[static_cast<std::size_t>(b)];
        if (stride_a != stride_b) {
            return stride_a < stride_b;
        }
        return node.layout == Layout::RowMajor ? a > b : a < b;
    });

    int remaining = linear_index;
    std::uint64_t element_offset = 0;
    for (int oi = 0; oi < ndims; ++oi) {
        const int dim_index = order[oi];
        const int tile_dim = node.tile.dim(dim_index);
        const int tile_index = remaining % tile_dim;
        remaining /= tile_dim;
        const int tensor_index = node.tile.off(dim_index) + tile_index;
        if (tensor_index < 0 ||
            static_cast<std::uint64_t>(tensor_index) >=
                tensor.shape[static_cast<std::size_t>(dim_index)]) {
            throw std::out_of_range("tile exceeds tensor bounds for node '" + node.name + "'");
        }
        element_offset += static_cast<std::uint64_t>(tensor_index) *
                          tensor.strides[static_cast<std::size_t>(dim_index)];
    }

    const std::uint64_t elem_bytes =
        static_cast<std::uint64_t>(bytes_per_element(tensor.dtype));
    if (element_offset >
        (std::numeric_limits<std::uint64_t>::max() - tensor.global_base_address) / elem_bytes) {
        throw std::overflow_error("global tensor element address overflows uint64_t");
    }
    return tensor.global_base_address + element_offset * elem_bytes;
}

inline std::vector<CacheLineKey> compute_dram_subop_lines(const DAGNode& node,
                                                          const TensorMetadata& tensor,
                                                          int access_granularity_bytes,
                                                          int subop_index,
                                                          int line_size_bytes) {
    std::vector<CacheLineKey> lines;
    if (node.tile.empty() || subop_index < 0 || line_size_bytes <= 0) {
        return lines;
    }
    const int elem_bytes = bytes_per_element(tensor.dtype);
    int access_bytes = access_granularity_bytes > 0 ? access_granularity_bytes : 4;
    int elems_per_thread = std::max(1, (access_bytes + elem_bytes - 1) / elem_bytes);
    int elems_per_subop = 32 * elems_per_thread;
    int total_elements = tile_total_elements(node.tile);
    int base_idx = subop_index * elems_per_subop;
    if (base_idx >= total_elements) {
        return lines;
    }

    std::unordered_set<std::uint64_t> seen;
    for (int t = 0; t < 32; ++t) {
        for (int e = 0; e < elems_per_thread; ++e) {
            int linear = base_idx + t * elems_per_thread + e;
            if (linear >= total_elements) {
                break;
            }
            const std::uint64_t addr = compute_global_element_addr(node, tensor, linear);
            const std::uint64_t global_line_addr =
                (addr / static_cast<std::uint64_t>(line_size_bytes)) *
                static_cast<std::uint64_t>(line_size_bytes);
            if (seen.insert(global_line_addr).second) {
                const std::uint64_t relative_addr = addr - tensor.global_base_address;
                const std::uint64_t line_addr =
                    (relative_addr / static_cast<std::uint64_t>(line_size_bytes)) *
                    static_cast<std::uint64_t>(line_size_bytes);
                lines.push_back({node.matrix_id, line_addr, global_line_addr});
            }
        }
    }
    return lines;
}

class L2Cache {
public:
    struct CacheLineState {
        bool dirty;
        std::list<CacheLineKey>::iterator lru_it;
    };

    struct MSHREntry {
        std::vector<L2Access> waiting;
        bool any_write;
        bool any_read;
        std::uint64_t backend_request_id;
        MemoryTier tier;
    };

    L2Cache(int cache_size_bytes, int line_size_bytes, int hit_latency_cycles,
            int l2_bandwidth_bytes_per_cycle,
            int l2_write_bandwidth_bytes_per_cycle, int l2_queue_depth,
            bool bypass_cache, const TensorRegistry& tensors)
        : line_size_bytes(line_size_bytes),
          hit_latency_cycles(hit_latency_cycles),
          l2_bandwidth_bytes_per_cycle(l2_bandwidth_bytes_per_cycle),
          l2_write_bandwidth_bytes_per_cycle(l2_write_bandwidth_bytes_per_cycle),
          l2_queue_depth(l2_queue_depth),
          bypass_cache(bypass_cache),
          max_lines(static_cast<size_t>(cache_size_bytes / line_size_bytes)),
          tensors(tensors) {}

    bool can_accept_transaction() const {
        return l2_queue_read.size() + l2_queue_write.size() <
               static_cast<size_t>(l2_queue_depth);
    }

    int get_line_size_bytes() const { return line_size_bytes; }

    int get_transaction_count(const DAGNode& node) const {
        return get_total_transaction_count(node);
    }

    CacheLineKey get_transaction_key(const DAGNode& node, int index) const {
        if (index < 0) {
            return tensor_base_key(node);
        }
        const int subops = get_subop_count(node);
        int offset = index;
        for (int subop = 0; subop < subops; ++subop) {
            auto lines = get_subop_lines(node, subop);
            if (offset < static_cast<int>(lines.size())) {
                return lines[static_cast<std::size_t>(offset)];
            }
            offset -= static_cast<int>(lines.size());
        }
        return tensor_base_key(node);
    }

    bool enqueue_transaction(const DAGNode& node, bool is_write, int index) {
        return enqueue_transaction_key(node.id, get_transaction_key(node, index), is_write);
    }

    bool enqueue_transaction_key(int node_id, const CacheLineKey& key, bool is_write) {
        if (!can_accept_transaction()) {
            return false;
        }
        if (key.global_line_addr % static_cast<std::uint64_t>(line_size_bytes) != 0) {
            throw std::invalid_argument("L2 transaction address is not cache-line aligned");
        }
        const L2Access access{node_id, key, line_size_bytes,
                              is_write ? MemoryOperation::Write : MemoryOperation::Read,
                              tensors.resolve_tier(key.global_line_addr)};
        if (is_write) {
            l2_queue_write.push_back(access);
        } else {
            l2_queue_read.push_back(access);
        }
        return true;
    }

    int get_subop_count(const DAGNode& node) const {
        const TensorMetadata& tensor = tensors.at(node.matrix_id);
        return compute_dram_subops(node, get_access_granularity_bytes(node),
                                   bytes_per_element(tensor.dtype));
    }

    int get_total_transaction_count(const DAGNode& node) const {
        int total = 0;
        for (int subop = 0; subop < get_subop_count(node); ++subop) {
            total += static_cast<int>(get_subop_lines(node, subop).size());
        }
        return std::max(1, total);
    }

    std::vector<CacheLineKey> get_subop_lines(const DAGNode& node, int subop_index) const {
        const TensorMetadata& tensor = tensors.at(node.matrix_id);
        if (node.tile.empty()) {
            return subop_index == 0 ? std::vector<CacheLineKey>{tensor_base_key(node)}
                                    : std::vector<CacheLineKey>{};
        }
        auto lines = compute_dram_subop_lines(node, tensor,
                                              get_access_granularity_bytes(node),
                                              subop_index, line_size_bytes);
        const int factor = get_coalesce_factor(node);
        if (factor <= 1 || lines.empty()) {
            return lines;
        }
        std::vector<CacheLineKey> expanded;
        expanded.reserve(lines.size() * static_cast<std::size_t>(factor));
        for (const auto& line : lines) {
            for (int i = 0; i < factor; ++i) {
                expanded.push_back(line);
            }
        }
        return expanded;
    }

    std::vector<int> step(int current_cycle) {
        std::vector<int> completed;
        // Both hit and response paths append at the current cycle plus the
        // same fixed L2 latency. Completion times are therefore nondecreasing.
        while (!completion_queue.empty() && completion_queue.front().second <= current_cycle) {
            completed.push_back(completion_queue.front().first);
            completion_queue.pop_front();
        }

        const int total_budget_tx = l2_bandwidth_bytes_per_cycle / line_size_bytes;
        int write_budget_tx = l2_write_bandwidth_bytes_per_cycle / line_size_bytes;
        if (write_budget_tx > total_budget_tx) {
            write_budget_tx = total_budget_tx;
        }
        const int read_budget_tx = total_budget_tx - write_budget_tx;
        const bool cache_enabled = !bypass_cache && max_lines > 0;

        int writes_processed = 0;
        while (!l2_queue_write.empty() && writes_processed < write_budget_tx) {
            L2Access access = std::move(l2_queue_write.front());
            l2_queue_write.pop_front();
            ++writes_processed;
            process_access(access, current_cycle, cache_enabled);
        }

        const int remaining_write_tx = write_budget_tx - writes_processed;
        const int read_budget = read_budget_tx + remaining_write_tx;
        int reads_processed = 0;
        while (!l2_queue_read.empty() && reads_processed < read_budget) {
            L2Access access = std::move(l2_queue_read.front());
            l2_queue_read.pop_front();
            ++reads_processed;
            process_access(access, current_cycle, cache_enabled);
        }
        return completed;
    }

    bool idle() const {
        return l2_queue_read.empty() && l2_queue_write.empty() &&
               mshr.empty() && backend_requests.empty() && completion_queue.empty();
    }

    std::vector<MemRequest> take_backend_requests() {
        std::vector<MemRequest> requests;
        requests.swap(backend_requests);
        return requests;
    }

    void handle_response(const MemResponse& response, int current_cycle) {
        const auto request_it = backend_request_to_line.find(response.request_id);
        if (request_it == backend_request_to_line.end()) {
            return;  // A writeback has no L2 waiter.
        }
        if (!response.completed) {
            throw std::runtime_error("memory backend reported a failed request");
        }

        const CacheLineKey key = request_it->second;
        if (response.global_addr != key.global_line_addr) {
            throw std::runtime_error("memory response address does not match its request");
        }
        const auto mshr_it = mshr.find(key);
        if (mshr_it == mshr.end() ||
            mshr_it->second.backend_request_id != response.request_id ||
            mshr_it->second.tier != response.tier) {
            throw std::runtime_error("memory response does not match an active L2 MSHR");
        }

        if (!bypass_cache && max_lines > 0 && mshr_it->second.any_read) {
            insert_line(key, mshr_it->second.any_write);
        }
        for (const auto& access : mshr_it->second.waiting) {
            completion_queue.push_back({access.node_id, current_cycle + hit_latency_cycles});
        }
        backend_request_to_line.erase(request_it);
        mshr.erase(mshr_it);
    }

private:
    int line_size_bytes;
    int hit_latency_cycles;
    int l2_bandwidth_bytes_per_cycle;
    int l2_write_bandwidth_bytes_per_cycle;
    int l2_queue_depth;
    bool bypass_cache;
    size_t max_lines;
    const TensorRegistry& tensors;

    std::list<CacheLineKey> lru_list;
    std::unordered_map<CacheLineKey, CacheLineState, CacheLineKeyHash> cache;
    std::unordered_map<CacheLineKey, MSHREntry, CacheLineKeyHash> mshr;
    std::unordered_map<std::uint64_t, CacheLineKey> backend_request_to_line;
    std::deque<L2Access> l2_queue_read;
    std::deque<L2Access> l2_queue_write;
    std::deque<std::pair<int, int>> completion_queue;
    std::vector<MemRequest> backend_requests;
    std::uint64_t next_request_id{1};

    CacheLineKey tensor_base_key(const DAGNode& node) const {
        const TensorMetadata& tensor = tensors.at(node.matrix_id);
        const std::uint64_t line_size = static_cast<std::uint64_t>(line_size_bytes);
        const std::uint64_t global_line_addr =
            (tensor.global_base_address / line_size) * line_size;
        return {node.matrix_id, 0, global_line_addr};
    }

    void process_access(const L2Access& access, int current_cycle, bool cache_enabled) {
        if (cache_enabled) {
            const auto cache_it = cache.find(access.line);
            if (cache_it != cache.end()) {
                touch_lru(access.line);
                if (access.operation == MemoryOperation::Write) {
                    cache_it->second.dirty = true;
                }
                completion_queue.push_back({access.node_id, current_cycle + hit_latency_cycles});
                return;
            }
        }

        auto mshr_it = mshr.find(access.line);
        if (mshr_it != mshr.end()) {
            mshr_it->second.waiting.push_back(access);
            if (access.operation == MemoryOperation::Write) {
                mshr_it->second.any_write = true;
            } else {
                mshr_it->second.any_read = true;
            }
            return;
        }

        const std::uint64_t request_id = next_request_id++;
        MSHREntry entry{{access},
                        access.operation == MemoryOperation::Write,
                        access.operation == MemoryOperation::Read,
                        request_id,
                        access.tier};
        mshr.emplace(access.line, std::move(entry));
        backend_request_to_line.emplace(request_id, access.line);
        backend_requests.push_back({request_id, access.line.global_line_addr,
                                    access.size_bytes, access.operation, access.tier});
    }

    void touch_lru(const CacheLineKey& key) {
        auto it = cache.find(key);
        if (it == cache.end()) {
            return;
        }
        lru_list.erase(it->second.lru_it);
        lru_list.push_front(key);
        it->second.lru_it = lru_list.begin();
    }

    void insert_line(const CacheLineKey& key, bool dirty) {
        auto it = cache.find(key);
        if (it != cache.end()) {
            it->second.dirty = it->second.dirty || dirty;
            touch_lru(key);
            return;
        }

        if (cache.size() >= max_lines && !lru_list.empty()) {
            const CacheLineKey victim = lru_list.back();
            lru_list.pop_back();
            const auto victim_it = cache.find(victim);
            if (victim_it != cache.end()) {
                if (victim_it->second.dirty) {
                    backend_requests.push_back({next_request_id++, victim.global_line_addr,
                                                line_size_bytes, MemoryOperation::Writeback,
                                                tensors.resolve_tier(victim.global_line_addr)});
                }
                cache.erase(victim_it);
            }
        }

        lru_list.push_front(key);
        cache.emplace(key, CacheLineState{dirty, lru_list.begin()});
    }

    int get_coalesce_factor(const DAGNode& node) const {
        int coalesce = node.memory_coalesce_bytes;
        if (coalesce <= 0) {
            coalesce = 128;
        }
        if (coalesce != 128 && coalesce != 64 && coalesce != 32) {
            std::cerr << "Invalid memory_coalesce_bytes=" << coalesce
                      << " for node " << node.name
                      << "; expected 128/64/32, defaulting to 128.\n";
            assert(false && "memory_coalesce_bytes must be 128/64/32");
            coalesce = 128;
        }
        if (coalesce > 128) {
            coalesce = 128;
        }
        return std::max(1, 128 / coalesce);
    }
};


inline constexpr int kMemoryControlBytes = 16;

struct TierMemoryStatistics {
    std::uint64_t request_count{};
    std::uint64_t read_requests{};
    std::uint64_t write_requests{};
    std::uint64_t read_bytes{};
    std::uint64_t write_bytes{};
    std::uint64_t completed_requests{};
    std::uint64_t total_latency_cycles{};

    double average_latency_cycles() const {
        return completed_requests == 0
                   ? 0.0
                   : static_cast<double>(total_latency_cycles) /
                         static_cast<double>(completed_requests);
    }
};

struct D2DLinkStatistics {
    std::uint64_t forward_bytes{};
    std::uint64_t reverse_bytes{};
    double forward_busy_cycles{};
    double reverse_busy_cycles{};
    double forward_utilization{};
    double reverse_utilization{};
};

struct MemorySystemStatistics {
    TierMemoryStatistics hbm{};
    TierMemoryStatistics hbf{};
    D2DLinkStatistics gpu_to_hbm{};
    D2DLinkStatistics gpu_to_hbf{};
    D2DLinkStatistics hbm_to_hbf{};
    bool has_gpu_to_hbm{};
    bool has_gpu_to_hbf{};
    bool has_hbm_to_hbf{};
};

// HBM controller with a shared read/write data path and an additional write
// throughput ceiling. Bandwidth is expressed in bytes per GPU core cycle.
class HBMModel : public MemoryBackend {
public:
    HBMModel(const HBMConfig& config, double core_frequency_mhz)
        : total_bandwidth_bytes_per_cycle_(bytes_per_cycle_from_GBps(
              config.read_bw_GBps, core_frequency_mhz)),
          write_max_bandwidth_bytes_per_cycle_(bytes_per_cycle_from_GBps(
              config.write_max_bw_GBps, core_frequency_mhz)),
          read_latency_cycles_(config.read_latency_cycles),
          write_latency_cycles_(config.write_latency_cycles),
          queue_capacity_(static_cast<std::size_t>(config.queue_capacity)) {}

    std::size_t pending_requests() const override { return queued_requests(); }

    // Compatibility constructor for code that used the previous DRAMModel API.
    HBMModel(int bandwidth_bytes_per_cycle, int latency_cycles,
             int /*line_size_bytes*/, double core_frequency_mhz,
             double dram_frequency_mhz)
        : total_bandwidth_bytes_per_cycle_(
              static_cast<double>(bandwidth_bytes_per_cycle) *
              dram_frequency_mhz / core_frequency_mhz),
          write_max_bandwidth_bytes_per_cycle_(
              static_cast<double>(bandwidth_bytes_per_cycle) *
              dram_frequency_mhz / core_frequency_mhz),
          read_latency_cycles_(latency_cycles),
          write_latency_cycles_(latency_cycles),
          queue_capacity_(65536) {}

    bool try_submit(const MemRequest& request, int now) override {
        if (queued_requests() >= queue_capacity_) {
            return false;
        }
        Request queued{request,
                       now + (request.operation == MemoryOperation::Read
                                  ? read_latency_cycles_
                                  : write_latency_cycles_),
                       static_cast<double>(std::max(1, request.size_bytes))};
        if (request.operation == MemoryOperation::Read) {
            read_requests_.push_back(queued);
        } else {
            write_requests_.push_back(queued);
        }
        return true;
    }

    void submit(const MemRequest& request, int now) {
        if (!try_submit(request, now)) {
            throw std::runtime_error("HBM request queue is full");
        }
    }

    void enqueue(const MemRequest& request, int now) { submit(request, now); }

    std::vector<MemResponse> step(int now) override {
        std::vector<MemResponse> completed;
        const bool read_ready = !read_requests_.empty() &&
                                read_requests_.front().ready_cycle <= now;
        const bool write_ready = !write_requests_.empty() &&
                                 write_requests_.front().ready_cycle <= now;
        if (!read_ready && !write_ready) {
            total_fractional_credit_ = 0.0;
            write_fractional_credit_ = 0.0;
            return completed;
        }

        double total_budget = total_bandwidth_bytes_per_cycle_ +
                              total_fractional_credit_;
        double write_budget = write_max_bandwidth_bytes_per_cycle_ +
                              write_fractional_credit_;
        while (total_budget > kBandwidthEpsilon) {
            const bool can_read = !read_requests_.empty() &&
                                  read_requests_.front().ready_cycle <= now;
            const bool can_write = !write_requests_.empty() &&
                                   write_requests_.front().ready_cycle <= now &&
                                   write_budget > kBandwidthEpsilon;
            if (!can_read && !can_write) {
                break;
            }

            const bool use_write = can_write && (!can_read || prefer_write_);
            Request& request = use_write ? write_requests_.front()
                                         : read_requests_.front();
            const double available = use_write
                                         ? std::min(total_budget, write_budget)
                                         : total_budget;
            const double transferred = std::min(
                static_cast<double>(request.remaining_bytes), available);
            if (transferred <= kBandwidthEpsilon) {
                break;
            }

            request.remaining_bytes -= transferred;
            total_budget -= transferred;
            if (use_write) {
                write_budget -= transferred;
            }
            prefer_write_ = !use_write;

            if (request.remaining_bytes <= kBandwidthEpsilon) {
                completed.push_back(make_response(request.memory_request));
                if (use_write) {
                    write_requests_.pop_front();
                } else {
                    read_requests_.pop_front();
                }
            }
        }

        total_fractional_credit_ =
            queued_requests() == 0 ? 0.0 : fractional_part(total_budget);
        write_fractional_credit_ =
            write_requests_.empty() ? 0.0 : fractional_part(write_budget);
        return completed;
    }

private:
    struct Request {
        MemRequest memory_request;
        int ready_cycle;
        double remaining_bytes;
    };

    static constexpr double kBandwidthEpsilon = 1.0e-9;

    double total_bandwidth_bytes_per_cycle_;
    double write_max_bandwidth_bytes_per_cycle_;
    int read_latency_cycles_;
    int write_latency_cycles_;
    std::size_t queue_capacity_;
    std::deque<Request> read_requests_;
    std::deque<Request> write_requests_;
    double total_fractional_credit_{};
    double write_fractional_credit_{};
    bool prefer_write_{};

    std::size_t queued_requests() const {
        return read_requests_.size() + write_requests_.size();
    }

    static double fractional_part(double value) {
        return value - std::floor(value);
    }

    static MemResponse make_response(const MemRequest& request) {
        const int response_bytes = request.operation == MemoryOperation::Read
                                       ? std::max(1, request.size_bytes)
                                       : kMemoryControlBytes;
        return {request.request_id, request.global_addr, request.tier, true,
                response_bytes, request.operation};
    }
};

// Keep the old type spelling available for downstream code while all new
// topology construction uses HBMModel.
using DRAMModel = HBMModel;

// HBF controller with independent read and write service queues and bandwidths.
class HBFModel {
public:
    HBFModel(const HBFConfig& config, double core_frequency_mhz)
        : read_bandwidth_bytes_per_cycle_(bytes_per_cycle_from_GBps(
              config.read_bw_GBps, core_frequency_mhz)),
          write_bandwidth_bytes_per_cycle_(bytes_per_cycle_from_GBps(
              config.write_bw_GBps, core_frequency_mhz)),
          read_latency_cycles_(config.read_latency_cycles),
          write_latency_cycles_(config.write_latency_cycles),
          queue_capacity_(static_cast<std::size_t>(config.queue_capacity)) {}

    bool try_submit(const MemRequest& request, int now) {
        if (queued_requests() >= queue_capacity_) {
            return false;
        }
        Request queued{request,
                       now + (request.operation == MemoryOperation::Read
                                  ? read_latency_cycles_
                                  : write_latency_cycles_),
                       static_cast<double>(std::max(1, request.size_bytes))};
        if (request.operation == MemoryOperation::Read) {
            read_requests_.push_back(queued);
        } else {
            write_requests_.push_back(queued);
        }
        return true;
    }

    void submit(const MemRequest& request, int now) {
        if (!try_submit(request, now)) {
            throw std::runtime_error("HBF request queue is full");
        }
    }

    std::vector<MemResponse> step(int now) {
        std::vector<MemResponse> completed;
        service_queue(read_requests_, read_bandwidth_bytes_per_cycle_,
                      read_fractional_credit_, now, completed);
        service_queue(write_requests_, write_bandwidth_bytes_per_cycle_,
                      write_fractional_credit_, now, completed);
        return completed;
    }

private:
    struct Request {
        MemRequest memory_request;
        int ready_cycle;
        double remaining_bytes;
    };

    static constexpr double kBandwidthEpsilon = 1.0e-9;

    double read_bandwidth_bytes_per_cycle_;
    double write_bandwidth_bytes_per_cycle_;
    int read_latency_cycles_;
    int write_latency_cycles_;
    std::size_t queue_capacity_;
    std::deque<Request> read_requests_;
    std::deque<Request> write_requests_;
    double read_fractional_credit_{};
    double write_fractional_credit_{};

    std::size_t queued_requests() const {
        return read_requests_.size() + write_requests_.size();
    }

    static double fractional_part(double value) {
        return value - std::floor(value);
    }

    static MemResponse make_response(const MemRequest& request) {
        const int response_bytes = request.operation == MemoryOperation::Read
                                       ? std::max(1, request.size_bytes)
                                       : kMemoryControlBytes;
        return {request.request_id, request.global_addr, request.tier, true,
                response_bytes, request.operation};
    }

    static void service_queue(std::deque<Request>& requests,
                              double bandwidth_bytes_per_cycle,
                              double& fractional_credit, int now,
                              std::vector<MemResponse>& completed) {
        if (requests.empty() || requests.front().ready_cycle > now) {
            fractional_credit = 0.0;
            return;
        }

        double budget = bandwidth_bytes_per_cycle + fractional_credit;
        while (budget > kBandwidthEpsilon && !requests.empty() &&
               requests.front().ready_cycle <= now) {
            Request& request = requests.front();
            const double transferred = std::min(request.remaining_bytes, budget);
            if (transferred <= kBandwidthEpsilon) {
                break;
            }
            request.remaining_bytes -= transferred;
            budget -= transferred;
            if (request.remaining_bytes <= kBandwidthEpsilon) {
                completed.push_back(make_response(request.memory_request));
                requests.pop_front();
            }
        }
        fractional_credit = requests.empty() ? 0.0 : fractional_part(budget);
    }
};

// Full-duplex point-to-point link. Each direction has its own serializer and
// propagation latency; queueing is FIFO and bounded by queue_capacity.
class D2DLink {
public:
    D2DLink(const D2DLinkConfig& config, double core_frequency_mhz)
        : bandwidth_bytes_per_cycle_(bytes_per_cycle_from_GBps(
              config.bandwidth_GBps, core_frequency_mhz)),
          latency_cycles_(config.latency_cycles),
          queue_capacity_(static_cast<std::size_t>(config.queue_capacity)) {}

    bool enqueue_request(const MemRequest& request, int now) {
        if (request_queue_.size() >= queue_capacity_) {
            return false;
        }
        const int transfer_bytes = request.operation == MemoryOperation::Read
                                       ? kMemoryControlBytes
                                       : std::max(1, request.size_bytes);
        const double serialization = serialization_cycles(transfer_bytes);
        const double start_cycle =
            std::max(static_cast<double>(now), next_forward_cycle_);
        const double finish_cycle = start_cycle + serialization;
        request_queue_.push_back({request, arrival_cycle(
                                               finish_cycle + latency_cycles_)});
        next_forward_cycle_ = finish_cycle;
        statistics_.forward_bytes += static_cast<std::uint64_t>(transfer_bytes);
        statistics_.forward_busy_cycles += serialization;
        return true;
    }

    bool enqueue_response(const MemResponse& response, int now) {
        if (response_queue_.size() >= queue_capacity_) {
            return false;
        }
        const int transfer_bytes = response.size_bytes > 0
                                       ? response.size_bytes
                                       : kMemoryControlBytes;
        const double serialization = serialization_cycles(transfer_bytes);
        const double start_cycle =
            std::max(static_cast<double>(now), next_reverse_cycle_);
        const double finish_cycle = start_cycle + serialization;
        response_queue_.push_back({response, arrival_cycle(
                                                 finish_cycle + latency_cycles_)});
        next_reverse_cycle_ = finish_cycle;
        statistics_.reverse_bytes += static_cast<std::uint64_t>(transfer_bytes);
        statistics_.reverse_busy_cycles += serialization;
        return true;
    }

    std::vector<MemRequest> step_requests(int now) {
        std::vector<MemRequest> ready;
        while (!request_queue_.empty() && request_queue_.front().ready_cycle <= now) {
            ready.push_back(request_queue_.front().value);
            request_queue_.pop_front();
        }
        return ready;
    }

    std::vector<MemResponse> step_responses(int now) {
        std::vector<MemResponse> ready;
        while (!response_queue_.empty() && response_queue_.front().ready_cycle <= now) {
            ready.push_back(response_queue_.front().value);
            response_queue_.pop_front();
        }
        return ready;
    }

    D2DLinkStatistics statistics(int elapsed_cycles) const {
        D2DLinkStatistics result = statistics_;
        const double duration = static_cast<double>(std::max(1, elapsed_cycles));
        result.forward_utilization = std::min(
            1.0, static_cast<double>(result.forward_busy_cycles) / duration);
        result.reverse_utilization = std::min(
            1.0, static_cast<double>(result.reverse_busy_cycles) / duration);
        return result;
    }

private:
    template <typename T>
    struct Transit {
        T value;
        int ready_cycle;
    };

    double bandwidth_bytes_per_cycle_;
    int latency_cycles_;
    std::size_t queue_capacity_;
    double next_forward_cycle_{};
    double next_reverse_cycle_{};
    std::deque<Transit<MemRequest>> request_queue_;
    std::deque<Transit<MemResponse>> response_queue_;
    D2DLinkStatistics statistics_{};

    double serialization_cycles(int bytes) const {
        return static_cast<double>(bytes) / bandwidth_bytes_per_cycle_;
    }

    static int arrival_cycle(double fractional_cycle) {
        return static_cast<int>(std::ceil(fractional_cycle - 1.0e-12));
    }
};

// Owns one of the four supported topologies. Routing is explicit and fixed;
// D2DLink instances are shared by the routes that physically traverse them.
class MemorySystem {
public:
    MemorySystem(int cache_size_bytes, int line_size_bytes, int hit_latency_cycles,
                 int l2_bandwidth_bytes_per_cycle,
                 int l2_write_bandwidth_bytes_per_cycle, int l2_queue_depth,
                 bool bypass_cache, const MemorySystemConfig& memory_config,
                 double core_frequency_mhz, const TensorRegistry& tensors)
        : memory_config_(memory_config),
          tensors_(tensors),
          l2_(std::make_unique<L2Cache>(cache_size_bytes, line_size_bytes,
                                        hit_latency_cycles,
                                        l2_bandwidth_bytes_per_cycle,
                                        l2_write_bandwidth_bytes_per_cycle,
                                        l2_queue_depth, bypass_cache, tensors)) {
        validate_memory_system_config(memory_config_);
        if (!std::isfinite(core_frequency_mhz) || core_frequency_mhz <= 0.0) {
            throw std::invalid_argument("core frequency must be positive");
        }
        switch (memory_config_.topology) {
            case MemoryTopology::FullHBM: build_full_hbm(core_frequency_mhz); break;
            case MemoryTopology::FullHBF: build_full_hbf(core_frequency_mhz); break;
            case MemoryTopology::CoLocated: build_co_located(core_frequency_mhz); break;
            case MemoryTopology::Cascaded: build_cascaded(core_frequency_mhz); break;
            default: throw std::invalid_argument("unsupported memory topology");
        }
        report_placement_override();
    }

    L2Cache* l2() const { return l2_.get(); }

    std::vector<int> step(int current_cycle) {
        last_step_cycle_ = current_cycle;
        std::vector<int> completed = l2_->step(current_cycle);
        collect_l2_requests(current_cycle);
        launch_pending_requests(current_cycle);
        advance_forward_links(current_cycle);
        retry_controller_admission(current_cycle);

        if (hbm_) {
            for (const auto& response : hbm_->step(current_cycle)) {
                route_response(response, current_cycle);
            }
        }
        if (hbf_) {
            for (const auto& response : hbf_->step(current_cycle)) {
                route_response(response, current_cycle);
            }
        }

        launch_pending_responses(current_cycle);
        advance_cascaded_return(current_cycle);
        advance_gpu_return_links(current_cycle);

        // A response can evict a dirty L2 line. Start its writeback on the
        // appropriate route in this same cycle, as the integrated model did.
        collect_l2_requests(current_cycle);
        launch_pending_requests(current_cycle);
        return completed;
    }

    bool idle() const {
        return l2_->idle() && request_routes_.empty() &&
               (!hbm_ || hbm_->pending_requests() == 0);
    }

    MemoryBackendStatistics hbm_backend_statistics() const {
        return hbm_ ? hbm_->backend_statistics() : MemoryBackendStatistics{};
    }

    MemorySystemStatistics statistics(int current_cycle) const {
        MemorySystemStatistics result;
        result.hbm = hbm_statistics_;
        result.hbf = hbf_statistics_;
        const int elapsed_cycles = std::max(1, std::max(current_cycle, last_step_cycle_) + 1);
        if (gpu_to_hbm_) {
            result.gpu_to_hbm = gpu_to_hbm_->statistics(elapsed_cycles);
            result.has_gpu_to_hbm = true;
        }
        if (gpu_to_hbf_) {
            result.gpu_to_hbf = gpu_to_hbf_->statistics(elapsed_cycles);
            result.has_gpu_to_hbf = true;
        }
        if (hbm_to_hbf_) {
            result.hbm_to_hbf = hbm_to_hbf_->statistics(elapsed_cycles);
            result.has_hbm_to_hbf = true;
        }
        return result;
    }

private:
    struct RequestRoute {
        MemoryTier tier;
        int start_cycle;
    };

    int last_step_cycle_{};
    MemorySystemConfig memory_config_;
    const TensorRegistry& tensors_;
    std::unique_ptr<L2Cache> l2_;
    std::unique_ptr<MemoryBackend> hbm_;
    std::unique_ptr<HBFModel> hbf_;
    std::unique_ptr<D2DLink> gpu_to_hbm_;
    std::unique_ptr<D2DLink> gpu_to_hbf_;
    std::unique_ptr<D2DLink> hbm_to_hbf_;
    std::deque<MemRequest> pending_to_hbm_;
    std::deque<MemRequest> pending_to_hbf_;
    std::deque<MemRequest> pending_to_base_;
    std::deque<MemRequest> pending_hbm_to_hbf_;
    std::deque<MemRequest> pending_hbm_device_;
    std::deque<MemRequest> pending_hbf_device_;
    std::deque<MemResponse> pending_hbm_response_;
    std::deque<MemResponse> pending_hbf_response_;
    std::deque<MemResponse> pending_hbm_to_hbf_response_;
    std::unordered_map<std::uint64_t, RequestRoute> request_routes_;
    TierMemoryStatistics hbm_statistics_{};
    TierMemoryStatistics hbf_statistics_{};

    void build_full_hbm(double core_frequency_mhz) {
        hbm_ = make_hbm_backend(memory_config_.hbm,
                                 core_frequency_mhz);
        gpu_to_hbm_ = std::make_unique<D2DLink>(memory_config_.gpu_to_hbm,
                                                 core_frequency_mhz);
    }

    void build_full_hbf(double core_frequency_mhz) {
        hbf_ = std::make_unique<HBFModel>(memory_config_.hbf,
                                          core_frequency_mhz);
        gpu_to_hbf_ = std::make_unique<D2DLink>(memory_config_.gpu_to_hbf,
                                                 core_frequency_mhz);
    }

    void build_co_located(double core_frequency_mhz) {
        hbm_ = make_hbm_backend(memory_config_.hbm,
                                 core_frequency_mhz);
        hbf_ = std::make_unique<HBFModel>(memory_config_.hbf,
                                          core_frequency_mhz);
        gpu_to_hbm_ = std::make_unique<D2DLink>(memory_config_.gpu_to_hbm,
                                                 core_frequency_mhz);
        gpu_to_hbf_ = std::make_unique<D2DLink>(memory_config_.gpu_to_hbf,
                                                 core_frequency_mhz);
    }

    void build_cascaded(double core_frequency_mhz) {
        hbm_ = make_hbm_backend(memory_config_.hbm,
                                 core_frequency_mhz);
        hbf_ = std::make_unique<HBFModel>(memory_config_.hbf,
                                          core_frequency_mhz);
        gpu_to_hbm_ = std::make_unique<D2DLink>(memory_config_.gpu_to_hbm,
                                                 core_frequency_mhz);
        hbm_to_hbf_ = std::make_unique<D2DLink>(memory_config_.hbm_to_hbf,
                                                 core_frequency_mhz);
    }

    MemoryTier effective_tier(MemoryTier requested) const {
        if (memory_config_.topology == MemoryTopology::FullHBM) {
            return MemoryTier::HBM;
        }
        if (memory_config_.topology == MemoryTopology::FullHBF) {
            return MemoryTier::HBF;
        }
        return requested;
    }

    void report_placement_override() const {
        if (memory_config_.topology == MemoryTopology::FullHBM &&
            tensors_.contains_tier(MemoryTier::HBF)) {
            std::clog << "MemorySystem: automatic placement override: "
                         "FullHBM routes HBF-tagged allocations to HBM.\n";
        } else if (memory_config_.topology == MemoryTopology::FullHBF &&
                   tensors_.contains_tier(MemoryTier::HBM)) {
            std::clog << "MemorySystem: automatic placement override: "
                         "FullHBF routes HBM-tagged allocations to HBF.\n";
        }
    }

    TierMemoryStatistics& statistics_for(MemoryTier tier) {
        return tier == MemoryTier::HBM ? hbm_statistics_ : hbf_statistics_;
    }

    void route_request(const MemRequest& request, int current_cycle) {
        const MemoryTier tier = effective_tier(request.tier);
        const auto [it, inserted] = request_routes_.emplace(
            request.request_id, RequestRoute{tier, current_cycle});
        if (!inserted) {
            throw std::logic_error("duplicate memory request id in MemorySystem");
        }
        TierMemoryStatistics& stats = statistics_for(tier);
        ++stats.request_count;
        if (request.operation == MemoryOperation::Read) {
            ++stats.read_requests;
            stats.read_bytes += static_cast<std::uint64_t>(
                std::max(1, request.size_bytes));
        } else {
            ++stats.write_requests;
            stats.write_bytes += static_cast<std::uint64_t>(
                std::max(1, request.size_bytes));
        }

        switch (memory_config_.topology) {
            case MemoryTopology::FullHBM:
            case MemoryTopology::CoLocated:
                if (tier == MemoryTier::HBM) pending_to_hbm_.push_back(request);
                else pending_to_hbf_.push_back(request);
                break;
            case MemoryTopology::FullHBF:
                pending_to_hbf_.push_back(request);
                break;
            case MemoryTopology::Cascaded:
                pending_to_base_.push_back(request);
                break;
        }
    }

    void collect_l2_requests(int current_cycle) {
        for (const auto& request : l2_->take_backend_requests()) {
            route_request(request, current_cycle);
        }
    }

    static void launch_requests(D2DLink* link, std::deque<MemRequest>& pending,
                                int current_cycle) {
        while (link && !pending.empty() &&
               link->enqueue_request(pending.front(), current_cycle)) {
            pending.pop_front();
        }
    }

    static void launch_responses(D2DLink* link,
                                 std::deque<MemResponse>& pending,
                                 int current_cycle) {
        while (link && !pending.empty() &&
               link->enqueue_response(pending.front(), current_cycle)) {
            pending.pop_front();
        }
    }

    void launch_pending_requests(int current_cycle) {
        switch (memory_config_.topology) {
            case MemoryTopology::FullHBM:
                launch_requests(gpu_to_hbm_.get(), pending_to_hbm_, current_cycle);
                break;
            case MemoryTopology::FullHBF:
                launch_requests(gpu_to_hbf_.get(), pending_to_hbf_, current_cycle);
                break;
            case MemoryTopology::CoLocated:
                launch_requests(gpu_to_hbm_.get(), pending_to_hbm_, current_cycle);
                launch_requests(gpu_to_hbf_.get(), pending_to_hbf_, current_cycle);
                break;
            case MemoryTopology::Cascaded:
                launch_requests(gpu_to_hbm_.get(), pending_to_base_, current_cycle);
                launch_requests(hbm_to_hbf_.get(), pending_hbm_to_hbf_,
                                current_cycle);
                break;
        }
    }

    void route_first_hop_arrivals(const std::vector<MemRequest>& requests) {
        for (const auto& request : requests) {
            const auto route = request_routes_.find(request.request_id);
            if (route == request_routes_.end()) {
                throw std::logic_error("memory request lost its route metadata");
            }
            if (memory_config_.topology == MemoryTopology::Cascaded &&
                route->second.tier == MemoryTier::HBF) {
                // The HBM base die forwards HBF traffic without submitting it
                // to HBM DRAM or consuming HBM controller bandwidth.
                pending_hbm_to_hbf_.push_back(request);
            } else if (route->second.tier == MemoryTier::HBM) {
                pending_hbm_device_.push_back(request);
            } else {
                pending_hbf_device_.push_back(request);
            }
        }
    }

    void advance_forward_links(int current_cycle) {
        switch (memory_config_.topology) {
            case MemoryTopology::FullHBM:
                route_first_hop_arrivals(
                    gpu_to_hbm_->step_requests(current_cycle));
                break;
            case MemoryTopology::FullHBF:
                route_first_hop_arrivals(
                    gpu_to_hbf_->step_requests(current_cycle));
                break;
            case MemoryTopology::CoLocated:
                route_first_hop_arrivals(
                    gpu_to_hbm_->step_requests(current_cycle));
                route_first_hop_arrivals(
                    gpu_to_hbf_->step_requests(current_cycle));
                break;
            case MemoryTopology::Cascaded:
                route_first_hop_arrivals(
                    gpu_to_hbm_->step_requests(current_cycle));
                launch_requests(hbm_to_hbf_.get(), pending_hbm_to_hbf_,
                                current_cycle);
                for (const auto& request :
                     hbm_to_hbf_->step_requests(current_cycle)) {
                    pending_hbf_device_.push_back(request);
                }
                break;
        }
    }

    void retry_controller_admission(int current_cycle) {
        while (hbm_ && !pending_hbm_device_.empty() &&
               hbm_->try_submit(pending_hbm_device_.front(), current_cycle)) {
            pending_hbm_device_.pop_front();
        }
        while (hbf_ && !pending_hbf_device_.empty() &&
               hbf_->try_submit(pending_hbf_device_.front(), current_cycle)) {
            pending_hbf_device_.pop_front();
        }
    }

    void route_response(const MemResponse& response, int /*current_cycle*/) {
        const auto route = request_routes_.find(response.request_id);
        if (route == request_routes_.end()) {
            throw std::logic_error("memory response lost its route metadata");
        }
        switch (memory_config_.topology) {
            case MemoryTopology::FullHBM:
                pending_hbm_response_.push_back(response);
                break;
            case MemoryTopology::FullHBF:
                pending_hbf_response_.push_back(response);
                break;
            case MemoryTopology::CoLocated:
                if (route->second.tier == MemoryTier::HBM) {
                    pending_hbm_response_.push_back(response);
                } else {
                    pending_hbf_response_.push_back(response);
                }
                break;
            case MemoryTopology::Cascaded:
                if (route->second.tier == MemoryTier::HBM) {
                    pending_hbm_response_.push_back(response);
                } else {
                    pending_hbm_to_hbf_response_.push_back(response);
                }
                break;
        }
    }

    void launch_pending_responses(int current_cycle) {
        switch (memory_config_.topology) {
            case MemoryTopology::FullHBM:
                launch_responses(gpu_to_hbm_.get(), pending_hbm_response_,
                                 current_cycle);
                break;
            case MemoryTopology::FullHBF:
                launch_responses(gpu_to_hbf_.get(), pending_hbf_response_,
                                 current_cycle);
                break;
            case MemoryTopology::CoLocated:
                launch_responses(gpu_to_hbm_.get(), pending_hbm_response_,
                                 current_cycle);
                launch_responses(gpu_to_hbf_.get(), pending_hbf_response_,
                                 current_cycle);
                break;
            case MemoryTopology::Cascaded:
                launch_responses(hbm_to_hbf_.get(),
                                 pending_hbm_to_hbf_response_, current_cycle);
                launch_responses(gpu_to_hbm_.get(), pending_hbm_response_,
                                 current_cycle);
                break;
        }
    }

    void advance_cascaded_return(int current_cycle) {
        if (!hbm_to_hbf_) {
            return;
        }
        for (const auto& response : hbm_to_hbf_->step_responses(current_cycle)) {
            pending_hbm_response_.push_back(response);
        }
        launch_responses(gpu_to_hbm_.get(), pending_hbm_response_,
                         current_cycle);
    }

    void advance_gpu_return_links(int current_cycle) {
        std::vector<MemResponse> responses;
        switch (memory_config_.topology) {
            case MemoryTopology::FullHBM:
                responses = gpu_to_hbm_->step_responses(current_cycle);
                break;
            case MemoryTopology::FullHBF:
                responses = gpu_to_hbf_->step_responses(current_cycle);
                break;
            case MemoryTopology::CoLocated: {
                auto hbm_responses = gpu_to_hbm_->step_responses(current_cycle);
                auto hbf_responses = gpu_to_hbf_->step_responses(current_cycle);
                responses.reserve(hbm_responses.size() + hbf_responses.size());
                responses.insert(responses.end(), hbm_responses.begin(),
                                 hbm_responses.end());
                responses.insert(responses.end(), hbf_responses.begin(),
                                 hbf_responses.end());
                break;
            }
            case MemoryTopology::Cascaded:
                responses = gpu_to_hbm_->step_responses(current_cycle);
                break;
        }

        for (const auto& response : responses) {
            const auto route = request_routes_.find(response.request_id);
            if (route == request_routes_.end()) {
                throw std::logic_error("completed response has no request route");
            }
            TierMemoryStatistics& stats = statistics_for(route->second.tier);
            ++stats.completed_requests;
            stats.total_latency_cycles += static_cast<std::uint64_t>(
                std::max(0, current_cycle - route->second.start_cycle));
            request_routes_.erase(route);
            l2_->handle_response(response, current_cycle);
        }
    }
};

} // namespace GTSim

#endif // MEMORY_H
