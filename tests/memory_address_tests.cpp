#include "memory.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

using namespace GTSim;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    try {
        TensorRegistry tensors;
        const TensorMetadata& tensor = tensors.allocate_tensor(
            7, "strided", {5, 8}, {16, 2}, DataType::FP16, MemoryTier::HBF);
        DAGNode node(3, "tile", "LD", "cp.dram2sram_tma", 0, {}, 0,
                     Tile(1, 2, 2, 3));
        node.matrix_id = tensor.id;

        const std::uint64_t first = compute_global_element_addr(node, tensor, 0);
        const std::uint64_t next_row = compute_global_element_addr(node, tensor, 3);
        require(first == tensor.global_base_address + 40,
                "tile offsets and non-contiguous tensor strides must form the global address");
        require(next_row == tensor.global_base_address + 72,
                "row-major tile enumeration must advance through tensor strides");
        require(tensors.resolve_tier(first) == MemoryTier::HBF,
                "tensor address ranges must preserve the registered memory tier");

        const auto lines = compute_dram_subop_lines(node, tensor, 4, 0, 128);
        require(lines.size() == 1, "addresses within one cache line must coalesce");
        require(lines.front().global_line_addr == tensor.global_base_address,
                "global cache-line address must be aligned to the cache-line size");
        require(lines.front().global_line_addr % 128 == 0,
                "global cache-line address must be aligned");

        const CacheLineKey first_key{7, 128, tensor.global_base_address};
        const CacheLineKey aliased_key{99, 256, tensor.global_base_address};
        require(first_key == aliased_key,
                "cache-line identity must use the global address, not legacy fields");
        std::unordered_set<CacheLineKey, CacheLineKeyHash> keys;
        keys.insert(first_key);
        keys.insert(aliased_key);
        require(keys.size() == 1, "global aliases must hash to one cache-line identity");

        MemorySystemConfig memory_config;
        memory_config.hbm = {10000.0, 10000.0, 0, 0, 8};
        memory_config.gpu_to_hbm = {10000.0, 0, 8};
        MemorySystem memory(0, 128, 0, 256, 128, 8, true,
                            memory_config, 1000.0, tensors);
        require(memory.l2()->enqueue_transaction_key(10, first_key, false),
                "first L2 access should enqueue");
        require(memory.l2()->enqueue_transaction_key(11, aliased_key, false),
                "second L2 access should enqueue");
        std::vector<int> completed;
        for (int cycle = 0; cycle < 16 && completed.size() < 2; ++cycle) {
            auto cycle_completed = memory.step(cycle);
            completed.insert(completed.end(), cycle_completed.begin(),
                             cycle_completed.end());
        }
        std::sort(completed.begin(), completed.end());
        require(completed == std::vector<int>({10, 11}),
                "same-line accesses must share one HBF-tagged backend request and response");
        require(memory.statistics(16).hbm.request_count == 1,
                "merged HBF-tagged access must use FullHBM's single controller");

        std::cout << "Global memory address tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Global memory address test failure: " << error.what() << '\n';
        return 1;
    }
}
