#ifndef GTSIM_MEMORY_LAYOUT_H
#define GTSIM_MEMORY_LAYOUT_H

#include <cstdint>
#include <limits>
#include <initializer_list>
#include <stdexcept>

namespace GTSim {

enum class MemoryTier : std::uint8_t { HBM, HBF };

// Optional legacy mode serves existing low-level graph/registry clients. Public
// workloads explicitly enable segmented allocation, including all-HBM runs.
struct MemoryAddressLayout {
    bool enabled{false};
    std::uint64_t hbm_base{0};
    std::uint64_t hbm_capacity_bytes{80ULL << 30};
    std::uint64_t hbf_base{80ULL << 30};
    std::uint64_t hbf_capacity_bytes{2560ULL << 30};
    std::uint64_t hbm_allocation_offset{1ULL << 32};

    bool operator==(const MemoryAddressLayout&) const = default;

    std::uint64_t base(MemoryTier tier) const { return tier == MemoryTier::HBM ? hbm_base : hbf_base; }
    std::uint64_t capacity(MemoryTier tier) const {
        return tier == MemoryTier::HBM ? hbm_capacity_bytes : hbf_capacity_bytes;
    }
    void validate() const {
        if (!enabled) return;
        if (hbm_base % 4096 || hbf_base % 4096 || hbm_capacity_bytes % 4096 ||
            hbf_capacity_bytes % 4096 || hbm_allocation_offset % 4096 ||
            hbm_allocation_offset > hbm_capacity_bytes ||
            hbm_base > std::numeric_limits<std::uint64_t>::max() - hbm_capacity_bytes ||
            hbf_base > std::numeric_limits<std::uint64_t>::max() - hbf_capacity_bytes ||
            hbf_base != hbm_base + hbm_capacity_bytes) {
            throw std::invalid_argument("memory segments must be aligned, contiguous, nonoverlapping and in range");
        }
    }
    MemoryTier resolve(std::uint64_t address, std::uint64_t bytes = 1) const {
        for (auto tier : {MemoryTier::HBM, MemoryTier::HBF}) {
            if (bytes > 0 && address >= base(tier) && address - base(tier) < capacity(tier) &&
                bytes <= capacity(tier) - (address - base(tier))) return tier;
        }
        throw std::out_of_range("memory access is outside a device segment or crosses a tier boundary");
    }
};

} // namespace GTSim
#endif
