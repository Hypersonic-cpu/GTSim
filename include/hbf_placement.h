#pragma once

#include "hbf_address.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace GTSim {

struct HBFPhysicalSpan {
    std::uint64_t address;
    std::uint64_t bytes;
};

// A tensor occupies a contiguous run of physical Megapages. The address codec
// stripes those pages over command groups, independent of tensor indexing.
struct HBFTensorPlacement {
    std::uint64_t base;
    std::uint64_t logical_bytes;
    std::uint64_t megapage_bytes;

    HBFPhysicalSpan map(std::uint64_t logical_offset) const {
        if (logical_offset >= logical_bytes)
            throw std::out_of_range("HBF tensor offset outside placement");
        return {base + logical_offset,
                std::min(logical_bytes - logical_offset,
                         megapage_bytes - logical_offset % megapage_bytes)};
    }
};

class HBFMegapageAllocator {
public:
    HBFMegapageAllocator(std::uint64_t base, const hbfine::Config& config)
        : base_(base) {
        hbfine::validate_config(config);
        if (config.geometry != hbfine::GeometryProfile::GroupedPlanes)
            throw std::invalid_argument(
                "Megapage placement requires grouped HBF geometry");
        megapage_bytes_ = static_cast<std::uint64_t>(config.page_bytes) *
                          config.planes_per_command_group;
        capacity_ = hbfine::capacity_bytes_per_stack(config) * config.stacks;
        if (base % config.page_bytes ||
            base > std::numeric_limits<std::uint64_t>::max() - capacity_)
            throw std::invalid_argument("invalid HBF placement aperture");
    }

    // Allocate in consumer order. The cursor continues across tensor boundaries.
    HBFTensorPlacement allocate(std::uint64_t bytes) {
        if (!bytes) throw std::invalid_argument("empty HBF tensor placement");
        const auto pages = 1 + (bytes - 1) / megapage_bytes_;
        if (pages > (capacity_ - cursor_) / megapage_bytes_)
            throw std::out_of_range("HBF placement exceeds device capacity");
        const HBFTensorPlacement placement{base_ + cursor_, bytes,
                                           megapage_bytes_};
        cursor_ += pages * megapage_bytes_;
        return placement;
    }

private:
    std::uint64_t base_;
    std::uint64_t megapage_bytes_{};
    std::uint64_t capacity_{};
    std::uint64_t cursor_{};
};

} // namespace GTSim
