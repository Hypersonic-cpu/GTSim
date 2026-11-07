#pragma once

#include "memory_backend.h"
#include <hbfine/address.hpp>

namespace GTSim {
// Physical SRAM quotas and tile leases are shared by ordinary and banked prefetch.
class ReadSramAllocator {
public:
    struct Page {
        int unit;
        bool plane;
    };
    struct Reservation {
        std::vector<Page> pages;
        std::optional<PrefetchBuffer> buffer;
    };

    ReadSramAllocator(const hbfine::Config& config, std::uint64_t base)
        : config_(config),
          base_(base) {}

    std::optional<Reservation> reserve(std::uint64_t address,
                                       std::uint64_t bytes,
                                       std::optional<PrefetchBuffer> buffer) {
        const int count = buffer ? buffer->count : 1;
        const int index = buffer ? buffer->slot : 0;
        if (count <= 0 || index < 0 || index >= count ||
            (count_ && count != count_))
            throw std::invalid_argument("invalid SRAM buffer partition");
        count_ = count;
        auto& slot = slots_[index];
        if (buffer && slot.owner && *slot.owner != buffer->epoch)
            return std::nullopt;
        if (slot.planes.empty()) {
            slot.planes.resize(config_.stacks * config_.dies_per_stack *
                               config_.planes_per_die);
            slot.stacks.resize(config_.stacks);
        }
        auto planes = slot.planes;
        auto stacks = slot.stacks;
        const auto plane_pages =
            config_.sram_bytes_per_plane / config_.page_bytes;
        const auto stack_limit = config_.base_die_sram_bytes_per_stack / count;
        Reservation reservation{{}, buffer};
        for (auto page = address / config_.page_bytes * config_.page_bytes;
             page < address + bytes;) {
            const auto a = hbfine::decode_address(page, base_, config_);
            const auto plane = hbfine::physical_plane_id(a, config_);
            const auto size = static_cast<std::uint64_t>(config_.page_bytes);
            // Allocate whole pages; rotate remainder pages across plane IDs
            // so a one-page plane is not split into unusable half-pages.
            const auto extra =
                static_cast<std::uint64_t>((index + count - plane % count) %
                                           count) < plane_pages % count;
            const auto plane_limit = (plane_pages / count + extra) * size;
            if (size <= plane_limit - planes[plane]) {
                planes[plane] += size;
                reservation.pages.push_back({plane, true});
            } else if (size <= stack_limit - stacks[a.stack]) {
                stacks[a.stack] += size;
                reservation.pages.push_back({a.stack, false});
            } else
                return std::nullopt;
            if (address + bytes - page <= size) break;
            page += size;
        }
        slot.planes = std::move(planes);
        slot.stacks = std::move(stacks);
        if (buffer) slot.owner = buffer->epoch;
        ++slot.live;
        slot.bytes += reservation.pages.size() * config_.page_bytes;
        slot.peak = std::max(slot.peak, slot.bytes);
        return reservation;
    }

    void release(const Reservation& reservation) {
        const auto index = reservation.buffer ? reservation.buffer->slot : 0;
        auto& slot = slots_.at(index);
        for (const auto page : reservation.pages) {
            auto& used = page.plane ? slot.planes : slot.stacks;
            used[page.unit] -= config_.page_bytes;
        }
        slot.bytes -= reservation.pages.size() * config_.page_bytes;
        if (!--slot.live) slot.owner.reset();
    }

    std::map<std::string, std::uint64_t> gauges() const {
        std::map<std::string, std::uint64_t> result;
        if (count_ <= 1) return result;
        result.emplace("buffer_count", count_);
        for (const auto& [index, slot] : slots_) {
            const auto prefix = "buffer_" + std::to_string(index);
            result.emplace(prefix + "_reserved_bytes", slot.bytes);
            result.emplace(prefix + "_peak_reserved_bytes", slot.peak);
            result.emplace(prefix + "_live_regions", slot.live);
            if (slot.owner) result.emplace(prefix + "_epoch", *slot.owner);
        }
        return result;
    }

private:
    struct Slot {
        std::vector<std::uint64_t> planes, stacks;
        std::optional<std::uint64_t> owner;
        std::size_t live{};
        std::uint64_t bytes{}, peak{};
    };
    const hbfine::Config& config_;
    std::uint64_t base_;
    int count_{};
    std::map<int, Slot> slots_;
};
} // namespace GTSim
