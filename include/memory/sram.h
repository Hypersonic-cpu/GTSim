#pragma once

#include "memory_backend.h"
#include <functional>

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

    struct Config {
        int stacks{}, dies_per_stack{}, planes_per_die{}, page_bytes{};
        std::uint64_t sram_bytes_per_plane{}, base_die_sram_bytes_per_stack{};
    };
    ReadSramAllocator(Config config,
                      std::function<MemoryLocation(std::uint64_t)> locate)
        : config_(config),
          locate_(std::move(locate)) {}

    std::optional<Reservation> reserve(std::uint64_t address,
                                       std::uint64_t bytes,
                                       std::optional<PrefetchBuffer> buffer) {
        const int count = buffer ? buffer->count : 1;
        const int index = buffer ? buffer->slot : 0;
        if (count <= 0 || index < 0 || index >= count ||
            (count_ && count != count_))
            throw std::invalid_argument("invalid SRAM buffer partition");
        if (!count_) {
            count_ = count;
            slots_.resize(static_cast<std::size_t>(count_));
        }
        auto& slot = slots_.at(static_cast<std::size_t>(index));
        if (buffer && slot.owner && *slot.owner != buffer->epoch)
            return std::nullopt;
        if (slot.planes.empty()) {
            slot.planes.resize(config_.stacks * config_.dies_per_stack *
                               config_.planes_per_die);
            slot.stacks.resize(config_.stacks);
        }
        if (plane_delta_.size() != slot.planes.size())
            plane_delta_.assign(slot.planes.size(), 0);
        if (stack_delta_.size() != slot.stacks.size())
            stack_delta_.assign(slot.stacks.size(), 0);
        touched_planes_.clear();
        touched_stacks_.clear();
        const auto clear_delta = [this]() {
            for (const auto plane : touched_planes_) plane_delta_[plane] = 0;
            for (const auto stack : touched_stacks_) stack_delta_[stack] = 0;
            touched_planes_.clear();
            touched_stacks_.clear();
        };
        const auto plane_pages =
            config_.sram_bytes_per_plane / config_.page_bytes;
        const auto stack_limit = config_.base_die_sram_bytes_per_stack / count;
        Reservation reservation{{}, buffer};
        for (auto page = address / config_.page_bytes * config_.page_bytes;
             page < address + bytes;) {
            const auto a = locate_(page);
            const auto plane = (a.stack * config_.dies_per_stack + a.die) *
                                   config_.planes_per_die +
                               a.plane;
            const auto size = static_cast<std::uint64_t>(config_.page_bytes);
            // Allocate whole pages; rotate remainder pages across plane IDs
            // so a one-page plane is not split into unusable half-pages.
            const auto extra =
                static_cast<std::uint64_t>((index + count - plane % count) %
                                           count) < plane_pages % count;
            const auto plane_limit = (plane_pages / count + extra) * size;
            const auto plane_used = slot.planes[plane] + plane_delta_[plane];
            if (plane_used <= plane_limit && size <= plane_limit - plane_used) {
                if (plane_delta_[plane] == 0) touched_planes_.push_back(plane);
                plane_delta_[plane] += size;
                reservation.pages.push_back({plane, true});
            } else {
                const auto stack_used =
                    slot.stacks[a.stack] + stack_delta_[a.stack];
                if (stack_used > stack_limit ||
                    size > stack_limit - stack_used) {
                    clear_delta();
                    return std::nullopt;
                }
                if (stack_delta_[a.stack] == 0)
                    touched_stacks_.push_back(a.stack);
                stack_delta_[a.stack] += size;
                reservation.pages.push_back({a.stack, false});
            }
            if (address + bytes - page <= size) break;
            page += size;
        }
        for (const auto plane : touched_planes_)
            slot.planes[plane] += plane_delta_[plane];
        for (const auto stack : touched_stacks_)
            slot.stacks[stack] += stack_delta_[stack];
        clear_delta();
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
        for (std::size_t index = 0; index < slots_.size(); ++index) {
            const auto& slot = slots_[index];
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
    Config config_;
    std::function<MemoryLocation(std::uint64_t)> locate_;
    int count_{};
    std::vector<Slot> slots_;
    std::vector<std::uint64_t> plane_delta_, stack_delta_;
    std::vector<int> touched_planes_, touched_stacks_;
};
} // namespace GTSim
