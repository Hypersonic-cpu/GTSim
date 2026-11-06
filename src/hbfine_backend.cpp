#include "memory_backend.h"
#include "hbf_placement.h"
#include "hbfine_sram.h"
#include <optional>
#include <hbfine/controller.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace GTSim {
namespace {
// The device uses physical time and knows nothing about GPU cycles or tensors.
// This adapter preserves the simulator's response metadata and clock boundary.
class HBFineBackend final : public MemoryBackend {
public:
    HBFineBackend(const HBFConfig& config, double frequency_mhz,
                  std::uint64_t base)
        : frequency_khz_(clock_khz(frequency_mhz)),
          config_(config.controller),
          base_(base),
          device_(config_, base, request_capacity(config)),
          sram_(config_, base) {}

    bool try_submit(const MemRequest& request, int now) override {
        if (request.tier != MemoryTier::HBF) throw std::invalid_argument("HBFineBackend received a non-HBF request");
        if (pending_.contains(request.request_id))
            throw std::logic_error("duplicate HBFine request ID");
        if (background_.contains(request.request_id)) return false;
        hbfine::Operation operation;
        switch (request.operation) {
        case MemoryOperation::Read: operation = hbfine::Operation::Read; break;
        case MemoryOperation::Write: operation = hbfine::Operation::Write; break;
        case MemoryOperation::Writeback: operation = hbfine::Operation::Writeback; break;
        default: throw std::invalid_argument("unsupported HBF operation");
        }
        auto path = hbfine::ReadPath::Host;
        if (request.read_alias) {
            const auto& region = regions_.at(request.read_alias);
            if (region.released || operation != hbfine::Operation::Read ||
                request.global_addr < region.address ||
                request.global_addr - region.address >= region.bytes ||
                static_cast<std::uint64_t>(request.size_bytes) >
                    region.bytes - (request.global_addr - region.address))
                throw std::invalid_argument("invalid read alias");
            if (region.ready()) path = hbfine::ReadPath::Resident;
        }
        if (!device_.try_submit({request.request_id, request.global_addr, request.size_bytes,
                                 operation, request.routing.axi_id, path}, physical_time(now))) return false;
        if (request.read_alias) {
            ++regions_.at(request.read_alias).consumers;
            if (path == hbfine::ReadPath::Resident)
                resident_read_bytes_ += request.size_bytes;
            else
                fallback_read_bytes_ += request.size_bytes;
        }
        pending_.emplace(request.request_id, request);
        return true;
    }

    std::vector<MemResponse> step(int now) override {
        const auto time = physical_time(now);
        submit_background(time);
        const auto completions = device_.advance(time);
        std::vector<MemResponse> responses;
        responses.reserve(completions.size());
        for (const auto& completion : completions) {
            if (const auto it = background_.find(completion.request_id); it != background_.end()) {
                const auto token = it->second;
                background_.erase(it);
                if (completion.error != hbfine::Error::None)
                    throw std::runtime_error("device prefetch failed");
                auto& region = regions_.at(token);
                --region.outstanding;
                if (region.ready()) {
                    --active_fills_;
                    ++completed_fills_;
                }
                reclaim(token);
                continue;
            }
            const auto& request = pending_.at(completion.request_id);
            responses.push_back({request.request_id, request.global_addr, request.tier,
                                 completion.error == hbfine::Error::None,
                                 request.operation == MemoryOperation::Read ? request.size_bytes : 16,
                                 request.operation, static_cast<int>(completion.error)});
            const auto alias = request.read_alias;
            pending_.erase(completion.request_id);
            if (alias) {
                --regions_.at(alias).consumers;
                reclaim(alias);
            }
        }
        return responses;
    }
    std::size_t pending_requests() const override { return pending_.size(); }
    std::uint64_t
    reserve_prefetch(std::uint64_t address, std::uint64_t bytes,
                     std::optional<PrefetchBuffer> buffer) override {
        if (!config_.enable_read_sram || !bytes || address % 64 || bytes % 64 ||
            address > std::numeric_limits<std::uint64_t>::max() - bytes)
            throw std::invalid_argument("prefetch requires enabled SRAM and a nonempty aligned range");
        hbfine::decode_address(address + bytes - 1, base_, config_);
        Region region;
        region.address = address;
        region.bytes = bytes;
        auto reservation = sram_.reserve(address, bytes, buffer);
        if (!reservation) return 0;
        region.reservation = std::move(*reservation);
        const auto token = next_token_++;
        reserved_bytes_ += region.reservation.pages.size() * config_.page_bytes;
        peak_reserved_bytes_ = std::max(peak_reserved_bytes_, reserved_bytes_);
        regions_.emplace(token, std::move(region));
        return token;
    }

    void start_prefetch(std::uint64_t token) override {
        auto& region = regions_.at(token);
        if (region.started || region.released)
            throw std::invalid_argument("invalid prefetch start");
        region.started = true;
        ++started_fills_;
        ++active_fills_;
        peak_active_fills_ = std::max(peak_active_fills_, active_fills_);
    }

    bool prefetch_ready(std::uint64_t token) const override {
        const auto& region = regions_.at(token);
        return region.ready() && !region.released;
    }

    void release_prefetch(std::uint64_t token) override {
        auto& region = regions_.at(token);
        if (region.released) throw std::invalid_argument("duplicate prefetch release");
        region.released = true;
        reclaim(token);
    }

    std::uint64_t residency_epoch() const override { return residency_epoch_; }

    std::size_t pending_background_requests() const override {
        return active_fills_;
    }
    DeviceCounters device_counters() const override {
        const auto& stats = device_.statistics();
        DeviceCounters counters{{{"sensed_bytes", stats.sensed_bytes},
                                 {"sensed_pages", stats.senses},
                                 {"sense_commands", stats.sense_commands}},
                                stats.bank_busy_ps,
                                {}};
        if (config_.geometry == hbfine::GeometryProfile::GroupedPlanes) {
            counters.cumulative.emplace("unique_sense_addresses",
                                        stats.unique_sense_addresses);
            counters.cumulative.emplace("repeated_sense_commands",
                                        stats.sense_commands -
                                            stats.unique_sense_addresses);
        }
        if (config_.enable_read_sram) {
            counters.cumulative.emplace("prefetch_bytes", prefetch_bytes_);
            counters.cumulative.emplace("prefetch_started", started_fills_);
            counters.cumulative.emplace("prefetch_completed", completed_fills_);
            counters.gauges = {{"sram_reserved_bytes", reserved_bytes_},
                               {"sram_peak_reserved_bytes", peak_reserved_bytes_},
                               {"prefetch_peak_outstanding", peak_active_fills_}};
            counters.gauges.merge(sram_.gauges());
            counters.cumulative.emplace("resident_read_bytes", resident_read_bytes_);
            counters.cumulative.emplace("fallback_read_bytes", fallback_read_bytes_);
        }
        return counters;
    }
    HBFBackendStatistics hbf_statistics() const override { return device_.statistics(); }
    std::vector<MemoryBandwidthSample> bandwidth_samples() const override {
        std::vector<MemoryBandwidthSample> result;
        for (const auto& bin : device_.statistics().bandwidth_bins)
            result.push_back({bin.start_ps,
                              bin.end_ps,
                              {{"media_read", bin.media_read_bytes},
                               {"demand_read", bin.host_read_bytes},
                               {"prefetch_fill", bin.prefetch_fill_bytes},
                               {"resident_read", bin.resident_read_bytes}},
                              {{"sense_commands", bin.sense_commands}}});
        return result;
    }
    void preload_region(std::uint64_t address, std::uint64_t bytes, bool /*read_only*/) override {
        device_.preload_region(address, bytes);
    }

private:
    static constexpr std::uint64_t background_id_base = std::uint64_t{1} << 63;
    struct Region {
        std::uint64_t address{}, bytes{}, submitted{}, outstanding{}, consumers{};
        bool started{}, released{};
        ReadSramAllocator::Reservation reservation;
        bool ready() const { return started && submitted == bytes && !outstanding; }
    };

    void submit_background(hbfine::Time time) {
        for (auto& [token, region] : regions_) {
            while (region.started && region.submitted < region.bytes) {
                const auto address = region.address + region.submitted;
                const auto bytes = std::min<std::uint64_t>(
                    config_.page_bytes - address % config_.page_bytes,
                    region.bytes - region.submitted);
                while (pending_.contains(next_background_id_) ||
                       background_.contains(next_background_id_))
                    ++next_background_id_;
                const auto id = next_background_id_;
                if (!device_.try_submit({id, address, static_cast<int>(bytes),
                                         hbfine::Operation::Read, -1,
                                         hbfine::ReadPath::Prefetch}, time))
                    break;
                ++next_background_id_;
                background_.emplace(id, token);
                region.submitted += bytes;
                ++region.outstanding;
                prefetch_bytes_ += bytes;
            }
        }
    }

    void reclaim(std::uint64_t token) {
        const auto& region = regions_.at(token);
        if (!region.released || region.consumers || (region.started && !region.ready()))
            return;
        sram_.release(region.reservation);
        reserved_bytes_ -= region.reservation.pages.size() * config_.page_bytes;
        regions_.erase(token);
        ++residency_epoch_;
    }

    static std::size_t request_capacity(const HBFConfig& config) {
        if (config.queue_capacity <= 0) throw std::invalid_argument("invalid HBF request capacity");
        return static_cast<std::size_t>(config.queue_capacity);
    }
    static std::uint64_t clock_khz(double mhz) {
        if (!std::isfinite(mhz) || mhz < 0.001 || mhz > 1000000)
            throw std::invalid_argument("invalid HBF core clock");
        return static_cast<std::uint64_t>(std::llround(mhz * 1000));
    }
    hbfine::Time physical_time(int cycle) const {
        if (cycle < 0) throw std::invalid_argument("negative HBF core cycle");
        return static_cast<std::uint64_t>(cycle) * 1000000000ULL / frequency_khz_;
    }
    std::uint64_t frequency_khz_;
    hbfine::Config config_;
    std::uint64_t base_;
    hbfine::Controller device_;
    ReadSramAllocator sram_;
    std::map<std::uint64_t, Region> regions_;
    std::unordered_map<std::uint64_t, std::uint64_t> background_;
    std::uint64_t next_token_{1}, next_background_id_{background_id_base};
    std::uint64_t residency_epoch_{};
    std::uint64_t reserved_bytes_{}, peak_reserved_bytes_{};
    std::uint64_t started_fills_{}, completed_fills_{}, active_fills_{}, peak_active_fills_{};
    std::uint64_t prefetch_bytes_{}, resident_read_bytes_{}, fallback_read_bytes_{};
    std::unordered_map<std::uint64_t, MemRequest> pending_;
};
} // namespace
TensorPlacement make_bank_plane_tensor_placement(const MemorySystemConfig& config) {
    std::optional<HBFMegapageAllocator> placement;
    const auto& controller = config.hbf.controller;
    if (config.hbf.align_tensors_to_command_group &&
        controller.geometry == HBFGeometryProfile::GroupedPlanes)
        placement.emplace(config.address_layout.hbf_base, controller);
    return [placement = std::move(placement)](TensorRegistry& registry,
                                              TensorMetadata tensor) mutable {
        if (!tensor.preloaded || !tensor.read_only)
            throw std::invalid_argument("HBF program tensors require preloaded read-only data");
        if (placement) {
            tensor.global_base_address = placement->allocate(tensor.byte_size()).base;
            registry.register_tensor(std::move(tensor));
        } else {
            allocate_contiguous_tensor(registry, std::move(tensor));
        }
    };
}

std::unique_ptr<MemoryBackend> make_bank_plane_hbf_backend(const HBFConfig& config,
                                                         double frequency, std::uint64_t base) {
    return std::make_unique<HBFineBackend>(config, frequency, base);
}
} // namespace GTSim
