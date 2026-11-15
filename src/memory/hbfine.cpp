#include "memory_backend.h"
#include <hbfine/address.hpp>
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
          device_(config_, base, request_capacity(config)) {}

    bool try_submit(const MemRequest& request, int now) override {
        if (request.tier != MemoryTier::HBF) throw std::invalid_argument("HBFineBackend received a non-HBF request");
        if (pending_.contains(request.request_id))
            throw std::logic_error("duplicate HBFine request ID");
        hbfine::Operation operation;
        switch (request.operation) {
        case MemoryOperation::Read: operation = hbfine::Operation::Read; break;
        case MemoryOperation::Write: operation = hbfine::Operation::Write; break;
        case MemoryOperation::Writeback: operation = hbfine::Operation::Writeback; break;
        default: throw std::invalid_argument("unsupported HBF operation");
        }
        const auto path=request.read_path==MemoryReadPath::Prefetch ? hbfine::ReadPath::Prefetch :
            request.read_path==MemoryReadPath::Resident ? hbfine::ReadPath::Resident : hbfine::ReadPath::Host;
        if (!device_.try_submit({request.request_id, request.global_addr, request.size_bytes,
                                 operation, request.routing.axi_id, path}, physical_time(now))) return false;
        pending_.emplace(request.request_id, request);
        return true;
    }

    std::vector<MemResponse> step(int now) override {
        const auto time = physical_time(now);
        const auto completions = device_.advance(time);
        std::vector<MemResponse> responses;
        responses.reserve(completions.size());
        for (const auto& completion : completions) {
            const auto& request = pending_.at(completion.request_id);
            responses.push_back({request.request_id, request.global_addr, request.tier,
                                 completion.error == hbfine::Error::None,
                                 request.operation == MemoryOperation::Read ? request.size_bytes : 16,
                                 request.operation, static_cast<int>(completion.error)});
            pending_.erase(completion.request_id);

        }
        return responses;
    }
    std::size_t pending_requests() const override { return pending_.size(); }
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
        return counters;
    }
    bool supports_sram_paths() const override { return true; }
    MemoryLocation locate(std::uint64_t address) const override {
        const auto a=hbfine::decode_address(address,base_,config_);
        return {a.stack,a.die,a.plane};
    }
    HBFBackendStatistics hbf_statistics() const override { const auto& s = device_.statistics();
        HBFBackendStatistics r;
        r.accepted_requests = s.accepted_requests;
        r.completed_requests = s.completed_requests;
        r.failed_requests = s.failed_requests;
        r.host_read_bytes = s.host_read_bytes;
        r.host_write_bytes = s.host_write_bytes;
        r.sensed_bytes = s.sensed_bytes;
        r.sense_commands = s.sense_commands;
        r.unique_sense_addresses = s.unique_sense_addresses;
        r.senses = s.senses;
        r.programs = s.programs;
        r.erases = s.erases;
        r.cache_hits = s.cache_hits;
        r.cache_misses = s.cache_misses;
        r.cache_evictions = s.cache_evictions;
        r.queue_stalls = s.queue_stalls;
        r.bank_wait_ps = s.bank_wait_ps;
        r.transfer_wait_ps = s.transfer_wait_ps;
        r.events_processed = s.events_processed;
        r.elapsed_ps = s.elapsed_ps;
        r.stack_read_bytes = s.stack_read_bytes;
        r.channel_read_bytes = s.channel_read_bytes;
        r.bank_reads = s.bank_reads;
        r.plane_reads = s.plane_reads;
        r.stack_busy_ps = s.stack_busy_ps;
        r.channel_read_busy_ps = s.channel_read_busy_ps;
        r.channel_write_busy_ps = s.channel_write_busy_ps;
        r.bank_busy_ps = s.bank_busy_ps;
        r.plane_sense_busy_ps = s.plane_sense_busy_ps;
        r.plane_program_busy_ps = s.plane_program_busy_ps;
        r.plane_erase_busy_ps = s.plane_erase_busy_ps;
        r.die_transfer_busy_ps = s.die_transfer_busy_ps;
        r.ecc_busy_ps = s.ecc_busy_ps;
        for (const auto& b : s.bandwidth_bins) r.bandwidth_bins.push_back({b.start_ps,b.end_ps,b.media_read_bytes,b.host_read_bytes,b.prefetch_fill_bytes,b.resident_read_bytes,b.sense_commands});
        return r; }
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
    std::unordered_map<std::uint64_t, MemRequest> pending_;
};
} // namespace
std::unique_ptr<MemoryBackend> make_bank_plane_hbf_backend(const HBFConfig& config,
                                                         double frequency, std::uint64_t base) {
    return std::make_unique<HBFineBackend>(config, frequency, base);
}
} // namespace GTSim
