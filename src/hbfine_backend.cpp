#include "memory_backend.h"
#include <hbfine/controller.hpp>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace GTSim {
namespace {
// The device uses physical time and knows nothing about GPU cycles or tensors.
// This adapter preserves the simulator's response metadata and clock boundary.
class HBFineBackend final : public MemoryBackend {
public:
    HBFineBackend(const HBFConfig& config, double frequency_mhz, std::uint64_t base)
        : frequency_khz_(clock_khz(frequency_mhz)), device_(config.controller, base, request_capacity(config)) {}

    bool try_submit(const MemRequest& request, int now) override {
        if (request.tier != MemoryTier::HBF) throw std::invalid_argument("HBFineBackend received a non-HBF request");
        if (pending_.contains(request.request_id)) throw std::logic_error("duplicate HBFine request ID");
        hbfine::Operation operation;
        switch (request.operation) {
        case MemoryOperation::Read: operation = hbfine::Operation::Read; break;
        case MemoryOperation::Write: operation = hbfine::Operation::Write; break;
        case MemoryOperation::Writeback: operation = hbfine::Operation::Writeback; break;
        default: throw std::invalid_argument("unsupported HBF operation");
        }
        if (!device_.try_submit({request.request_id, request.global_addr, request.size_bytes,
                                 operation, request.routing.axi_id}, physical_time(now))) return false;
        pending_.emplace(request.request_id, request);
        return true;
    }

    std::vector<MemResponse> step(int now) override {
        const auto completions = device_.advance(physical_time(now));
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
    HBFBackendStatistics hbf_statistics() const override { return device_.statistics(); }
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
    hbfine::Controller device_;
    std::unordered_map<std::uint64_t, MemRequest> pending_;
};
} // namespace
std::unique_ptr<MemoryBackend> make_bank_plane_hbf_backend(const HBFConfig& config,
                                                         double frequency, std::uint64_t base) {
    return std::make_unique<HBFineBackend>(config, frequency, base);
}
} // namespace GTSim
