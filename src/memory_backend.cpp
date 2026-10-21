#include "memory.h"
#include "hbm_address.h"

namespace GTSim {

std::unique_ptr<MemoryBackend> make_bank_plane_hbf_backend(
    const HBFConfig&, double, std::uint64_t);

namespace {
// Per-channel copies of the existing HBM service law. Only active channels are
// advanced; historical aggregate-naive profiles retain their original model.
class ChannelizedHBM final : public MemoryBackend {
public:
    ChannelizedHBM(const HBMConfig& config, double frequency) : stacks_(config.stacks) {
        const int count = config.stacks * config.channels_per_stack;
        auto channel = config;
        channel.read_bw_GBps /= count;
        channel.write_max_bw_GBps /= count;
        channel.queue_capacity = std::max(1, config.queue_capacity / count);
        active_flags_.resize(count);
        stats_.channel_transactions.resize(count);
        for (int i = 0; i < count; ++i) channels_.push_back(std::make_unique<HBMModel>(channel, frequency));
    }
    bool try_submit(const MemRequest& request, int now) override {
        if (request.size_bytes <= 0 || request.size_bytes > 128 ||
            request.global_addr % 128 + request.size_bytes > 128)
            throw std::invalid_argument("channelized naive HBM transaction crosses a 128B stripe");
        const auto address = decode_h100_hbm_address(request.global_addr, stacks_);
        if (!channels_[address.channel]->try_submit(request, now)) return false;
        if (!active_flags_[address.channel]) {
            active_flags_[address.channel] = true;
            active_.push_back(address.channel);
        }
        ++pending_;
        ++stats_.accepted_requests;
        ++stats_.channel_transactions[address.channel];
        return true;
    }
    std::vector<MemResponse> step(int now) override {
        std::vector<MemResponse> result;
        std::size_t retained = 0;
        for (int id : active_) {
            auto completed = channels_[id]->step(now);
            pending_ -= completed.size();
            stats_.completed_requests += completed.size();
            result.insert(result.end(), completed.begin(), completed.end());
            if (channels_[id]->pending_requests()) active_[retained++] = id;
            else active_flags_[id] = false;
        }
        active_.resize(retained);
        return result;
    }
    std::size_t pending_requests() const override { return pending_; }
    MemoryBackendStatistics backend_statistics() const override { return stats_; }
private:
    int stacks_;
    std::size_t pending_{};
    std::vector<std::unique_ptr<HBMModel>> channels_;
    std::vector<int> active_;
    std::vector<bool> active_flags_;
    MemoryBackendStatistics stats_;
};
}

#ifdef GTSIM_ENABLE_RAMULATOR2
std::unique_ptr<MemoryBackend> make_ramulator_hbm_backend(
    const HBMConfig& config, double core_frequency_mhz);
#endif

std::unique_ptr<MemoryBackend> make_hbm_backend(const HBMConfig& config,
                                              double core_frequency_mhz) {
    switch (config.backend) {
    case HBMBackend::Naive:
        if (config.model_channels) return std::make_unique<ChannelizedHBM>(config, core_frequency_mhz);
        return std::make_unique<HBMModel>(config, core_frequency_mhz);
    case HBMBackend::Ramulator2:
#ifdef GTSIM_ENABLE_RAMULATOR2
        return make_ramulator_hbm_backend(config, core_frequency_mhz);
#else
        throw std::runtime_error(
            "Ramulator2 backend requested; rebuild with -DGTSIM_ENABLE_RAMULATOR2=ON");
#endif
    }
    throw std::invalid_argument("unknown HBM backend");
}

std::unique_ptr<MemoryBackend> make_hbf_backend(const HBFConfig& config,
                                              double frequency, std::uint64_t base) {
    if (config.backend == HBFBackend::Naive) return std::make_unique<HBFModel>(config, frequency);
    if (config.backend == HBFBackend::BankPlane) return make_bank_plane_hbf_backend(config, frequency, base);
    throw std::invalid_argument("unknown HBF backend");
}

} // namespace GTSim
