#include "memory_backend.h"
#include "hbm_address.h"
#include "memory_clock.h"

#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/dram/dram_spec.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace GTSim {
namespace {

class RamulatorHBMBackend final : public MemoryBackend {
public:
    RamulatorHBMBackend(const HBMConfig& config, double core_frequency_mhz)
        : capacity_(static_cast<std::size_t>(config.queue_capacity)),
          config_(Ramulator::Config::parse_config_file(config.ramulator.config_file)),
          frontend_(Ramulator::Factory::create_frontend(config_)),
          memory_(Ramulator::Factory::create_memory_system(config_)),
          clock_(core_frequency_mhz, static_cast<int>(std::lround(memory_->get_tCK() * 1000))),
          pending_(config.ramulator.stacks * 16), stacks_(config.ramulator.stacks) {
        const auto controllers = config_["memory_system"]["controllers"];
        if (controllers.size() != pending_.size() || memory_->get_tx_bytes() != 32 ||
            config_["memory_system"]["channel_mapper"]["impl"].as<std::string>() !=
                "PassThroughChannelMapper") {
            throw std::invalid_argument("Ramulator H100 profile requires 80 channels, "
                                        "32B bursts and pass-through channel mapping");
        }
        // Validate every channel; native address vectors below target this exact
        // HBM3 hierarchy. A different profile must never silently alias addresses.
        for (const auto& controller : controllers.seq()) {
            if (controller["impl"].as<std::string>() != "HBM34" ||
                controller["addr_mapper"]["impl"].as<std::string>() != "PassThroughAddrMapper" ||
                controller["dram"]["impl"].as<std::string>() != "HBM3") {
                throw std::invalid_argument("unsupported Ramulator H100 controller profile");
            }
            const Ramulator::ConfigNode wrapper(Ramulator::ConfigNode::Map{{"dram", controller["dram"]}});
            const auto spec = Ramulator::DRAMSpec::create("HBM3", wrapper);
            const std::vector<int> expected{1, 2, 2, 4, 4, 16384, 256};
            if (spec->organization.level_sizes != expected || spec->channel_width != 32 ||
                spec->get_timing_value("tCK_ps") !=
                    static_cast<int>(std::lround(memory_->get_tCK() * 1000))) {
                throw std::invalid_argument("inconsistent HBM3 organization or channel clocks");
            }
            const int write_ticks = spec->get_timing_value("nCWL") + spec->get_timing_value("nBL");
            if (write_data_ticks_ && write_data_ticks_ != write_ticks) {
                throw std::invalid_argument("inconsistent HBM3 write latency across channels");
            }
            write_data_ticks_ = write_ticks;
        }
        frontend_->connect_memory_system(memory_.get());
        memory_->connect_frontend(frontend_.get());
        stats_.channel_transactions.resize(pending_.size());
    }

    bool try_submit(const MemRequest& request, int now) override {
        if (now < last_core_cycle_) throw std::invalid_argument("memory time moved backwards");
        if (request.operation != MemoryOperation::Read && request.operation != MemoryOperation::Write &&
            request.operation != MemoryOperation::Writeback) {
            throw std::invalid_argument("unsupported DRAM memory operation");
        }
        if (request.size_bytes <= 0 || request.size_bytes % 32 || request.global_addr % 32 ||
            request.global_addr >= (static_cast<std::uint64_t>(stacks_) * (16ULL << 30)) ||
            static_cast<std::uint64_t>(request.size_bytes) > (static_cast<std::uint64_t>(stacks_) * (16ULL << 30)) - request.global_addr) {
            throw std::invalid_argument("HBM3 requires aligned, in-range 32B-multiple requests");
        }
        // MemorySystem chooses the physical device. Preserve the original tier
        // in responses so FullHBM placement overrides still match L2 MSHRs.
        if (parents_.contains(request.request_id)) throw std::logic_error("duplicate HBM request ID");
        if (parents_.size() >= capacity_) {
            ++stats_.admission_retries;
            return false;
        }
        parents_.emplace(request.request_id, Parent{request, request.size_bytes / 32});
        for (int offset = 0; offset < request.size_bytes; offset += 32) {
            const auto address = request.global_addr + static_cast<std::uint64_t>(offset);
            const auto decoded = decode_h100_hbm_address(address, stacks_);
            pending_[decoded.channel].push_back(
                Burst{next_token_++, request.request_id, address, clock_.first_edge_at_or_after(now),
                      request.operation != MemoryOperation::Read});
        }
        ++stats_.accepted_requests;
        return true;
    }

    std::vector<MemResponse> step(int now) override {
        if (now < last_core_cycle_) throw std::invalid_argument("memory time moved backwards");
        const auto target = clock_.ticks_at(now);
        while (stats_.dram_ticks < target) {
            // Dispatch only requests that have arrived at this DRAM edge. A
            // request admitted at 'now' must not enter an earlier half-CK tick.
            dispatch();
            ++stats_.dram_ticks;
            memory_->tick();
            while (!write_completions_.empty() &&
                   write_completions_.top().tick <= stats_.dram_ticks) {
                const auto token = write_completions_.top().token;
                write_completions_.pop();
                finish_burst(token);
            }
        }
        last_core_cycle_ = now;
        std::vector<MemResponse> result;
        result.swap(completed_);
        return result;
    }

    std::size_t pending_requests() const override { return parents_.size() + completed_.size(); }
    MemoryBackendStatistics backend_statistics() const override {
        auto result = stats_;
        const auto snapshot = memory_->collect_stats();
        const auto controllers = snapshot["controller"];
        for (const auto& controller : controllers.seq()) {
            result.controller_reads += controller["num_read_reqs"].as<std::uint64_t>();
            result.controller_writes += controller["num_write_reqs"].as<std::uint64_t>();
            result.controller_served_reads += controller["num_read_reqs_served"].as<std::uint64_t>();
            result.controller_served_writes += controller["num_write_reqs_served"].as<std::uint64_t>();
            result.row_hits += controller["row_hits"].as<std::uint64_t>();
            result.row_misses += controller["row_misses"].as<std::uint64_t>();
            result.row_conflicts += controller["row_conflicts"].as<std::uint64_t>();
        }
        return result;
    }

private:
    struct Parent { MemRequest request; int remaining; };
    struct Burst {
        std::uint64_t token;
        std::uint64_t parent_id;
        std::uint64_t address;
        std::uint64_t ready_tick;
        bool write;
        bool callback_received{};
    };
    struct WriteCompletion {
        std::uint64_t tick;
        std::uint64_t token;
        bool operator>(const WriteCompletion& other) const {
            return tick != other.tick ? tick > other.tick : token > other.token;
        }
    };

    std::size_t capacity_;
    Ramulator::ConfigNode config_;
    std::unique_ptr<Ramulator::IFrontEnd> frontend_;
    std::unique_ptr<Ramulator::IMemorySystem> memory_;
    MemoryClock clock_;
    int write_data_ticks_{};
    int last_core_cycle_{};
    std::uint64_t next_token_{};
    MemoryBackendStatistics stats_;
    std::vector<std::deque<Burst>> pending_;
    int stacks_;
    std::unordered_map<std::uint64_t, Parent> parents_;
    std::unordered_map<std::uint64_t, Burst> inflight_;
    std::unordered_set<std::uint64_t> writing_addresses_;
    std::priority_queue<WriteCompletion, std::vector<WriteCompletion>,
                        std::greater<WriteCompletion>> write_completions_;
    std::vector<MemResponse> completed_;

    void dispatch() {
        for (auto& queue : pending_) {
            while (!queue.empty() && queue.front().ready_tick <= stats_.dram_ticks + 1) {
                const Burst burst = queue.front();
                // The upstream API acknowledges coalesced writes immediately.
                // Serialize same-address requests until write data has arrived
                // so acknowledgements and forwarded reads cannot retire early.
                if (writing_addresses_.contains(burst.address)) break;
                const auto address = decode_h100_hbm_address(burst.address, stacks_);
                Ramulator::Request request(
                    static_cast<Ramulator::Addr_t>(burst.address),
                    burst.write ? Ramulator::Request::Type::Write : Ramulator::Request::Type::Read,
                    0, [this, token = burst.token](Ramulator::Request&) { on_callback(token); });
                request.size_bytes = 32;
                request.addr_vec = {address.channel, address.pseudochannel, address.sid,
                                    address.bank_group, address.bank, address.row, address.column};
                inflight_.emplace(burst.token, burst);
                if (!memory_->send(request)) {
                    inflight_.erase(burst.token);
                    ++stats_.admission_retries;
                    break;
                }
                if (burst.write) writing_addresses_.insert(burst.address);
                queue.pop_front();
                ++stats_.submitted_transactions;
                stats_.transaction_bytes += 32;
                ++stats_.channel_transactions[address.channel];
            }
        }
    }

    void on_callback(std::uint64_t token) {
        auto& burst = inflight_.at(token);
        if (burst.callback_received) throw std::logic_error("duplicate Ramulator callback");
        burst.callback_received = true;
        if (burst.write) {
            // This revision calls back at WR issue. Delay acknowledgement until
            // nCWL+nBL has elapsed, using the profile's half-CK timing values.
            write_completions_.push({stats_.dram_ticks + write_data_ticks_, token});
        } else {
            finish_burst(token);
        }
    }

    void finish_burst(std::uint64_t token) {
        const auto burst = inflight_.at(token);
        auto& parent = parents_.at(burst.parent_id);
        if (--parent.remaining == 0) {
            const auto& request = parent.request;
            completed_.push_back({request.request_id, request.global_addr, request.tier, true,
                                  request.operation == MemoryOperation::Read ? request.size_bytes : 16,
                                  request.operation});
            parents_.erase(burst.parent_id);
            ++stats_.completed_requests;
        }
        if (burst.write) writing_addresses_.erase(burst.address);
        inflight_.erase(token);
        ++stats_.completed_transactions;
    }
};
} // namespace

std::unique_ptr<MemoryBackend> make_ramulator_hbm_backend(
    const HBMConfig& config, double core_frequency_mhz) {
    return std::make_unique<RamulatorHBMBackend>(config, core_frequency_mhz);
}

} // namespace GTSim
