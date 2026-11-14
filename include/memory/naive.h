#pragma once
#include "memory/cache.h"
namespace GTSim {
class HBMModel : public MemoryBackend {
public:
    void preload_region(std::uint64_t, std::uint64_t, bool) override {}
    HBMModel(const HBMConfig& config, double core_frequency_mhz)
        : total_bandwidth_bytes_per_cycle_(bytes_per_cycle_from_GBps(
              config.read_bw_GBps, core_frequency_mhz)),
          write_max_bandwidth_bytes_per_cycle_(bytes_per_cycle_from_GBps(
              config.write_max_bw_GBps, core_frequency_mhz)),
          read_latency_cycles_(config.read_latency_cycles),
          write_latency_cycles_(config.write_latency_cycles),
          queue_capacity_(static_cast<std::size_t>(config.queue_capacity)),
          statistics_interval_ps_(config.statistics_interval_ps),
          frequency_khz_(static_cast<std::uint64_t>(
              std::llround(core_frequency_mhz * 1000))) {}

    std::size_t pending_requests() const override { return queued_requests(); }

    // Compatibility constructor for code that used the previous DRAMModel API.
    HBMModel(int bandwidth_bytes_per_cycle, int latency_cycles,
             int /*line_size_bytes*/, double core_frequency_mhz,
             double dram_frequency_mhz)
        : total_bandwidth_bytes_per_cycle_(
              static_cast<double>(bandwidth_bytes_per_cycle) *
              dram_frequency_mhz / core_frequency_mhz),
          write_max_bandwidth_bytes_per_cycle_(
              static_cast<double>(bandwidth_bytes_per_cycle) *
              dram_frequency_mhz / core_frequency_mhz),
          read_latency_cycles_(latency_cycles),
          write_latency_cycles_(latency_cycles),
          queue_capacity_(65536) {}

    bool try_submit(const MemRequest& request, int now) override {
        if (queued_requests() >= queue_capacity_) {
            return false;
        }
        Request queued{request,
                       now + (request.operation == MemoryOperation::Read
                                  ? read_latency_cycles_
                                  : write_latency_cycles_),
                       static_cast<double>(std::max(1, request.size_bytes))};
        if (request.operation == MemoryOperation::Read) {
            read_requests_.push_back(queued);
        } else {
            write_requests_.push_back(queued);
        }
        return true;
    }

    void submit(const MemRequest& request, int now) {
        if (!try_submit(request, now)) {
            throw std::runtime_error("HBM request queue is full");
        }
    }

    void enqueue(const MemRequest& request, int now) { submit(request, now); }

    std::vector<MemResponse> step(int now) override {
        std::vector<MemResponse> completed;
        const bool read_ready = !read_requests_.empty() &&
                                read_requests_.front().ready_cycle <= now;
        const bool write_ready = !write_requests_.empty() &&
                                 write_requests_.front().ready_cycle <= now;
        if (!read_ready && !write_ready) {
            total_fractional_credit_ = 0.0;
            write_fractional_credit_ = 0.0;
            return completed;
        }

        double total_budget = total_bandwidth_bytes_per_cycle_ +
                              total_fractional_credit_;
        double write_budget = write_max_bandwidth_bytes_per_cycle_ +
                              write_fractional_credit_;
        while (total_budget > kBandwidthEpsilon) {
            const bool can_read = !read_requests_.empty() &&
                                  read_requests_.front().ready_cycle <= now;
            const bool can_write = !write_requests_.empty() &&
                                   write_requests_.front().ready_cycle <= now &&
                                   write_budget > kBandwidthEpsilon;
            if (!can_read && !can_write) {
                break;
            }

            const bool use_write = can_write && (!can_read || prefer_write_);
            Request& request = use_write ? write_requests_.front()
                                         : read_requests_.front();
            const double available = use_write
                                         ? std::min(total_budget, write_budget)
                                         : total_budget;
            const double transferred = std::min(
                static_cast<double>(request.remaining_bytes), available);
            if (transferred <= kBandwidthEpsilon) {
                break;
            }

            request.remaining_bytes -= transferred;
            total_budget -= transferred;
            if (use_write) {
                write_budget -= transferred;
            }
            prefer_write_ = !use_write;

            if (request.remaining_bytes <= kBandwidthEpsilon) {
                completed.push_back(make_response(request.memory_request));
                if (use_write) {
                    write_requests_.pop_front();
                } else {
                    if (statistics_interval_ps_) {
                        const auto time_ps = static_cast<std::uint64_t>(now) *
                                             1000000000ULL / frequency_khz_;
                        read_bandwidth_bins_[time_ps /
                                             statistics_interval_ps_] +=
                            completed.back().size_bytes;
                    }
                    read_requests_.pop_front();
                }
            }
        }

        total_fractional_credit_ =
            queued_requests() == 0 ? 0.0 : fractional_part(total_budget);
        write_fractional_credit_ =
            write_requests_.empty() ? 0.0 : fractional_part(write_budget);
        return completed;
    }

    std::vector<MemoryBandwidthSample> bandwidth_samples() const override {
        std::vector<MemoryBandwidthSample> samples;
        samples.reserve(read_bandwidth_bins_.size());
        for (const auto& [index, bytes] : read_bandwidth_bins_) {
            const auto start = index * statistics_interval_ps_;
            samples.push_back({start,
                               start + statistics_interval_ps_,
                               {{"demand_read", bytes}},
                               {}});
        }
        return samples;
    }

private:
    struct Request {
        MemRequest memory_request;
        int ready_cycle;
        double remaining_bytes;
    };

    static constexpr double kBandwidthEpsilon = 1.0e-9;

    double total_bandwidth_bytes_per_cycle_;
    double write_max_bandwidth_bytes_per_cycle_;
    int read_latency_cycles_;
    int write_latency_cycles_;
    std::size_t queue_capacity_;
    std::deque<Request> read_requests_;
    std::deque<Request> write_requests_;
    double total_fractional_credit_{};
    double write_fractional_credit_{};
    bool prefer_write_{};
    std::uint64_t statistics_interval_ps_{};
    std::uint64_t frequency_khz_{};
    std::map<std::uint64_t, std::uint64_t> read_bandwidth_bins_;

    std::size_t queued_requests() const {
        return read_requests_.size() + write_requests_.size();
    }

    static double fractional_part(double value) {
        return value - std::floor(value);
    }

    static MemResponse make_response(const MemRequest& request) {
        const int response_bytes = request.operation == MemoryOperation::Read
                                       ? std::max(1, request.size_bytes)
                                       : kMemoryControlBytes;
        return {request.request_id, request.global_addr, request.tier, true,
                response_bytes, request.operation, 0,
                request.operation == MemoryOperation::Read
                    ? request.shadow_digest
                    : 0};
    }
};

// Keep the old type spelling available for downstream code while all new
// topology construction uses HBMModel.
using DRAMModel = HBMModel;

// HBF controller with independent read and write service queues and bandwidths.
class HBFModel : public MemoryBackend {
public:
    void preload_region(std::uint64_t, std::uint64_t, bool) override {}
    HBFModel(const HBFConfig& config, double core_frequency_mhz)
        : read_bandwidth_bytes_per_cycle_(bytes_per_cycle_from_GBps(
              config.read_bw_GBps, core_frequency_mhz)),
          write_bandwidth_bytes_per_cycle_(bytes_per_cycle_from_GBps(
              config.write_bw_GBps, core_frequency_mhz)),
          read_latency_cycles_(config.read_latency_cycles),
          write_latency_cycles_(config.write_latency_cycles),
          queue_capacity_(static_cast<std::size_t>(config.queue_capacity)) {}

    bool try_submit(const MemRequest& request, int now) override {
        if (queued_requests() >= queue_capacity_) {
            return false;
        }
        Request queued{request,
                       now + (request.operation == MemoryOperation::Read
                                  ? read_latency_cycles_
                                  : write_latency_cycles_),
                       static_cast<double>(std::max(1, request.size_bytes))};
        if (request.operation == MemoryOperation::Read) {
            read_requests_.push_back(queued);
        } else {
            write_requests_.push_back(queued);
        }
        return true;
    }

    std::size_t pending_requests() const override { return queued_requests(); }

    void submit(const MemRequest& request, int now) {
        if (!try_submit(request, now)) {
            throw std::runtime_error("HBF request queue is full");
        }
    }

    std::vector<MemResponse> step(int now) override {
        std::vector<MemResponse> completed;
        service_queue(read_requests_, read_bandwidth_bytes_per_cycle_,
                      read_fractional_credit_, now, completed);
        service_queue(write_requests_, write_bandwidth_bytes_per_cycle_,
                      write_fractional_credit_, now, completed);
        return completed;
    }

private:
    struct Request {
        MemRequest memory_request;
        int ready_cycle;
        double remaining_bytes;
    };

    static constexpr double kBandwidthEpsilon = 1.0e-9;

    double read_bandwidth_bytes_per_cycle_;
    double write_bandwidth_bytes_per_cycle_;
    int read_latency_cycles_;
    int write_latency_cycles_;
    std::size_t queue_capacity_;
    std::deque<Request> read_requests_;
    std::deque<Request> write_requests_;
    double read_fractional_credit_{};
    double write_fractional_credit_{};

    std::size_t queued_requests() const {
        return read_requests_.size() + write_requests_.size();
    }

    static double fractional_part(double value) {
        return value - std::floor(value);
    }

    static MemResponse make_response(const MemRequest& request) {
        const int response_bytes = request.operation == MemoryOperation::Read
                                       ? std::max(1, request.size_bytes)
                                       : kMemoryControlBytes;
        return {request.request_id, request.global_addr, request.tier, true,
                response_bytes, request.operation};
    }

    static void service_queue(std::deque<Request>& requests,
                              double bandwidth_bytes_per_cycle,
                              double& fractional_credit, int now,
                              std::vector<MemResponse>& completed) {
        if (requests.empty() || requests.front().ready_cycle > now) {
            fractional_credit = 0.0;
            return;
        }

        double budget = bandwidth_bytes_per_cycle + fractional_credit;
        while (budget > kBandwidthEpsilon && !requests.empty() &&
               requests.front().ready_cycle <= now) {
            Request& request = requests.front();
            const double transferred = std::min(request.remaining_bytes, budget);
            if (transferred <= kBandwidthEpsilon) {
                break;
            }
            request.remaining_bytes -= transferred;
            budget -= transferred;
            if (request.remaining_bytes <= kBandwidthEpsilon) {
                completed.push_back(make_response(request.memory_request));
                requests.pop_front();
            }
        }
        fractional_credit = requests.empty() ? 0.0 : fractional_part(budget);
    }
};

// Full-duplex point-to-point link. Each direction has its own serializer and
// propagation latency; queueing is FIFO and bounded by queue_capacity.

}
