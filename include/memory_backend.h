#ifndef GTSIM_MEMORY_BACKEND_H
#define GTSIM_MEMORY_BACKEND_H

#include "dag_node.h"
#include "mem_conf.h"
#include <hbfine/types.hpp>
#include <cstdint>
#include <memory>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace GTSim {

enum class MemoryOperation : std::uint8_t { Read, Write, Writeback };

// Hints describe upstream routing without imposing an L2 slice model on DRAM.
// Negative values mean that the producer has not assigned that hierarchy level.
struct MemoryRoutingHint {
    int l2_slice{-1};
    int logical_mc_group{-1};
    int physical_channel{-1};
    std::int64_t axi_id{-1};
};

struct MemRequest {
    std::uint64_t request_id;
    std::uint64_t global_addr;
    int size_bytes;
    MemoryOperation operation;
    MemoryTier tier;
    MemoryRoutingHint routing{};
    std::uint64_t read_alias{};
};

struct MemResponse {
    std::uint64_t request_id;
    std::uint64_t global_addr;
    MemoryTier tier;
    bool completed;
    int size_bytes{};
    MemoryOperation operation{MemoryOperation::Read};
    int error_code{}; // zero on success; backend-specific diagnostic on failure
};

using HBFBackendStatistics = hbfine::Statistics;

struct MemoryBackendStatistics {
    std::uint64_t accepted_requests{};
    std::uint64_t completed_requests{};
    std::uint64_t submitted_transactions{};
    std::uint64_t completed_transactions{};
    std::uint64_t transaction_bytes{};
    std::uint64_t admission_retries{};
    std::uint64_t dram_ticks{};
    std::uint64_t controller_reads{};
    std::uint64_t controller_writes{};
    std::uint64_t controller_served_reads{};
    std::uint64_t controller_served_writes{};
    std::uint64_t row_hits{};
    std::uint64_t row_misses{};
    std::uint64_t row_conflicts{};
    std::vector<std::uint64_t> channel_transactions;
};

struct DeviceCounters {
    std::map<std::string, std::uint64_t> cumulative;
    std::vector<std::uint64_t> unit_busy_ticks;
};

struct TierCounterSnapshot {
    std::uint64_t read_bytes{}, write_bytes{};
    DeviceCounters device;
};

// Placement runs before construction of the persistent memory system.
using TensorPlacement = std::function<void(TensorRegistry&, TensorMetadata)>;
void allocate_contiguous_tensor(TensorRegistry& registry, TensorMetadata tensor);
TensorPlacement make_tensor_placement(const MemorySystemConfig& config, MemoryTier tier);

// A rejected request remains owned by the caller. An accepted request produces
// exactly one response, even when the backend splits it into multiple bursts.
// All interface times are GPU core cycles; each backend owns clock conversion.
class MemoryBackend {
public:
    virtual ~MemoryBackend() = default;
    virtual bool try_submit(const MemRequest& request, int now) = 0;
    virtual std::vector<MemResponse> step(int now) = 0;
    virtual std::size_t pending_requests() const = 0;
    virtual std::uint64_t reserve_prefetch(std::uint64_t, std::uint64_t) {
        throw std::invalid_argument("device does not support prefetch");
    }
    virtual void start_prefetch(std::uint64_t) {
        throw std::invalid_argument("device does not support prefetch");
    }
    virtual bool prefetch_ready(std::uint64_t) const { return false; }
    virtual void release_prefetch(std::uint64_t) {
        throw std::invalid_argument("device does not support prefetch");
    }
    virtual std::uint64_t residency_epoch() const { return 0; }
    virtual std::size_t pending_background_requests() const { return 0; }
    virtual MemoryBackendStatistics backend_statistics() const { return {}; }
    virtual DeviceCounters device_counters() const { return {}; }
    virtual HBFBackendStatistics hbf_statistics() const { return {}; }
    virtual void preload_region(std::uint64_t, std::uint64_t, bool) {}
};

std::unique_ptr<MemoryBackend> make_hbm_backend(const HBMConfig& config,
                                              double core_frequency_mhz);
std::unique_ptr<MemoryBackend> make_hbf_backend(const HBFConfig& config,
                                              double core_frequency_mhz, std::uint64_t base_address);

} // namespace GTSim

#endif
