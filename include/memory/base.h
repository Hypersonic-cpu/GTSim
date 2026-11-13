#ifndef GTSIM_MEMORY_BACKEND_H
#define GTSIM_MEMORY_BACKEND_H

#include "dag_node.h"
#include "memory/stats.h"
#include "mem_conf.h"

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
    int consumer_node{-1};
};

enum class MemoryReadPath : std::uint8_t { Host, Prefetch, Resident };
struct MemoryLocation { int stack{}, die{}, plane{}; };

struct MemRequest {
    std::uint64_t request_id;
    std::uint64_t global_addr;
    int size_bytes;
    MemoryOperation operation;
    MemoryTier tier;
    MemoryRoutingHint routing{};
    std::uint64_t read_alias{};
    MemoryReadPath read_path{MemoryReadPath::Host};
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

struct PrefetchBuffer {
    int slot{}, count{};
    std::uint64_t epoch{};
};

struct MemoryBandwidthSample {
    std::uint64_t start_ps{}, end_ps{};
    std::map<std::string, std::uint64_t> bytes;
    std::map<std::string, std::uint64_t> counts;
};

struct DeviceCounters {
    std::map<std::string, std::uint64_t> cumulative;
    std::vector<std::uint64_t> unit_busy_ticks;
    std::map<std::string, std::uint64_t> gauges;
};

struct TierCounterSnapshot {
    std::uint64_t read_bytes{}, write_bytes{};
    DeviceCounters device;
    std::uint64_t read_completions{}, read_latency_cycles{}, unadmitted_alias_read_bytes{};
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
    virtual std::uint64_t
    reserve_prefetch(std::uint64_t, std::uint64_t,
                     std::optional<PrefetchBuffer> = std::nullopt) {
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
    virtual std::vector<MemoryBandwidthSample> bandwidth_samples() const {
        return {};
    }
    virtual bool supports_sram_paths() const { return false; }
    virtual MemoryLocation locate(std::uint64_t) const {
        throw std::invalid_argument("backend does not expose physical SRAM placement");
    }
    virtual bool supports_prefetch() const { return false; }
    virtual void preload_region(std::uint64_t, std::uint64_t, bool) {
        throw std::invalid_argument("backend does not support preload");
    }
};

std::unique_ptr<MemoryBackend> make_sram_backend(std::unique_ptr<MemoryBackend>, const HBFConfig&, std::uint64_t);

std::unique_ptr<MemoryBackend> make_hbm_backend(const HBMConfig& config,
                                              double core_frequency_mhz);
std::unique_ptr<MemoryBackend> make_hbf_backend(const HBFConfig& config,
                                              double core_frequency_mhz, std::uint64_t base_address);

} // namespace GTSim

#endif
