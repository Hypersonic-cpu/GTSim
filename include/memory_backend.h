#ifndef GTSIM_MEMORY_BACKEND_H
#define GTSIM_MEMORY_BACKEND_H

#include "dag_node.h"
#include "mem_conf.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace GTSim {

enum class MemoryOperation : std::uint8_t { Read, Write, Writeback };

// Hints describe upstream routing without imposing an L2 slice model on DRAM.
// Negative values mean that the producer has not assigned that hierarchy level.
struct MemoryRoutingHint {
    int l2_slice{-1};
    int logical_mc_group{-1};
    int physical_channel{-1};
};

struct MemRequest {
    std::uint64_t request_id;
    std::uint64_t global_addr;
    int size_bytes;
    MemoryOperation operation;
    MemoryTier tier;
    MemoryRoutingHint routing{};
};

struct MemResponse {
    std::uint64_t request_id;
    std::uint64_t global_addr;
    MemoryTier tier;
    bool completed;
    int size_bytes{};
    MemoryOperation operation{MemoryOperation::Read};
};

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

// A rejected request remains owned by the caller. An accepted request produces
// exactly one response, even when the backend splits it into multiple bursts.
// All interface times are GPU core cycles; each backend owns clock conversion.
class MemoryBackend {
public:
    virtual ~MemoryBackend() = default;
    virtual bool try_submit(const MemRequest& request, int now) = 0;
    virtual std::vector<MemResponse> step(int now) = 0;
    virtual std::size_t pending_requests() const = 0;
    virtual MemoryBackendStatistics backend_statistics() const { return {}; }
};

std::unique_ptr<MemoryBackend> make_hbm_backend(const HBMConfig& config,
                                              double core_frequency_mhz);

} // namespace GTSim

#endif
