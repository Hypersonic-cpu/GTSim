#pragma once
#include <cstdint>
#include <vector>
namespace GTSim {
struct BandwidthBin {
    std::uint64_t start_ps{}, end_ps{};
    std::uint64_t media_read_bytes{}, host_read_bytes{}, prefetch_fill_bytes{};
    std::uint64_t resident_read_bytes{}, sense_commands{};
};
struct HBFBackendStatistics {
    std::uint64_t accepted_requests{}, completed_requests{}, failed_requests{};
    std::uint64_t host_read_bytes{}, host_write_bytes{}, sensed_bytes{};
    // Counts actual Sense commands, independently of the number of sensed planes.
    std::uint64_t sense_commands{}, unique_sense_addresses{};
    std::uint64_t senses{}, programs{}, erases{}, cache_hits{}, cache_misses{}, cache_evictions{};
    std::uint64_t queue_stalls{}, bank_wait_ps{}, transfer_wait_ps{}, events_processed{};
    std::uint64_t elapsed_ps{};
    std::vector<BandwidthBin> bandwidth_bins;
    std::vector<std::uint64_t> stack_read_bytes, channel_read_bytes, bank_reads, plane_reads;
    std::vector<std::uint64_t> stack_busy_ps, channel_read_busy_ps, channel_write_busy_ps;
    std::vector<std::uint64_t> bank_busy_ps, plane_sense_busy_ps, plane_program_busy_ps, plane_erase_busy_ps;
    std::vector<std::uint64_t> die_transfer_busy_ps, ecc_busy_ps;
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

}
