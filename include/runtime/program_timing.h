#pragma once

#include "memory.h"

namespace GTSim::ExecutionIR {

struct TierTiming {
    std::uint64_t read_bytes{}, write_bytes{};
    std::map<std::string, std::uint64_t> counters;
    std::vector<std::size_t> active_units;
    std::map<std::size_t, std::uint64_t> unit_busy_ticks;
    std::uint64_t read_completions{}, read_latency_cycles{},
        unadmitted_alias_read_bytes{};
    std::map<std::string, std::uint64_t> gauges;
};

struct PrefetchWindowTiming {
    int slot{};
    std::uint64_t epoch{}, bytes{};
    int enqueue_cycle{-1}, consume_cycle{-1}, release_request_cycle{-1};
    bool skipped{};
};

struct KernelTiming {
    std::string id, state;
    int launch_cycle{}, start_cycle{}, device_complete_cycle{},
        complete_cycle{}, device_cycles{};
    std::size_t l2_lines_before{}, l2_lines_after{};
    std::vector<PrefetchWindowTiming> prefetch_windows;
    std::map<std::string, TierTiming> memory;
    std::size_t background_at_launch{}, background_at_completion{};
    std::uint64_t background_compute_cycles{};
};

struct ProgramTiming {
    int total_cycles{};
    std::vector<KernelTiming> per_kernel;
    std::map<std::string, int> events;
    std::map<std::string, std::vector<std::string>> state_history;
    std::map<std::string, std::uint64_t> traffic_bytes;
    std::map<std::string, std::vector<MemoryBandwidthSample>> memory_bandwidth;
    std::map<std::string, std::map<std::string, std::uint64_t>>
        tensor_addresses;
    std::uint64_t prefetch_capacity_stalls{}, prefetch_canceled{},
        background_cycles{};
    std::map<std::string, int> prefetch_release_cycles;
    bool persistent_gpu{}, persistent_memory{}, persistent_l2{},
        stable_allocations{};
};

} // namespace GTSim::ExecutionIR
