#pragma once

#include "memory.h"

namespace GTSim::ExecutionIR {

struct TierTiming {
    std::uint64_t read_bytes{}, write_bytes{};
    std::map<std::string, std::uint64_t> counters;
    std::vector<std::size_t> active_units;
    std::map<std::size_t, std::uint64_t> unit_busy_ticks;
    std::uint64_t read_completions{}, read_latency_cycles{},
        unadmitted_alias_read_bytes{}, unadmitted_alias_read_requests{};
    std::map<std::string, std::uint64_t> gauges;
};

struct PrefetchWindowTiming {
    int slot{};
    std::uint64_t epoch{}, bytes{};
    int enqueue_cycle{-1}, consume_cycle{-1}, release_request_cycle{-1};
    bool skipped{};
};

struct PrefetchRegionTiming {
    int enqueue_cycle{-1}, ready_cycle{-1}, first_use_cycle{-1},
        last_consumer_complete_cycle{-1};
    std::uint64_t bytes{};
    bool ready_at_first_use{};
};

struct KernelTiming {
    std::string id, state;
    int launch_cycle{}, start_cycle{}, device_complete_cycle{},
        complete_cycle{}, device_cycles{};
    std::size_t l2_lines_before{}, l2_lines_after{};
    std::uint64_t l2_read_hit_bytes{}, l2_read_merged_bytes{},
        l2_read_miss_bytes{}, tma_load_bytes{}, tma_store_bytes{};
    std::size_t cta_count{}, active_sms{};
    std::uint64_t tma_load_operations{}, tma_store_operations{}, wgmma_warp_operations{};
    std::uint64_t active_sm_cycles{};
    std::map<std::string, double> pipeline_inflight_fraction;
    std::map<std::string, std::uint64_t> dependency_path_cycles;
    std::uint64_t dependency_path_wait_cycles{};
    int critical_path_cta{-1}, longest_path_iteration{-1};
    std::uint64_t longest_path_iteration_cycles{};
    std::vector<PrefetchWindowTiming> prefetch_windows;
    std::vector<PrefetchRegionTiming> prefetch_regions;
    std::map<std::string, TierTiming> memory;
    std::size_t background_at_launch{}, background_at_completion{};
    std::uint64_t background_compute_cycles{};
};

struct PrefetchStats {
    std::uint64_t queued{}, admitted{}, completed{}, resident_read_bytes{},
        fallback_read_bytes{}, canceled{};
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
    PrefetchStats prefetch_stats;
};

} // namespace GTSim::ExecutionIR
