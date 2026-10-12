#ifndef GPU_H
#define GPU_H

#include "sm.h"
#include "memory.h"
#include "noc.h"
#include <vector>
#include <deque>
#include <memory>
#include <unordered_map>

namespace GTSim {

// GPU containing multiple SMs and L2 cache (DRAM)
// Tracks a remote SRAM access after NoC forward completes,
// waiting for the target SM's SRAM to finish.
struct NoCPendingEntry {
    int node_id;
    int src_sm_id;   // Originating SM (where the subpartition lives)
    int dst_sm_id;   // Target SM (whose SRAM was accessed)
    int bytes;
    int bank_conflict;
};

// Retry forward completion when destination SRAM queue is full.
struct NoCForwardRetryEntry {
    NoCRequest req;
};

// Retry return-hop enqueue when NoC queue is full.
struct NoCReturnRetryEntry {
    NoCPendingEntry entry;
};

class GPU {
public:
    std::vector<SM*> sms;
    L2Cache* l2;  // L2 cache
    std::unique_ptr<MemorySystem> memory_system;
    NoC* noc;  // NoC interconnect (nullptr if disabled)
    std::vector<Subpartition*>* node_id_to_sp;
    std::vector<int>* node_id_to_sm;

    // Pending remote reads: after SRAM completes, need NoC return hop
    // Uses multimap since multiple sub-ops of the same node can be in-flight
    std::unordered_multimap<int, NoCPendingEntry> pending_remote_reads;
    // Pending remote writes: after SRAM completes, route directly to origin
    std::unordered_multimap<int, NoCPendingEntry> pending_remote_writes;
    // NoC forward completions waiting to enter destination SRAM.
    std::deque<NoCForwardRetryEntry> forward_to_sram_retry_queue;
    // Remote reads whose return hop could not be enqueued this cycle.
    std::deque<NoCReturnRetryEntry> return_retry_queue;

    int num_sms;

    GPU(int num_streaming_multiprocessors, int warp_count, int tensor_throughput,
        int simd_throughput, int sfu_throughput, int ls_throughput_bytes,
        int tensor_latency_cycles, int simd_latency_cycles,
        int sfu_latency_cycles, int shfl_latency_cycles, int ls_latency_cycles,
        int tensor_width, int simd_width, int sfu_width, int ls_width_bytes,
        int sram_latency_cycles, int sram_bandwidth_bytes_per_cycle, int sram_queue_depth,
        bool o2, SchedulePolicy policy,
        int max_blocks_per_sm, int l2_cache_size_bytes, int l2_line_size_bytes,
        int l2_hit_latency, int l2_bandwidth_bytes_per_cycle,
        int l2_write_bandwidth_bytes_per_cycle, int l2_queue_depth,
        int l2_max_transactions_per_cycle_per_sp,
        bool l2_bypass_cache,
        int dram_bandwidth_bytes_per_cycle, int dram_latency_cycles,
        double core_freq_mhz, double dram_freq_mhz,
        int tma_setup_latency_cycles,
        int tma_issue_interval_cycles,
        int tma_issue_rate_bytes_per_cycle,
        int tmem_cols_per_cta,
        int tmem_issue_interval_cycles,
        int tmem_read_latency_cycles,
        int tmem_read_bandwidth_bytes_per_cycle,
        int tmem_write_latency_cycles,
        int tmem_write_bandwidth_bytes_per_cycle,
        int tmem_queue_depth,
        int startup_delay_cycles,
        int block_schedule_latency_cycles,
        const TensorRegistry& tensors)
        : l2(nullptr), noc(nullptr), num_sms(num_streaming_multiprocessors) {

        // Create SMs
        for (int i = 0; i < num_sms; ++i) {
            sms.push_back(new SM(i, warp_count, tensor_throughput, simd_throughput,
                                 sfu_throughput, ls_throughput_bytes,
                                 tensor_latency_cycles, simd_latency_cycles,
                                 sfu_latency_cycles, shfl_latency_cycles, ls_latency_cycles,
                                 tensor_width, simd_width, sfu_width, ls_width_bytes,
                                 sram_latency_cycles, sram_bandwidth_bytes_per_cycle,
                                 sram_queue_depth, o2, policy, max_blocks_per_sm,
                                 l2_max_transactions_per_cycle_per_sp,
                                 tma_setup_latency_cycles,
                                 tma_issue_interval_cycles,
                                 tma_issue_rate_bytes_per_cycle,
                                 tmem_cols_per_cta,
                                 tmem_issue_interval_cycles,
                                 tmem_read_latency_cycles,
                                 tmem_read_bandwidth_bytes_per_cycle,
                                 tmem_write_latency_cycles,
                                 tmem_write_bandwidth_bytes_per_cycle,
                                 tmem_queue_depth,
                                 startup_delay_cycles,
                                 block_schedule_latency_cycles));
        }

        memory_system = std::make_unique<MemorySystem>(
            l2_cache_size_bytes, l2_line_size_bytes, l2_hit_latency,
            l2_bandwidth_bytes_per_cycle, l2_write_bandwidth_bytes_per_cycle,
            l2_queue_depth, l2_bypass_cache, dram_bandwidth_bytes_per_cycle,
            dram_latency_cycles, core_freq_mhz, dram_freq_mhz, tensors);
        l2 = memory_system->l2();
        node_id_to_sp = nullptr;
        node_id_to_sm = nullptr;
    }

    ~GPU() {
        for (auto* sm : sms) delete sm;
        delete noc;
    }

    void set_node_to_sp_map(std::vector<Subpartition*>* map) {
        node_id_to_sp = map;
        for (auto* sm : sms) {
            sm->set_node_to_sp_map(map);
        }
    }

    void set_node_to_sm_map(std::vector<int>* map) {
        node_id_to_sm = map;
    }

    // Step function for GPU
    void step(int current_cycle);
};

} // namespace GTSim

#endif // GPU_H
