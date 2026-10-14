#ifndef SM_H
#define SM_H

#include "subpartition.h"
#include "memory.h"
#include "tma.h"
#include "tmem.h"
#include <algorithm>
#include <vector>
#include <set>
#include <queue>
#include <deque>
#include <unordered_map>

namespace GTSim {

// Streaming Multiprocessor containing 4 subpartitions and shared memory
class SM {
public:
    int sm_id;
    std::vector<Subpartition*> sps;  // 4 subpartitions
    std::vector<Scheduler*> sp_schedulers;
    Memory* sram;                     // Shared memory
    Memory* tmem_mem;                 // Tensor memory access model
    TMAUnit* tma;                     // TMA unit
    TMEMUnit* tmem;                   // Tensor Memory unit
    std::vector<Subpartition*>* node_id_to_sp;

    std::unordered_map<int, CTAResources> cta_resources;
    int shared_capacity{228 * 1024};
    int register_capacity{65536 * 4};
    int warp_capacity{64};
    int shared_used{0};
    int registers_used{0};
    int warps_used{0};
    bool can_admit(int tb) const;
    void reserve_cta(int tb, int sign);

    int max_concurrent_blocks;  // Maximum concurrent thread blocks per SM

    // Thread block residency scheduling
    int max_resident_tbs;                          // R: maximum resident TBs (e.g., 2)
    std::set<int> resident_tb_ids;                 // Currently resident TB IDs
    std::queue<int> waiting_tb_queue;              // TBs waiting to be dispatched
    std::deque<int> tb_dispatch_ready;             // Ready cycles for dispatching waiting TBs
    std::set<int> all_tb_ids;                      // All unique TB IDs in workload
    std::unordered_map<int, std::vector<DAGNode*>> tb_to_nodes;  // Map TB ID to its nodes
    std::unordered_map<int, int> tb_remaining_count;  // Remaining nodes per TB (for fast completion check)
    std::vector<int> node_id_to_tb_id;             // Direct mapping: node ID → TB ID (O(1) lookup)
    int block_schedule_latency_cycles;            // Delay before scheduling next TB after completion

    SM(int id, int warp_count, int tensor_throughput, int simd_throughput,
       int sfu_throughput, int ls_throughput_bytes,
       int tensor_latency_cycles, int simd_latency_cycles,
       int sfu_latency_cycles, int shfl_latency_cycles, int ls_latency_cycles,
       int tensor_width, int simd_width, int sfu_width, int ls_width_bytes,
       int sram_latency_cycles, int sram_bandwidth_bytes_per_cycle, int sram_queue_depth,
       bool o2, SchedulePolicy policy, int max_blocks,
       int l2_max_transactions_per_cycle_per_sp,
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
       int block_schedule_latency_cycles)
        : sm_id(id), max_concurrent_blocks(max_blocks), max_resident_tbs(max_blocks),
          block_schedule_latency_cycles(block_schedule_latency_cycles) {

        // Create 4 subpartitions
        for (int i = 0; i < 4; ++i) {
            sps.push_back(new Subpartition(i, warp_count, tensor_throughput, simd_throughput,
                                           sfu_throughput, ls_throughput_bytes,
                                           tensor_latency_cycles, simd_latency_cycles,
                                           sfu_latency_cycles, shfl_latency_cycles, ls_latency_cycles,
                                           tensor_width, simd_width, sfu_width, ls_width_bytes,
                                           o2, policy, l2_max_transactions_per_cycle_per_sp));
        }

        for (auto* sp : sps) {
            sp_schedulers.push_back(sp->sp_scheduler);
            sp->tmem_write_extra_latency_cycles =
                std::max(0, tmem_write_latency_cycles - tmem_read_latency_cycles);
        }
        for (auto* sp : sps) {
            sp->sp_scheduler->set_peer_schedulers(&sp_schedulers);
        }

        // Create shared memory (SRAM) with separate read/write ports
        sram = new Memory(sram_latency_cycles, sram_bandwidth_bytes_per_cycle, sram_queue_depth);
        tmem_mem = new Memory(tmem_read_latency_cycles, tmem_read_bandwidth_bytes_per_cycle,
                              tmem_queue_depth);
        tmem_mem->memory_write_bandwidth = tmem_write_bandwidth_bytes_per_cycle;
        tmem_mem->memory_write_queue_depth = tmem_queue_depth;
        tmem_mem->memory_write_occupied_until = 0;
        tmem_mem->memory_latency = tmem_read_latency_cycles;
        // Memory class has shared latency field for read/write; use write enqueue
        // bank_conflict_factor to model additional write-side latency delta.
        tma = new TMAUnit(tma_setup_latency_cycles, tma_issue_interval_cycles,
                          tma_issue_rate_bytes_per_cycle);
        tmem = new TMEMUnit(tmem_cols_per_cta, tmem_issue_interval_cycles);
        tmem->init_warp_state(warp_count);
        for (auto* sp : sps) {
            sp->sp_scheduler->set_tma_unit(tma);
            sp->sp_scheduler->set_tmem_unit(tmem);
            sp->sp_scheduler->startup_delay_cycles = startup_delay_cycles;
        }
        node_id_to_sp = nullptr;
    }

    ~SM() {
        for (auto* sp : sps) delete sp;
        delete sram;
        delete tmem_mem;
        delete tma;
        delete tmem;
    }

    // Initialize thread block scheduler with all nodes assigned to this SM
    void initialize_tb_scheduler(const std::vector<DAGNode*>& all_nodes);

    // Check if a thread block has completed (all its nodes finished)
    bool is_tb_complete(int tb_id);

    // Dispatch the next waiting TB from the queue
    bool dispatch_next_tb(int current_cycle);

    // Check if a TB is currently resident
    bool is_tb_resident(int tb_id) const {
        return resident_tb_ids.find(tb_id) != resident_tb_ids.end();
    }

    // Step function for SM
    // If unhandled_sram_completions is non-null, SRAM completions for nodes
    // not owned by this SM's subpartitions are appended there (for NoC routing).
    void step(int current_cycle, L2Cache* l2,
              std::vector<int>* unhandled_sram_completions = nullptr);

    void on_node_complete(int node_id);

    void set_node_to_sp_map(std::vector<Subpartition*>* map) {
        node_id_to_sp = map;
    }

    // Set NoC pointer on all subpartitions
    void set_noc(class NoC* noc_ptr);
};

} // namespace GTSim

#endif // SM_H
