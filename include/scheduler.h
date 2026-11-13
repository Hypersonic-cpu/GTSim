#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "dag_node.h"
#include "pipeline.h"
#include "memory.h"
#include "tma.h"
#include "tmem.h"
#include <vector>
#include <deque>
#include <cstdint>
#include <unordered_map>
#include <optional>
#include <set>
#include <array>
#include <algorithm>

namespace GTSim {

// Forward declarations
class Memory;

enum class SchedulePolicy { GTO, RR };

// Scheduler implementing per-subpartition GTO (Greedy-Then-Oldest) policy
class Scheduler {
public:
    int scheduler_warp_num;
    std::vector<std::vector<DAGNode*>> scheduler_warp; // Nodes per warp
    std::vector<std::uint8_t>* scoreboard; // Track completed nodes (global)
    std::vector<std::deque<DAGNode*>> ready_warp; // Ready nodes per warp
    std::unordered_map<int, DAGNode*> node_dict;  // Quick node lookup
    std::vector<int> warp_head_index;             // Current head per warp

    // Track last issued pipeline per warp for fairness (O2 feature)
    std::vector<std::optional<std::string>> last_issued_pipeline;

    // Track last issue cycle for each warp (for GTO scheduling)
    std::vector<int> last_issue_cycle;

    // Track which warp issued last cycle
    int last_issued_warp;

    // Track which subpartition this scheduler belongs to
    int subpartition_id;

    // Configuration flags
    bool O2_enabled; // Dynamic allocation
    SchedulePolicy schedule_policy;
    int rr_cursor;

    // TB residency optimization: O(1) lookup with event-driven updates
    std::vector<bool> warp_is_resident; // Per-warp residency flag
    std::unordered_map<int, std::vector<int>>
        tb_to_warps;                 // Map TB ID to its warp IDs
    std::vector<int> resident_warps; // Event-maintained scheduler candidates

    // Pipeline lookup optimization: O(1) hash map instead of O(N) linear search
    std::unordered_map<std::string, Pipeline*> pipeline_map;
    std::vector<Scheduler*>* peer_schedulers;
    std::vector<Pipeline*>* pipelines_ptr;
    TMAUnit* tma_unit;
    TMEMUnit* tmem_unit;
    std::vector<int> tma_next_issue_cycle;
    std::vector<int> tma_issue_interval_cache;

    int sram_next_issue_cycle;
    static constexpr int kMemoryOpTypes = 11;
    std::array<std::vector<int>, kMemoryOpTypes> memory_next_issue_cycle;
    int startup_delay_cycles;

    Scheduler(int warp_count, bool o2 = false,
              SchedulePolicy policy = SchedulePolicy::GTO)
        : scheduler_warp_num(warp_count),
          scheduler_warp(warp_count),
          scoreboard(nullptr),
          ready_warp(warp_count),
          warp_head_index(warp_count, 0),
          last_issued_pipeline(warp_count, std::nullopt),
          last_issue_cycle(warp_count, -1000),
          last_issued_warp(-1),
          subpartition_id(-1),
          O2_enabled(o2),
          schedule_policy(policy),
          rr_cursor(0),
          warp_is_resident(warp_count, false),
          peer_schedulers(nullptr),
          pipelines_ptr(nullptr),
          tma_unit(nullptr),
          tmem_unit(nullptr),
          tma_next_issue_cycle(warp_count, 0),
          tma_issue_interval_cache(warp_count, -1),
          sram_next_issue_cycle(0),
          memory_next_issue_cycle{},
          startup_delay_cycles(0) {} // Default: no warps resident

    // Clear graph-local state while retaining the scheduler object and its
    // persistent pipeline/TMA/TMEM bindings.  A new kernel can therefore
    // reuse the GPU's scheduling resources without reallocating them.
    void reset_for_graph(int warp_count) {
        scheduler_warp_num = warp_count;
        scheduler_warp.assign(warp_count, {});
        ready_warp.assign(warp_count, {});
        warp_head_index.assign(warp_count, 0);
        last_issued_pipeline.assign(warp_count, std::nullopt);
        last_issue_cycle.assign(warp_count, -1000);
        warp_is_resident.assign(warp_count, false);
        tma_next_issue_cycle.assign(warp_count, 0);
        tma_issue_interval_cache.assign(warp_count, -1);
        node_dict.clear();
        tb_to_warps.clear();
        resident_warps.clear();
        scoreboard = nullptr;
        subpartition_id = -1;
        last_issued_warp = -1;
        rr_cursor = 0;
        sram_next_issue_cycle = 0;
        init_memory_issue_cycles(warp_count);
    }

    void init_memory_issue_cycles(int warp_count) {
        for (auto& vec : memory_next_issue_cycle) {
            vec.assign(warp_count, 0);
        }
    }

    // Initialize scheduler with nodes
    void init_with_nodes(const std::vector<DAGNode*>& nodes, int total_nodes,
                         int sp_id,
                         std::vector<std::uint8_t>* global_scoreboard) {
        reset_for_graph(scheduler_warp_num);
        scoreboard = global_scoreboard;
        if (scoreboard &&
            scoreboard->size() < static_cast<size_t>(total_nodes)) {
            scoreboard->resize(total_nodes, 0);
        }
        subpartition_id = sp_id;

        for (auto* node : nodes) {
            node_dict[node->id] = node;
            scheduler_warp[node->warp_id].push_back(node);
            if (node->remaining_deps == 0) {
                ready_warp[node->warp_id].push_back(node);
            }
        }

        // Build TB-to-warps mapping for event-driven residency tracking
        for (int warp_id = 0; warp_id < scheduler_warp_num; ++warp_id) {
            if (!scheduler_warp[warp_id].empty()) {
                // All nodes in a warp belong to the same TB - check first node
                int tb_id = scheduler_warp[warp_id][0]->thread_block_id;
                tb_to_warps[tb_id].push_back(warp_id);
            }
        }
    }

    // Event-driven residency updates: called when a TB is dispatched to this SM
    void on_tb_dispatch(int tb_id) {
        auto it = tb_to_warps.find(tb_id);
        if (it != tb_to_warps.end()) {
            for (int warp_id : it->second) {
                if (!warp_is_resident[warp_id]) {
                    resident_warps.push_back(warp_id);
                }
                warp_is_resident[warp_id] = true;
            }
        }
    }

    // Event-driven residency updates: called when a TB completes and is retired
    void on_tb_retire(int tb_id) {
        auto it = tb_to_warps.find(tb_id);
        if (it != tb_to_warps.end()) {
            for (int warp_id : it->second) {
                warp_is_resident[warp_id] = false;
                auto resident_it = std::find(resident_warps.begin(),
                                             resident_warps.end(), warp_id);
                if (resident_it != resident_warps.end()) {
                    resident_warps.erase(resident_it);
                }
            }
        }
    }

    // Main scheduling function - returns list of issued nodes
    std::vector<DAGNode*> schedule(int current_cycle,
                                   std::vector<Pipeline*>& pipelines,
                                   Memory* sram, L2Cache* l2);

    void enqueue_ready(DAGNode* node) {
        if (node->finished || node->remaining_deps != 0) return;
        ready_warp[node->warp_id].push_back(node);
    }

    void throttle_sram_issue(int next_cycle) {
        if (next_cycle > sram_next_issue_cycle) {
            sram_next_issue_cycle = next_cycle;
        }
    }

    void throttle_memory_issue(int op_index, int warp_id, int next_cycle) {
        if (op_index < 0 || op_index >= kMemoryOpTypes) {
            return;
        }
        auto& cycles = memory_next_issue_cycle[op_index];
        if (warp_id < 0 || warp_id >= static_cast<int>(cycles.size())) {
            return;
        }
        if (next_cycle > cycles[warp_id]) {
            cycles[warp_id] = next_cycle;
        }
    }

    void set_peer_schedulers(std::vector<Scheduler*>* peers) {
        peer_schedulers = peers;
    }

    void set_pipelines(std::vector<Pipeline*>* pipes) {
        pipelines_ptr = pipes;
    }

    void set_tma_unit(TMAUnit* unit) {
        tma_unit = unit;
    }

    void set_tmem_unit(TMEMUnit* unit) {
        tmem_unit = unit;
    }

    bool tma_ready(const DAGNode* node, int current_cycle) const {
        if (node->warp_id < 0 ||
            node->warp_id >= static_cast<int>(tma_next_issue_cycle.size())) {
            return false;
        }
        return current_cycle >= tma_next_issue_cycle[node->warp_id];
    }

    void ensure_pipeline_map() {
        if (!pipeline_map.empty() || pipelines_ptr == nullptr) {
            return;
        }
        for (auto* pipe : *pipelines_ptr) {
            pipeline_map[pipe->pipeline_name] = pipe;
        }
    }

private:
    // Helper: Check if node is ready and find its pipeline
    Pipeline* node_ready_and_pipe(DAGNode* node, int current_cycle,
                                  std::vector<Pipeline*>& pipelines,
                                  Memory* sram, L2Cache* l2);

    // Helper: Try to issue one instruction from given warp
    std::pair<DAGNode*, Pipeline*>
    try_issue_from_warp(int warp_id, int current_cycle,
                        std::vector<Pipeline*>& pipelines, Memory* sram,
                        L2Cache* l2);

    // Helper: Issue a node and update tracking state
    void issue_node(DAGNode* node, Pipeline* pipe, int current_cycle,
                    Memory* sram, L2Cache* l2, int issue_span_override = -1);
};

} // namespace GTSim

#endif // SCHEDULER_H
