#include "subpartition.h"
#include "noc.h"
#include <algorithm>
#include <utility>
#include <cstdio>

namespace GTSim {

static bool is_l2_dram_op(const DAGNode* node) {
    return node->op_type == OpType::LD_DRAM2REG ||
           node->op_type == OpType::ST_REG2DRAM ||
           node->op_type == OpType::CP_DRAM2SRAM ||
           node->op_type == OpType::CP_SRAM2DRAM;
}

static bool is_memory_op(const DAGNode* node) {
    return node->op_type == OpType::LD_SRAM2REG ||
           node->op_type == OpType::LD_SRAM2REG_DSM ||
           node->op_type == OpType::ST_REG2SRAM ||
           node->op_type == OpType::ST_REG2SRAM_DSM ||
           node->op_type == OpType::TCGEN05_LD ||
           node->op_type == OpType::TCGEN05_ST ||
           node->op_type == OpType::CP_SRAM2TMEM ||
           is_l2_dram_op(node);
}

static int compute_chunk_bytes(const DAGNode* node, int subop_index, int throughput_bytes) {
    int total_bytes = node->tile.r * node->tile.c * node->element_size_bytes();
    int offset = subop_index * throughput_bytes;
    int remaining = total_bytes - offset;
    if (remaining <= 0) return 0;
    return std::min(throughput_bytes, remaining);
}

static int get_pipeline_throughput(const Subpartition* sp, const std::string& name) {
    for (auto* pipe : sp->pipelines) {
        if (pipe->pipeline_name == name) {
            return pipe->pipeline_throughput;
        }
    }
    return 1;
}

static bool is_load_like_op(const DAGNode* node) {
    return node->op_type == OpType::LD_SRAM2REG ||
           node->op_type == OpType::LD_SRAM2REG_DSM ||
           node->op_type == OpType::LD_DRAM2REG ||
           node->op_type == OpType::CP_DRAM2SRAM ||
           node->op_type == OpType::CP_SRAM2TMEM;
}

static bool is_store_like_op(const DAGNode* node) {
    return node->op_type == OpType::ST_REG2SRAM ||
           node->op_type == OpType::ST_REG2SRAM_DSM ||
           node->op_type == OpType::ST_REG2DRAM ||
           node->op_type == OpType::TCGEN05_ST ||
           node->op_type == OpType::CP_SRAM2DRAM;
}

static int get_memory_pipeline_throughput(const Subpartition* sp, const DAGNode* node) {
    if (is_load_like_op(node)) {
        return get_pipeline_throughput(sp, "LD");
    }
    if (is_store_like_op(node)) {
        return get_pipeline_throughput(sp, "ST");
    }
    return get_pipeline_throughput(sp, node->pipeline_type);
}

void Subpartition::complete_node(DAGNode* node, int current_cycle) {
    if (node->finished) {
        return;
    }
    (*sp_scheduler->scoreboard)[node->id] = 1;
    node->finished = true;
    node->end = current_cycle;
    if (completed_node_count) {
        ++(*completed_node_count);
    }
    if (sp_scheduler->tmem_unit) {
        sp_scheduler->tmem_unit->on_complete(node);
    }

    // Count register usage
    sp_register->register_totally_used += node->register_change;

    for (auto* child : node->children) {
        if (child->remaining_deps > 0) {
            child->remaining_deps--;
            if (child->remaining_deps == 0) {
                child->ready_cycle = current_cycle + 1;
                if (node_id_to_sp && child->id >= 0 &&
                    child->id < static_cast<int>(node_id_to_sp->size())) {
                    if (auto* target_sp = (*node_id_to_sp)[child->id]) {
                        target_sp->sp_scheduler->enqueue_ready(child);
                    }
                }
            }
        }
    }
}

void Subpartition::resolve_issue_dependencies(DAGNode* node, int current_cycle) {
    if (node->issue_deps_resolved) {
        return;
    }
    node->issue_deps_resolved = true;

    for (auto* child : node->issue_children) {
        if (child->remaining_deps > 0) {
            child->remaining_deps--;
            if (child->remaining_deps == 0) {
                child->ready_cycle = current_cycle + 1;
                if (node_id_to_sp && child->id >= 0 &&
                    child->id < static_cast<int>(node_id_to_sp->size())) {
                    if (auto* target_sp = (*node_id_to_sp)[child->id]) {
                        target_sp->sp_scheduler->enqueue_ready(child);
                    }
                }
            }
        }
    }
}

bool Subpartition::handle_memory_completion(int node_id, int current_cycle) {
    auto it = sp_scheduler->node_dict.find(node_id);
    if (it != sp_scheduler->node_dict.end()) {
        DAGNode* node = it->second;
        if (node->pending_transactions > 0) {
            node->pending_transactions--;
            if (node->pending_transactions > 0) {
                return false;
            }
        }
        complete_node(node, current_cycle);
        pending_memory_ops.erase(node_id);
        return true;
    }
    return false;
}

void Subpartition::handle_tma_issue_complete(int node_id, int current_cycle) {
    auto it = sp_scheduler->node_dict.find(node_id);
    if (it == sp_scheduler->node_dict.end()) {
        return;
    }
    DAGNode* node = it->second;
    if (!node->issue_done) {
        node->issue_done = true;
        node->tma_issue_complete_cycle = current_cycle;
        resolve_issue_dependencies(node, current_cycle);
    }
}

bool Subpartition::try_enqueue_memory(DAGNode* node, int bytes_val, int current_cycle,
                                       int bank_conflict_factor, int subop_index,
                                       const CacheLineKey& line_key, bool use_line_key,
                                       Memory* sram, Memory* tmem_mem, L2Cache* l2) {
    // Dispatch based on operation type
    if (node->op_type == OpType::LD_SRAM2REG) {
        // Check for remote SRAM access via NoC
        if (noc && node->target_sm_id >= 0 && node->target_sm_id != owning_sm_id) {
            // Cross-cluster remote access is not supported (DSMEM only within GPC)
            if (!noc->same_cluster(owning_sm_id, node->target_sm_id)) {
                static bool warned_cross_cluster = false;
                if (!warned_cross_cluster) {
                    std::fprintf(stderr, "[NoC] Warning: cross-cluster remote SRAM access "
                                 "rejected (SM %d -> SM %d, clusters %d vs %d). "
                                 "Falling back to local SRAM.\n",
                                 owning_sm_id, node->target_sm_id,
                                 noc->cluster_of(owning_sm_id),
                                 noc->cluster_of(node->target_sm_id));
                    warned_cross_cluster = true;
                }
                // Fall through to local SRAM path
            } else {
                return noc->enqueue(node->id, owning_sm_id, node->target_sm_id,
                                    bytes_val, false, bank_conflict_factor, current_cycle);
            }
        }
        // ld.sram2reg uses read queue with bank conflicts
        if (!sram->is_full(false)) {
            return sram->enqueue(node->id, bytes_val, current_cycle,
                               bank_conflict_factor, false);
        }
    } else if (node->op_type == OpType::CP_SRAM2TMEM) {
        // cp.sram2tmem models one-way shared-memory read into Tensor Memory.
        int write_extra = tmem_write_extra_latency_cycles;
        if (!tmem_mem->is_full(true)) {
            return tmem_mem->enqueue(node->id, bytes_val, current_cycle,
                                     write_extra, true);
        }
    } else if (node->op_type == OpType::TCGEN05_LD) {
        if (!tmem_mem->is_full(false)) {
            return tmem_mem->enqueue(node->id, bytes_val, current_cycle,
                                     bank_conflict_factor, false);
        }
    } else if (node->op_type == OpType::TCGEN05_ST) {
        int write_extra = tmem_write_extra_latency_cycles;
        if (!tmem_mem->is_full(true)) {
            return tmem_mem->enqueue(node->id, bytes_val, current_cycle,
                                     write_extra, true);
        }
    } else if (node->op_type == OpType::LD_SRAM2REG_DSM) {
        if (noc != nullptr) {
            int dst_sm_id = (node->target_sm_id >= 0) ? node->target_sm_id : owning_sm_id;
            return noc->enqueue(node->id, owning_sm_id, dst_sm_id,
                                bytes_val, false, bank_conflict_factor, current_cycle);
        }
    } else if (node->op_type == OpType::ST_REG2SRAM) {
        // Check for remote SRAM access via NoC
        if (noc && node->target_sm_id >= 0 && node->target_sm_id != owning_sm_id) {
            // Cross-cluster remote access is not supported (DSMEM only within GPC)
            if (!noc->same_cluster(owning_sm_id, node->target_sm_id)) {
                static bool warned_cross_cluster_st = false;
                if (!warned_cross_cluster_st) {
                    std::fprintf(stderr, "[NoC] Warning: cross-cluster remote SRAM store "
                                 "rejected (SM %d -> SM %d, clusters %d vs %d). "
                                 "Falling back to local SRAM.\n",
                                 owning_sm_id, node->target_sm_id,
                                 noc->cluster_of(owning_sm_id),
                                 noc->cluster_of(node->target_sm_id));
                    warned_cross_cluster_st = true;
                }
                // Fall through to local SRAM path
            } else {
                return noc->enqueue(node->id, owning_sm_id, node->target_sm_id,
                                    bytes_val, true, bank_conflict_factor, current_cycle);
            }
        }
        // st.reg2sram uses write queue
        if (!sram->is_full(true)) {
            return sram->enqueue(node->id, bytes_val, current_cycle,
                                 bank_conflict_factor, true);
        }
    } else if (node->op_type == OpType::ST_REG2SRAM_DSM) {
        if (noc != nullptr) {
            int dst_sm_id = (node->target_sm_id >= 0) ? node->target_sm_id : owning_sm_id;
            return noc->enqueue(node->id, owning_sm_id, dst_sm_id,
                                bytes_val, true, bank_conflict_factor, current_cycle);
        }
    } else if (is_l2_dram_op(node)) {
        if (l2_issue_cycle != current_cycle) {
            l2_issue_cycle = current_cycle;
            l2_issue_count = 0;
        }
        if (l2_max_transactions_per_cycle > 0 &&
            l2_issue_count >= l2_max_transactions_per_cycle) {
            return false;
        }
        bool is_write = (node->op_type == OpType::ST_REG2DRAM || node->op_type == OpType::CP_SRAM2DRAM);
        if (use_line_key) {
            if (l2->enqueue_transaction_key(node->id, line_key, is_write)) {
                l2_issue_count++;
                return true;
            }
            return false;
        }
        if (l2->enqueue_transaction(*node, is_write, subop_index)) {
            l2_issue_count++;
            return true;
        }
    }
    return false;
}

std::vector<int> Subpartition::step(int current_cycle, Memory* sram, Memory* tmem_mem,
                                    L2Cache* l2) {
    if (l2_issue_cycle != current_cycle) {
        l2_issue_cycle = current_cycle;
        l2_issue_count = 0;
    }
    std::vector<int> non_memory_completions;

    // Step -1: Retry deferred 2-CTA UMMA completions when peer completion arrives.
    if (!deferred_tmem_pair_completions.empty() && sp_scheduler->tmem_unit) {
        std::vector<int> still_deferred;
        still_deferred.reserve(deferred_tmem_pair_completions.size());
        for (int node_id : deferred_tmem_pair_completions) {
            auto it = sp_scheduler->node_dict.find(node_id);
            if (it == sp_scheduler->node_dict.end()) {
                continue;
            }
            if (sp_scheduler->tmem_unit->consume_release_token(node_id)) {
                complete_node(it->second, current_cycle);
                non_memory_completions.push_back(node_id);
            } else {
                still_deferred.push_back(node_id);
            }
        }
        deferred_tmem_pair_completions.swap(still_deferred);
    }

    // Step 0: Complete barrier nodes
    while (!barrier_queue.empty()) {
        const auto& [node_id, complete_cycle] = barrier_queue.front();
        if (current_cycle < complete_cycle) break;
        auto it = sp_scheduler->node_dict.find(node_id);
        if (it != sp_scheduler->node_dict.end()) {
            complete_node(it->second, current_cycle);
            non_memory_completions.push_back(node_id);
        }
        barrier_queue.pop_front();
    }
    // Step 1: Retry any pending memory enqueues that failed due to full queue
    // Use remove_if to eliminate successful retries in-place (avoids temporary vector allocation)
    retry_queue_sram.erase(
        std::remove_if(retry_queue_sram.begin(), retry_queue_sram.end(),
            [&](const auto& retry) {
                const auto& [node_id, bytes_val, bank_conflict_factor, subop_index,
                             is_l2, ready_cycle, line_key, use_line_key] = retry;

                if (current_cycle < ready_cycle) {
                    return false;  // Not ready to retry yet
                }

                auto it = sp_scheduler->node_dict.find(node_id);
                if (it == sp_scheduler->node_dict.end()) {
                    return true;  // Remove invalid node
                }

                DAGNode* node = it->second;

                // Try to enqueue using helper function
                bool success = try_enqueue_memory(node, bytes_val, current_cycle,
                                                 bank_conflict_factor,
                                                 is_l2 ? subop_index : 0,
                                                 line_key, use_line_key, sram, tmem_mem, l2);

                if (success) {
                    pending_memory_ops[node->id] = node->op;
                    if (node->op_type == OpType::LD_SRAM2REG ||
                        node->op_type == OpType::LD_SRAM2REG_DSM) {
                        sp_scheduler->throttle_sram_issue(current_cycle + bank_conflict_factor);
                    }
                    return true;  // Remove from retry queue
                }
                return false;  // Keep in retry queue
            }),
        retry_queue_sram.end()
    );

    retry_queue_l2.erase(
        std::remove_if(retry_queue_l2.begin(), retry_queue_l2.end(),
            [&](const auto& retry) {
                const auto& [node_id, bytes_val, bank_conflict_factor, subop_index,
                             is_l2, ready_cycle, line_key, use_line_key] = retry;

                if (current_cycle < ready_cycle) {
                    return false;
                }

                auto it = sp_scheduler->node_dict.find(node_id);
                if (it == sp_scheduler->node_dict.end()) {
                    return true;
                }

                DAGNode* node = it->second;

                bool success = try_enqueue_memory(node, bytes_val, current_cycle,
                                                 bank_conflict_factor,
                                                 is_l2 ? subop_index : 0,
                                                 line_key, use_line_key, sram, tmem_mem, l2);

                if (success) {
                    pending_memory_ops[node->id] = node->op;
                    if (node->op_type == OpType::LD_SRAM2REG ||
                        node->op_type == OpType::LD_SRAM2REG_DSM) {
                        sp_scheduler->throttle_sram_issue(current_cycle + bank_conflict_factor);
                    }
                    return true;
                }
                return false;
            }),
        retry_queue_l2.end()
    );

    // Step 2: Step pipelines
    std::vector<std::pair<int, int>> all_completed;
    for (auto* pipe : pipelines) {
        auto completed = pipe->step(current_cycle);
        all_completed.insert(all_completed.end(), completed.begin(), completed.end());
    }

    // Step 3: Handle completed pipeline executions
    for (const auto& [node_id, subop_index] : all_completed) {
        auto it = sp_scheduler->node_dict.find(node_id);
        if (it == sp_scheduler->node_dict.end()) continue;

        DAGNode* node = it->second;

        // Use enum for fast dispatch
        if (is_memory_op(node)) {
            int bytes_val = 0;
            if (node->op_type == OpType::LD_SRAM2REG ||
                node->op_type == OpType::ST_REG2SRAM ||
                node->op_type == OpType::LD_SRAM2REG_DSM ||
                node->op_type == OpType::ST_REG2SRAM_DSM) {
                bytes_val = compute_sram_subop_bytes(*node, get_access_granularity_bytes(*node),
                                                     subop_index);
            } else if (node->op_type == OpType::CP_SRAM2TMEM ||
                       node->op_type == OpType::TCGEN05_LD ||
                       node->op_type == OpType::TCGEN05_ST) {
                // Bulk one-shot TMEM transfer size in bytes.
                bytes_val = node->tile.r * node->tile.c * node->element_size_bytes();
            } else {
                int throughput_bytes = get_memory_pipeline_throughput(this, node);
                bytes_val = compute_chunk_bytes(node, subop_index, throughput_bytes);
            }

            int bank_conflict_factor = 0;
            if (node->op_type == OpType::LD_SRAM2REG ||
                node->op_type == OpType::LD_SRAM2REG_DSM) {
                bank_conflict_factor = compute_sram_subop_conflict(*node,
                                                                  get_access_granularity_bytes(*node),
                                                                  subop_index);
            }
            int setup_latency = 0;
            if (is_memory_op(node)) {
                setup_latency = std::max(0, node->setup_latency);
            }

            if (is_l2_dram_op(node)) {
                auto lines = l2->get_subop_lines(*node, subop_index);
                if (lines.empty()) {
                    lines.push_back(l2->get_transaction_key(*node, 0));
                }
                for (const auto& line_key : lines) {
                    bool success = false;
                    if (setup_latency > 0) {
                        retry_queue_l2.push_back({node_id, bytes_val, bank_conflict_factor,
                                                  subop_index, true, current_cycle + setup_latency,
                                                  line_key, true});
                    } else {
                        success = try_enqueue_memory(node, bytes_val, current_cycle,
                                                     bank_conflict_factor, subop_index,
                                                     line_key, true, sram, tmem_mem, l2);
                    }

                    if (success) {
                        pending_memory_ops[node->id] = node->op;
                    } else {
                        if (setup_latency == 0) {
                            retry_queue_l2.push_back({node_id, bytes_val, bank_conflict_factor,
                                                      subop_index, true, current_cycle,
                                                      line_key, true});
                        }
                    }
                }
                continue;
            }

            // Try to enqueue using helper function (SRAM paths)
            bool success = false;
            if (setup_latency > 0) {
                retry_queue_sram.push_back({node_id, bytes_val, bank_conflict_factor,
                                            subop_index, false, current_cycle + setup_latency,
                                            CacheLineKey{0, 0, 0}, false});
            } else {
                success = try_enqueue_memory(node, bytes_val, current_cycle,
                                             bank_conflict_factor, subop_index,
                                             CacheLineKey{0, 0, 0}, false, sram, tmem_mem, l2);
            }

            if (success) {
                pending_memory_ops[node->id] = node->op;
                if (node->op_type == OpType::LD_SRAM2REG ||
                    node->op_type == OpType::LD_SRAM2REG_DSM ||
                    node->op_type == OpType::CP_SRAM2TMEM) {
                    sp_scheduler->throttle_sram_issue(current_cycle + bank_conflict_factor);
                }
            } else {
                if (setup_latency == 0) {
                    retry_queue_sram.push_back({node_id, bytes_val, bank_conflict_factor,
                                                subop_index, false, current_cycle,
                                                CacheLineKey{0, 0, 0}, false});
                }
            }
        } else {
            if (node->pending_transactions > 0) {
                node->pending_transactions--;
            }
            if (node->pending_transactions <= 0) {
                if (sp_scheduler->tmem_unit &&
                    node->op_type == OpType::TCGEN05_MMA &&
                    !sp_scheduler->tmem_unit->on_mma_completion_arrival(node)) {
                    deferred_tmem_pair_completions.push_back(node_id);
                    continue;
                }
                complete_node(node, current_cycle);
                non_memory_completions.push_back(node_id);
            }
        }
    }

    // Schedule new nodes (uses O(1) event-driven warp residency checks)
    auto issued = sp_scheduler->schedule(current_cycle, pipelines, sram, l2);
    for (auto* node : issued) {
        if (node->op_type == OpType::BARRIER) {
            int latency = std::max(0, node->setup_latency);
            barrier_queue.push_back({node->id, current_cycle + latency});
        }
        if (node->issue_done) {
            resolve_issue_dependencies(node, current_cycle);
        }
    }

    return non_memory_completions;
}

} // namespace GTSim
