#include "gpu.h"

namespace GTSim {

// Route a completed remote access back to the originating subpartition
static void route_completion_to_origin(int node_id, int src_sm_id,
                                        int current_cycle,
                                        std::vector<Subpartition*>* node_id_to_sp,
                                        std::vector<SM*>& sms) {
    if (!node_id_to_sp || node_id < 0 ||
        node_id >= static_cast<int>(node_id_to_sp->size())) {
        return;
    }
    auto* sp = (*node_id_to_sp)[node_id];
    if (!sp) return;

    bool finished = sp->handle_memory_completion(node_id, current_cycle);
    if (finished && src_sm_id >= 0 &&
        src_sm_id < static_cast<int>(sms.size())) {
        sms[src_sm_id]->on_node_complete(node_id);
    }
}

static bool enqueue_forward_to_destination_sram(
    const NoCRequest& req, int current_cycle, std::vector<SM*>& sms,
    std::unordered_multimap<int, NoCPendingEntry>& pending_remote_reads,
    std::unordered_multimap<int, NoCPendingEntry>& pending_remote_writes) {
    int dst = req.dst_sm_id;
    if (dst < 0 || dst >= static_cast<int>(sms.size())) {
        return true;
    }

    bool enqueued = sms[dst]->sram->enqueue(req.node_id, req.bytes, current_cycle,
                                            req.bank_conflict, req.is_write);
    if (!enqueued) {
        return false;
    }

    NoCPendingEntry entry{req.node_id, req.src_sm_id, req.dst_sm_id,
                          req.bytes, req.bank_conflict};
    if (req.is_write) {
        pending_remote_writes.insert({req.node_id, entry});
    } else {
        pending_remote_reads.insert({req.node_id, entry});
    }
    return true;
}

void GPU::step(int current_cycle) {
    // Step 1: Step L2 and its DRAM backend through the memory system.
    auto completed = memory_system->step(current_cycle);
    for (int node_id : completed) {
        if (node_id_to_sp && node_id >= 0 &&
            node_id < static_cast<int>(node_id_to_sp->size())) {
            if (auto* sp = (*node_id_to_sp)[node_id]) {
                bool finished = sp->handle_memory_completion(node_id, current_cycle);
                if (finished && node_id_to_sm &&
                    node_id < static_cast<int>(node_id_to_sm->size())) {
                    int sm_id = (*node_id_to_sm)[node_id];
                    if (sm_id >= 0 && sm_id < static_cast<int>(sms.size())) {
                        sms[sm_id]->on_node_complete(node_id);
                    }
                }
            }
        }
    }

    // Step 2: Retry forward->SRAM failures (destination SRAM backpressure).
    if (noc) {
        size_t forward_retry_count = forward_to_sram_retry_queue.size();
        while (forward_retry_count-- > 0) {
            NoCForwardRetryEntry retry = forward_to_sram_retry_queue.front();
            forward_to_sram_retry_queue.pop_front();
            if (!enqueue_forward_to_destination_sram(
                    retry.req, current_cycle, sms,
                    pending_remote_reads, pending_remote_writes)) {
                forward_to_sram_retry_queue.push_back(retry);
            }
        }
    }

    // Step 3: Step NoC and handle completed NoC requests
    if (noc) {
        auto noc_completed = noc->step(current_cycle);
        for (const auto& req : noc_completed) {
            if (req.phase == NoCPhase::FORWARD) {
                // Forward complete: enqueue to target SM's SRAM.
                // If SRAM is full, keep request and retry next cycle.
                if (!enqueue_forward_to_destination_sram(
                        req, current_cycle, sms,
                        pending_remote_reads, pending_remote_writes)) {
                    forward_to_sram_retry_queue.push_back({req});
                }
            } else {
                // RETURN complete: read data arrived back at source SM
                route_completion_to_origin(req.node_id, req.src_sm_id,
                                           current_cycle, node_id_to_sp, sms);
            }
        }

        // Retry return-hop enqueue after NoC step so freed queue slots are
        // visible before new forward traffic is injected by SM pipelines.
        size_t return_retry_count = return_retry_queue.size();
        while (return_retry_count-- > 0) {
            NoCReturnRetryEntry retry = return_retry_queue.front();
            return_retry_queue.pop_front();
            const auto& entry = retry.entry;
            if (!noc->enqueue_return(entry.node_id, entry.src_sm_id,
                                     entry.dst_sm_id, entry.bytes,
                                     entry.bank_conflict, current_cycle)) {
                return_retry_queue.push_back(retry);
            }
        }
    }

    // Step 4: Step each SM
    // If NoC is enabled, collect unhandled SRAM completions (remote accesses)
    if (noc) {
        for (auto* sm_inst : sms) {
            std::vector<int> unhandled;
            sm_inst->step(current_cycle, l2, &unhandled);

            for (int node_id : unhandled) {
                // Check pending_remote_reads: need NoC return hop
                auto read_range = pending_remote_reads.equal_range(node_id);
                auto read_it = read_range.first;
                if (read_it != pending_remote_reads.end()) {
                    const auto& entry = read_it->second;
                    // Enqueue return: data travels from dst_sm back to src_sm
                    if (!noc->enqueue_return(entry.node_id, entry.src_sm_id,
                                             entry.dst_sm_id, entry.bytes,
                                             entry.bank_conflict, current_cycle)) {
                        return_retry_queue.push_back({entry});
                    }
                    pending_remote_reads.erase(read_it);
                    continue;
                }

                // Check pending_remote_writes: route directly to origin
                auto write_range = pending_remote_writes.equal_range(node_id);
                auto write_it = write_range.first;
                if (write_it != pending_remote_writes.end()) {
                    const auto& entry = write_it->second;
                    route_completion_to_origin(entry.node_id, entry.src_sm_id,
                                               current_cycle, node_id_to_sp, sms);
                    pending_remote_writes.erase(write_it);
                }
            }
        }
    } else {
        // No NoC: original path
        for (auto* sm_inst : sms) {
            sm_inst->step(current_cycle, l2);
        }
    }
}

} // namespace GTSim
