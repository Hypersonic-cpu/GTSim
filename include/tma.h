#ifndef TMA_H
#define TMA_H

#include "dag_node.h"
#include "memory.h"
#include <deque>
#include <vector>

namespace GTSim {

struct TMARequest {
    int node_id;
    bool is_write;
    int ready_cycle;
    int line_size_bytes;
    int issue_rate_bytes_per_cycle;
    double credit_bytes;
    size_t next_line;
    std::vector<CacheLineKey> lines;
};

class TMAUnit {
public:
    TMAUnit(int setup_latency, int issue_interval, int issue_rate_bytes_per_cycle)
        : setup_latency_cycles(setup_latency),
          issue_interval_cycles(issue_interval),
          issue_rate_bytes_per_cycle(issue_rate_bytes_per_cycle) {}

    int get_issue_interval(const DAGNode* node) const {
        if (node && node->tma_issue_interval >= 0) {
            return node->tma_issue_interval;
        }
        return issue_interval_cycles;
    }

    int enqueue(DAGNode* node, int current_cycle, L2Cache* l2) {
        int line_size = l2->get_line_size_bytes();
        std::vector<CacheLineKey> lines;
        int subops = l2->get_subop_count(*node);
        for (int subop = 0; subop < subops; ++subop) {
            auto sub_lines = l2->get_subop_lines(*node, subop);
            lines.insert(lines.end(), sub_lines.begin(), sub_lines.end());
        }
        if (lines.empty()) {
            lines.push_back(l2->get_transaction_key(*node, 0));
        }

        int node_setup = node->tma_setup_latency >= 0 ? node->tma_setup_latency
                                                      : setup_latency_cycles;
        int node_issue_rate = node->tma_issue_rate_bytes_per_cycle >= 0
                                  ? node->tma_issue_rate_bytes_per_cycle
                                  : issue_rate_bytes_per_cycle;
        TMARequest req{
            node->id,
            node->op_type == OpType::CP_SRAM2DRAM_TMA,
            current_cycle + node_setup,
            line_size,
            node_issue_rate,
            0.0,
            0,
            std::move(lines)
        };
        setup_queue.push_back(std::move(req));
        return static_cast<int>(setup_queue.back().lines.size());
    }

    std::vector<int> step(int current_cycle, L2Cache* l2) {
        std::vector<int> issued_complete;
        while (!setup_queue.empty() && setup_queue.front().ready_cycle <= current_cycle) {
            pending_queue.push_back(std::move(setup_queue.front()));
            setup_queue.pop_front();
        }

        if (!active_valid && !pending_queue.empty()) {
            active = std::move(pending_queue.front());
            pending_queue.pop_front();
            active_valid = true;
        }

        if (!active_valid) {
            return issued_complete;
        }

        active.credit_bytes += static_cast<double>(active.issue_rate_bytes_per_cycle);

        while (active.next_line < active.lines.size() &&
               active.credit_bytes >= active.line_size_bytes) {
            const auto& line_key = active.lines[active.next_line];
            if (!l2->enqueue_transaction_key(active.node_id, line_key, active.is_write)) {
                break;
            }
            active.credit_bytes -= active.line_size_bytes;
            active.next_line++;
        }

        if (active.next_line >= active.lines.size()) {
            issued_complete.push_back(active.node_id);
            active_valid = false;
        }
        return issued_complete;
    }

private:
    int setup_latency_cycles;
    int issue_interval_cycles;
    int issue_rate_bytes_per_cycle;
    std::deque<TMARequest> setup_queue;
    std::deque<TMARequest> pending_queue;
    TMARequest active{};
    bool active_valid = false;
};

} // namespace GTSim

#endif // TMA_H
