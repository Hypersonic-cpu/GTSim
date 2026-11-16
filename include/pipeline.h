#ifndef PIPELINE_H
#define PIPELINE_H

#include <string>
#include <vector>
#include <deque>
#include <tuple>
#include <utility>

namespace GTSim {

// Pipeline for executing operations (Tensor, SIMD, SFU, LS)
class Pipeline {
public:
    std::string pipeline_name;
    int pipeline_latency;
    int pipeline_throughput;
    int pipeline_width;
    std::deque<std::tuple<int, int, int>>
        pipeline_executing_nodes; // (node_id, subop_index, complete_cycle)
    int pipeline_next_issue_cycle;

    Pipeline(const std::string& name, int latency, int throughput, int width)
        : pipeline_name(name),
          pipeline_latency(latency),
          pipeline_throughput(throughput),
          pipeline_width(width),
          pipeline_next_issue_cycle(0) {}

    // Step function: returns list of completed (node_id, subop_index) pairs
    // Optimized: exploits FIFO property, only checks front of deque
    std::vector<std::pair<int, int>> step(int current_cycle) {
        std::vector<std::pair<int, int>> completed;

        // Pop completed nodes from front (FIFO order guaranteed by fixed latency)
        while (!pipeline_executing_nodes.empty()) {
            const auto& [node_id, subop_index, complete_cycle] =
                pipeline_executing_nodes.front();

            if (current_cycle >= complete_cycle) {
                // Node completed, pop from front
                completed.push_back({node_id, subop_index});
                pipeline_executing_nodes.pop_front();
            } else {
                // Front node not ready, all subsequent nodes also not ready (FIFO)
                break;
            }
        }

        return completed;
    }

    // Check if pipeline can issue new operation
    bool can_issue(int current_cycle) const {
        return current_cycle >= pipeline_next_issue_cycle;
    }

    // Issue a new operation (appends to back, maintains FIFO order)
    void issue_node(int node_id, int subop_index, int current_cycle,
                    int issue_span_override = -1) {
        int issue_span;
        if (issue_span_override > 0) {
            issue_span = issue_span_override;
        } else {
            issue_span =
                (pipeline_throughput + pipeline_width - 1) / pipeline_width;
        }
        if (issue_span < 1) issue_span = 1;

        int start_cycle = current_cycle + issue_span;
        int complete_cycle = start_cycle + pipeline_latency;
        pipeline_executing_nodes.push_back(
            {node_id, subop_index, complete_cycle});
        pipeline_next_issue_cycle = current_cycle + issue_span;
    }
};

} // namespace GTSim

#endif // PIPELINE_H
