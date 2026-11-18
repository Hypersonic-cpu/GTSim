#pragma once

#include "execution_ir/kernel_graph.h"
#include "runtime/program_timing.h"

#include <algorithm>

namespace GTSim::ExecutionIR {

// Logical regions share disjoint physical spans. Each span is reserved once,
// and retained until all actual memory instructions using it have completed.
class LookaheadPrefetchSession {
public:
    LookaheadPrefetchSession(MemorySystem& memory, const LookaheadPlan& plan,
                             const DAG& dag,
                             const std::vector<std::pair<int, int>>& bindings,
                             int cycle)
        : memory_(memory), plan_(plan), consumers_(dag.nodes.size(), -1),
          spans_(plan.spans.size()), traces_(plan.regions.size()),
          aliases_(std::make_shared<std::vector<MemorySystem::ReadAlias>>()) {
        if (plan.distance <= 0 || plan.regions.empty() || plan.spans.empty())
            throw std::invalid_argument("empty or invalid prefetch lookahead");
        std::vector<int> global_ids(bindings.size(), -1);
        for (const auto& [local, global] : bindings)
            global_ids.at(local) = global;
        aliases_->reserve(plan.spans.size());
        for (const auto& span : plan.spans) {
            auto alias = memory_.prefetch_range(global_ids.at(span.tensor),
                                                span.offset, span.bytes);
            if (alias.tier != MemoryTier::HBF || !span.bytes ||
                (!aliases_->empty() && aliases_->back().address + aliases_->back().bytes >
                                       alias.address))
                throw std::invalid_argument("invalid or overlapping lookahead spans");
            aliases_->push_back(alias);
        }
        for (std::size_t i = 0; i < plan.regions.size(); ++i) {
            if (!valid_ids(plan.regions[i].spans, plan.spans.size()))
                throw std::invalid_argument("invalid lookahead region span set");
            for (const auto index : plan.regions[i].spans)
                traces_[i].bytes += plan.spans[index].bytes;
        }
        std::vector<int> nodes;
        nodes.reserve(plan.consumers.size());
        std::vector<bool> used_regions(plan.regions.size(), false);
        for (std::size_t i = 0; i < plan.consumers.size(); ++i) {
            const auto& consumer = plan.consumers[i];
            auto& entry = consumers_.at(consumer.node);
            if (entry >= 0 || dag.nodes.at(consumer.node)->op_type !=
                                  OpType::CP_DRAM2SRAM_TMA ||
                !valid_ids(consumer.spans, plan.spans.size()) ||
                !valid_ids(consumer.regions, plan.regions.size()))
                throw std::invalid_argument("invalid lookahead consumer");
            entry = i;
            nodes.push_back(consumer.node);
            for (const auto index : consumer.spans)
                ++spans_[index].remaining;
            for (const auto index : consumer.regions)
                used_regions[index] = true;
        }
        if (std::any_of(spans_.begin(), spans_.end(),
                        [](const auto& span) { return !span.remaining; }) ||
            std::find(used_regions.begin(), used_regions.end(), false) != used_regions.end())
            throw std::invalid_argument("lookahead span or region has no consumer");
        memory_.bind_node_read_aliases(nodes, aliases_);
        for (int i = 0; i < plan.distance && i < int(traces_.size()); ++i)
            enqueue(i, cycle, global_ids);
        global_ids_ = std::move(global_ids);
    }

    void issue(int node, int cycle) {
        const auto index = consumers_.at(node);
        if (index < 0) return;
        for (const auto id : plan_.consumers[index].regions) {
            auto& trace = traces_[id];
            if (trace.first_use_cycle >= 0) continue;
            trace.first_use_cycle = cycle;
            const auto future = id + plan_.distance;
            if (future < int(traces_.size()))
                enqueue(future, cycle, global_ids_);
        }
    }

    void complete(int node, int cycle) {
        const auto index = consumers_.at(node);
        if (index < 0) return;
        const auto& consumer = plan_.consumers[index];
        for (const auto region : consumer.regions)
            traces_[region].last_consumer_complete_cycle = cycle;
        for (const auto id : consumer.spans) {
            auto& span = spans_[id];
            if (--span.remaining) continue;
            auto& alias = aliases_->at(id);
            span.ready_cycle = memory_.prefetch_ready_cycle(alias);
            memory_.release_prefetch(alias, false);
            alias.queue_id = 0;
        }
    }

    std::vector<PrefetchRegionTiming> finish() {
        for (const auto& span : spans_)
            if (span.remaining) throw std::logic_error("unretired lookahead span");
        for (std::size_t i = 0; i < traces_.size(); ++i) {
            auto& trace = traces_[i];
            int ready = -1;
            for (const auto id : plan_.regions[i].spans) {
                if (spans_[id].ready_cycle < 0) {
                    ready = -1;
                    break;
                }
                ready = std::max(ready, spans_[id].ready_cycle);
            }
            trace.ready_cycle = ready;
            trace.ready_at_first_use = ready >= 0 && ready <= trace.first_use_cycle;
        }
        return std::move(traces_);
    }

private:
    struct Span {
        std::size_t remaining{};
        int ready_cycle{-1};
    };

    static bool valid_ids(const std::vector<int>& ids, std::size_t count) {
        return !ids.empty() && ids.front() >= 0 && std::size_t(ids.back()) < count &&
               std::adjacent_find(ids.begin(), ids.end(), std::greater_equal<int>()) == ids.end();
    }

    void enqueue(int region, int cycle, const std::vector<int>& ids) {
        auto& trace = traces_[region];
        if (trace.enqueue_cycle >= 0) return;
        trace.enqueue_cycle = cycle;
        for (const auto id : plan_.regions[region].spans) {
            auto& alias = aliases_->at(id);
            if (!spans_[id].remaining || alias.queue_id) continue;
            const auto& range = plan_.spans[id];
            alias = memory_.enqueue_prefetch(ids.at(range.tensor), range.offset,
                                             range.bytes);
            // Preflight bounds the entire set by the configured queue capacity.
            if (!alias.queue_id)
                throw std::runtime_error("lookahead prefetch descriptor queue exceeded preflight bound");
        }
    }

    MemorySystem& memory_;
    const LookaheadPlan& plan_;
    std::vector<int> consumers_;
    std::vector<Span> spans_;
    std::vector<PrefetchRegionTiming> traces_;
    std::shared_ptr<std::vector<MemorySystem::ReadAlias>> aliases_;
    std::vector<int> global_ids_;
};

} // namespace GTSim::ExecutionIR
