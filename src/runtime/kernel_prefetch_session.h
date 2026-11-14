#pragma once

#include "execution_ir/kernel_graph.h"
#include "runtime/program_timing.h"

#include <cassert>

namespace GTSim::ExecutionIR {
// Kernel-local memory windows follow actual copy completion, independently of
// operation names, workload type, or numerical tensor values.
class KernelPrefetchSession {
public:
    KernelPrefetchSession(MemorySystem& memory,
                          const std::vector<PrefetchWindow>& plans,
                          std::size_t op_count,
                          const std::vector<std::pair<int, int>>& bindings,
                          std::uint64_t& next_epoch, int cycle)
        : memory_(memory),
          owners_(op_count, -1) {
        initialize(
            plans, op_count, bindings, [](int) { return true; }, next_epoch,
            cycle);
    }

    KernelPrefetchSession(const KernelPrefetchSession&) = delete;

private:
    template <class ConsumerPredicate>
    void initialize(const std::vector<PrefetchWindow>& plans,
                    std::size_t op_count,
                    const std::vector<std::pair<int, int>>& bindings,
                    ConsumerPredicate consumer_predicate,
                    std::uint64_t& next_epoch, int cycle) {
        (void)op_count;
#ifdef NDEBUG
        (void)consumer_predicate;
#endif
        std::vector<int> ids(bindings.size(), -1);
        for (const auto& [local_id, registry_id] : bindings) {
            assert(local_id >= 0 &&
                   static_cast<std::size_t>(local_id) < ids.size());
            assert(ids[static_cast<std::size_t>(local_id)] < 0);
            ids[static_cast<std::size_t>(local_id)] = registry_id;
        }
#ifndef NDEBUG
        for (const int id : ids) assert(id >= 0);
#endif
        if (plans.empty())
            throw std::invalid_argument("empty streaming window plan");
        count_ = plans.front().buffer.count;
        if (count_ < 2)
            throw std::invalid_argument(
                "streaming prefetch needs multiple buffers");
        const auto base = (next_epoch + count_ - 1) / count_ * count_;
        next_epoch = base + plans.size();
        for (std::size_t i = 0; i < plans.size(); ++i) {
            const auto& plan = plans[i];
            if (plan.buffer.count != count_ ||
                plan.buffer.slot != static_cast<int>(i % count_) ||
                plan.consumers.empty())
                throw std::invalid_argument("invalid streaming window");
            Window window;
            window.buffer = {plan.buffer.slot, count_, base + i};
            window.consumers = plan.consumers;
            window.remaining = plan.consumers.size();
            std::vector<std::pair<PrefetchRange, MemorySystem::ReadAlias>>
                ranges;
            for (auto range : plan.ranges) {
                assert(range.tensor >= 0 &&
                       static_cast<std::size_t>(range.tensor) < ids.size());
                range.tensor = ids[static_cast<std::size_t>(range.tensor)];
                ranges.push_back(
                    {range, memory_.prefetch_range(range.tensor, range.offset,
                                                   range.bytes)});
            }
            std::sort(ranges.begin(), ranges.end(),
                      [](const auto& a, const auto& b) {
                          return a.second.address < b.second.address;
                      });
            window.aliases =
                std::make_shared<std::vector<MemorySystem::ReadAlias>>();
            std::uint64_t bytes = 0;
            for (const auto& [range, alias] : ranges) {
                window.ranges.push_back(range);
                window.aliases->push_back(alias);
                bytes += range.bytes;
            }
            for (int node : window.consumers) {
                assert(node >= 0 && static_cast<std::size_t>(node) < owners_.size() &&
                       owners_[node] == -1 && consumer_predicate(node));
                owners_[node] = i;
            }
            memory_.bind_node_read_aliases(window.consumers, window.aliases);
            windows_.push_back(std::move(window));
            traces_.push_back({plan.buffer.slot, base + i, bytes});
        }
        for (int i = 0;
             i < count_ && static_cast<std::size_t>(i) < windows_.size(); ++i)
            advance(i, cycle);
    }

public:
    void complete(int node, int cycle) {
        const int index = owners_.at(node);
        if (index < 0) return;
        auto& window = windows_[index];
        if (--window.remaining) return;
        traces_[index].consume_cycle = cycle;
        memory_.unbind_node_read_aliases(window.consumers);
        if (window.active) {
            for (const auto& alias : *window.aliases)
                memory_.release_prefetch(alias, false);
            window.active = false;
            traces_[index].release_request_cycle = cycle;
            advance(index + count_, cycle);
        }
    }

    std::vector<PrefetchWindowTiming> finish() {
        for (const auto& window : windows_)
            if (window.remaining || window.active)
                throw std::logic_error("unretired prefetch window");
        return std::move(traces_);
    }

private:
    struct Window {
        PrefetchBuffer buffer;
        std::vector<int> consumers;
        std::vector<PrefetchRange> ranges;
        std::shared_ptr<std::vector<MemorySystem::ReadAlias>> aliases;
        std::size_t remaining{};
        bool active{};
    };
    void advance(std::size_t index, int cycle) {
        while (index < windows_.size()) {
            auto& window = windows_[index];
            if (!window.remaining) {
                traces_[index].skipped = true;
                index += count_;
                continue;
            }
            for (std::size_t r = 0; r < window.ranges.size(); ++r) {
                const auto& range = window.ranges[r];
                window.aliases->at(r) = memory_.enqueue_prefetch(
                    range.tensor, range.offset, range.bytes, window.buffer);
            }
            window.active = true;
            traces_[index].enqueue_cycle = cycle;
            return;
        }
    }
    MemorySystem& memory_;
    int count_{};
    std::vector<int> owners_;
    std::vector<Window> windows_;
    std::vector<PrefetchWindowTiming> traces_;
};
} // namespace GTSim::ExecutionIR
