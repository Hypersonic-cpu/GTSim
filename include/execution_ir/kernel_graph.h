#pragma once
#include "simulator.h"
#include <memory>

namespace GTSim::ExecutionIR {
struct CTA {
    int id{}, sm{}, warps{}, shared_bytes{}, register_bytes{};
};
struct Instruction {
    std::string name, op, pipeline, dtype;
    int id{}, cta{}, warp{}, tensor{}, leading_dim{};
    std::array<int, 4> tile{};
    std::vector<int> after_issue, after_complete;
    int pipeline_iteration{-1};
};
struct PrefetchRange {
    int tensor{};
    std::uint64_t offset{}, bytes{};
};
struct PrefetchWindow {
    PrefetchBuffer buffer;
    std::vector<int> consumers;
    std::vector<PrefetchRange> ranges;
};
struct LookaheadRegion {
    std::vector<int> spans;
};
struct LookaheadConsumer {
    int node{};
    std::vector<int> regions, spans;
};
struct LookaheadPlan {
    int distance{};
    std::vector<PrefetchRange> spans;
    std::vector<LookaheadRegion> regions;
    std::vector<LookaheadConsumer> consumers;
};
} // namespace GTSim::ExecutionIR
