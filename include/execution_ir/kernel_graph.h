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
} // namespace GTSim::ExecutionIR
