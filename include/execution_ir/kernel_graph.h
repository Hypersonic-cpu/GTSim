#pragma once
#include "simulator.h"
#include <filesystem>
#include <memory>
#include <map>

namespace GTSim::ExecutionIR {
struct Tensor {
    std::string record, name, dtype, tier;
    int id{};
    std::vector<std::uint64_t> shape, strides;
    bool preloaded{}, read_only{};
    std::optional<TileMajorLayout> tile_major{};
};
struct CTA {
    std::string record;
    int id{}, sm{}, warps{}, shared_bytes{}, register_bytes{};
};
struct Instruction {
    std::string record, name, op, pipeline, dtype;
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
struct Kernel {
    std::string record, format, symbol, source_hash, target,
        sync_semantics_version;
    int schema_version{};
    std::vector<Tensor> tensors;
    std::vector<CTA> ctas;
    std::vector<Instruction> ops;
    std::vector<int> partial_outputs;
    std::vector<int> outputs;
    std::optional<std::vector<PrefetchWindow>> prefetch_windows;
};
Kernel read_kernel(const std::filesystem::path& path);
std::unique_ptr<DAG>
materialize(const Kernel& kernel, const SimulatorConfig& config,
            const TensorRegistry& registry,
            const std::vector<std::pair<int, int>>& bindings);
} // namespace GTSim::ExecutionIR
