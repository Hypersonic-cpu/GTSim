#pragma once
#include "execution_ir/kernel_graph.h"
#include <optional>

namespace GTSim::ExecutionIR {
struct Launch {
    std::string record, id, kernel_file;
    std::vector<std::array<int, 2>> bindings;
    std::optional<std::vector<std::string>> read_aliases;
};
struct Command {
    std::string record, kind, target;
};
struct PrefetchRegion {
    std::string id;
    int tensor{};
    std::uint64_t offset{}, bytes{};
};
struct Program {
    std::string record, format, symbol, source_hash;
    int schema_version{}, launch_overhead_cycles{};
    std::vector<std::string> streams;
    std::vector<Tensor> tensors;
    std::vector<Launch> launches;
    std::vector<Command> commands;
    std::optional<std::vector<PrefetchRegion>> prefetches;
};
Program read_program(const std::filesystem::path& path);
struct TierTiming {
    std::uint64_t read_bytes{}, write_bytes{};
    std::map<std::string, std::uint64_t> counters;
    std::vector<std::size_t> active_units;
};
struct KernelTiming {
    std::string id, state;
    int launch_cycle{}, start_cycle{}, device_complete_cycle{},
        complete_cycle{}, device_cycles{};
    std::size_t l2_lines_before{}, l2_lines_after{};
    std::map<std::string, TierTiming> memory;
    std::size_t background_at_launch{}, background_at_completion{};
};
struct ProgramTiming {
    int total_cycles{};
    std::vector<KernelTiming> per_kernel;
    std::map<std::string, int> events;
    std::map<std::string, std::vector<std::string>> state_history;
    std::map<std::string, std::uint64_t> traffic_bytes;
    std::map<std::string, std::map<std::string, std::uint64_t>> tensor_addresses;
    bool persistent_gpu{}, persistent_memory{}, persistent_l2{},
        stable_allocations{};
};
ProgramTiming execute(const Program& program,
                      const std::filesystem::path& directory,
                      const SimulatorConfig& config,
                      int max_cycles = 100000000);
} // namespace GTSim::ExecutionIR
