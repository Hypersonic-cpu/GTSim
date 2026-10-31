#pragma once
#include "execution_ir/kernel_graph.h"

namespace GTSim::ExecutionIR {
struct Launch {
    std::string record, id, kernel_file;
    std::vector<std::array<int, 2>> bindings;
};
struct Command {
    std::string record, kind, target;
};
struct Program {
    std::string record, format, symbol, source_hash;
    int schema_version{}, launch_overhead_cycles{};
    std::vector<std::string> streams;
    std::vector<Tensor> tensors;
    std::vector<Launch> launches;
    std::vector<Command> commands;
};
Program read_program(const std::filesystem::path& path);
struct KernelTiming {
    std::string id, state;
    int launch_cycle{}, start_cycle{}, device_complete_cycle{},
        complete_cycle{}, device_cycles{};
    std::size_t l2_lines_before{}, l2_lines_after{};
    std::uint64_t hbm_read_bytes{};
};
struct ProgramTiming {
    int total_cycles{};
    std::vector<KernelTiming> per_kernel;
    std::map<std::string, int> events;
    bool persistent_gpu{}, persistent_memory{}, persistent_l2{},
        stable_allocations{};
};
ProgramTiming execute(const Program& program,
                      const std::filesystem::path& directory,
                      const SimulatorConfig& config,
                      int max_cycles = 100000000);
} // namespace GTSim::ExecutionIR
