#pragma once

#include "execution_ir/kernel_graph_builder.h"
#include "runtime/program_timing.h"
#include "runtime/stream_protocol.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace GTSim::ExecutionIR {

class StreamingProgramRunner {
public:
    StreamingProgramRunner(SimulatorConfig config, FramedTransport& transport,
                           int max_cycles = 100000000);

    ProgramTiming run();

private:
    struct LaunchDef {
        std::string id;
        std::string stream_id;
        std::uint64_t sequence{};
        std::vector<std::pair<int, int>> bindings;
        std::vector<std::string> read_aliases;
    };
    struct CommandDef {
        std::string kind;
        std::string target;
        std::vector<std::string> after;
    };
    struct PrefetchDef {
        std::string id;
        int tensor{};
        std::uint64_t offset{}, bytes{};
        std::optional<PrefetchBuffer> buffer;
    };

    SimulatorConfig config_;
    FramedTransport& transport_;
    int max_cycles_;
    std::string symbol_, stream_id_;
    int launch_overhead_cycles_{};
    TensorRegistry registry_;
    std::unique_ptr<Simulator> simulator_;
    GPU* gpu_{};
    MemorySystem* memory_{};
    L2Cache* l2_{};
    std::map<std::string, LaunchDef> launches_;
    std::vector<CommandDef> commands_;
    std::map<std::string, PrefetchDef> prefetch_regions_;
    std::map<std::string, MemorySystem::ReadAlias> aliases_;
    std::map<std::string, int> events_;
    struct StreamState {
        std::uint64_t completed_launches{};
        std::int64_t last_completion_cycle{};
    } stream_state_;
    ProgramTiming timing_;
    std::uint64_t next_prefetch_epoch_{};
    bool manifest_ready_{};

    void receive_program_manifest();
    void validate_commands() const;
    void execute_command(const CommandDef& command);
    void execute_launch(const LaunchDef& launch);
    void start_prefetch(const CommandDef& command);
    void synchronize_memory();
    void send_error(const std::exception& error) noexcept;
};

} // namespace GTSim::ExecutionIR
