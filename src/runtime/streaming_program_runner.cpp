#include "runtime/streaming_program_runner.h"

#include "memory/base.h"
#include "kernel_prefetch_session.h"

#include <algorithm>
#include <charconv>
#include <set>

namespace GTSim::ExecutionIR {
namespace {

struct ProgramBeginWire {
    int protocol_version{};
    std::string stream_id;
    std::string symbol;
    int launch_overhead_cycles{};
};
struct TensorDefWire {
    int id{};
    std::string name;
    std::vector<std::uint64_t> shape, strides;
    std::string dtype, tier;
    bool preloaded{}, read_only{};
    std::optional<TileMajorLayout> tile_major;
};
struct LaunchDefWire {
    std::string id, stream_id, signal_token;
    std::vector<std::array<int, 2>> tensor_bindings;
    std::vector<std::string> wait_tokens, read_aliases;
};
struct CommandDefWire {
    std::string kind, target;
    std::vector<std::string> after;
};
struct PrefetchDefWire {
    std::string id;
    int tensor{};
    std::uint64_t offset{}, bytes{};
    std::optional<PrefetchBuffer> buffer;
};
struct KernelBeginWire {
    std::string launch_id, stream_id, signal_token;
    std::vector<std::string> wait_tokens, read_aliases;
    std::vector<std::array<int, 2>> tensor_bindings;
};
struct KernelDoneWire {
    std::string launch_id, stream_id;
    int complete_cycle{}, device_complete_cycle{};
};
struct KernelEndWire {
    std::size_t op_count{}, cta_count{};
    std::vector<int> outputs, partial_outputs;
};

DataType parse_dtype(const std::string& value) {
    if (value == "fp16") return DataType::FP16;
    if (value == "bf16") return DataType::BF16;
    if (value == "fp32") return DataType::FP32;
    if (value == "int32") return DataType::INT32;
    throw std::invalid_argument("unsupported program dtype " + value);
}

MemoryTier parse_tier(const std::string& value) {
    return parse_memory_tier(value);
}

void require_type(const Message& message, std::string_view expected) {
    if (message.type != expected)
        throw std::invalid_argument("expected stream message " +
                                    std::string(expected) + ", got " +
                                    message.type);
}

} // namespace

StreamingProgramRunner::StreamingProgramRunner(SimulatorConfig config,
                                               FramedTransport& transport,
                                               int max_cycles)
    : config_(std::move(config)), transport_(transport), max_cycles_(max_cycles),
      registry_(config_.memory.address_layout) {
    if (max_cycles_ <= 0) throw std::invalid_argument("max_cycles must be positive");
}

void StreamingProgramRunner::send_error(const std::exception& error) noexcept {
    try {
        transport_.send_json(make_typed_json(
            "Error", "{\"code\":\"execution_error\",\"message\":" +
                         json_quote(error.what()) + "}"));
    } catch (...) {
    }
}

void StreamingProgramRunner::receive_program_manifest() {
    const auto first = transport_.receive();
    require_type(first, "ProgramBegin");
    const auto begin = decode_message<ProgramBeginWire>(first);
    if (begin.protocol_version != static_cast<int>(kStreamProtocolVersion) ||
        begin.stream_id.empty() || begin.symbol.empty() ||
        begin.launch_overhead_cycles < 0)
        throw std::invalid_argument("unsupported or invalid ProgramBegin");
    stream_id_ = begin.stream_id;
    symbol_ = begin.symbol;
    launch_overhead_cycles_ = begin.launch_overhead_cycles;

    std::map<MemoryTier, TensorPlacement> placement_policies;
    std::set<std::string> tensor_names;
    while (true) {
        const auto message = transport_.receive();
        if (message.type == "TensorDef") {
            const auto wire = decode_message<TensorDefWire>(message);
            if (wire.id < 0 || wire.name.empty() ||
                !tensor_names.insert(wire.name).second || wire.shape.empty() ||
                wire.shape.size() != wire.strides.size())
                throw std::invalid_argument("invalid stream TensorDef");
            const auto dtype = parse_dtype(wire.dtype);
            const auto tier = parse_tier(wire.tier);
            auto policy = placement_policies.find(tier);
            if (policy == placement_policies.end())
                policy = placement_policies.emplace(
                    tier, make_tensor_placement(config_.memory, tier)).first;
            policy->second(registry_,
                           TensorMetadata{wire.id,
                                          wire.name,
                                          0,
                                          wire.shape,
                                          wire.strides,
                                          dtype,
                                          tier,
                                          wire.preloaded,
                                          wire.read_only,
                                          wire.tile_major});
            const auto& allocated = registry_.at(wire.id);
            timing_.tensor_addresses[wire.tier].emplace(
                wire.name, allocated.global_base_address);
            continue;
        }
        if (message.type == "LaunchDef") {
            const auto wire = decode_message<LaunchDefWire>(message);
            if (wire.id.empty() || wire.stream_id != stream_id_ ||
                wire.signal_token.empty() || launches_.contains(wire.id))
                throw std::invalid_argument("invalid/duplicate stream LaunchDef");
            LaunchDef launch;
            launch.id = wire.id;
            launch.stream_id = wire.stream_id;
            launch.signal_token = wire.signal_token;
            launch.read_aliases = wire.read_aliases;
            launch.wait_tokens = wire.wait_tokens;
            for (const auto& binding : wire.tensor_bindings)
                launch.bindings.emplace_back(binding[0], binding[1]);
            launches_.emplace(launch.id, launch);
            timing_.state_history[launch.id].push_back("CREATED");
            continue;
        }
        if (message.type == "PrefetchDef") {
            const auto wire = decode_message<PrefetchDefWire>(message);
            if (wire.id.empty() || wire.bytes == 0 ||
                !prefetch_regions_.emplace(
                    wire.id,
                    PrefetchDef{wire.id, wire.tensor, wire.offset, wire.bytes,
                                wire.buffer})
                     .second)
                throw std::invalid_argument("invalid/duplicate PrefetchDef");
            (void)registry_.at(wire.tensor);
            continue;
        }
        if (message.type == "CommandDef") {
            const auto wire = decode_message<CommandDefWire>(message);
            if (wire.kind.empty()) throw std::invalid_argument("empty CommandDef");
            commands_.push_back({wire.kind, wire.target, wire.after});
            continue;
        }
        if (message.type == "ProgramReady") {
            const auto ready = decode_message<ProgramBeginWire>(message);
            if (ready.protocol_version != 0 &&
                ready.protocol_version != static_cast<int>(kStreamProtocolVersion))
                throw std::invalid_argument("ProgramReady version mismatch");
            break;
        }
        throw std::invalid_argument("unexpected manifest message " + message.type);
    }
    if (launches_.empty() || commands_.empty())
        throw std::invalid_argument("empty streaming program manifest");
    for (const auto& [id, launch] : launches_) {
        tokens_.declare(launch.signal_token);
        for (const auto& wait : launch.wait_tokens) {
            if (wait.empty()) throw std::invalid_argument("empty wait token");
            try {
                tokens_.add_consumer(wait);
            } catch (const std::logic_error&) {
                tokens_.declare(wait, 1);
            }
        }
        (void)id;
    }
    for (const auto& command : commands_)
        if (command.kind == "LaunchKernel" && !launches_.contains(command.target))
            throw std::invalid_argument("CommandDef references unknown launch");
    simulator_ = std::make_unique<Simulator>(config_, registry_);
    gpu_ = simulator_->simulated_gpu;
    memory_ = gpu_->memory_system.get();
    l2_ = gpu_->l2;
    manifest_ready_ = true;
}

void StreamingProgramRunner::start_prefetch(const CommandDef& command) {
    if (!memory_) throw std::logic_error("prefetch before ProgramReady");
    const auto it = prefetch_regions_.find(command.target);
    if (it == prefetch_regions_.end())
        throw std::invalid_argument("unknown prefetch region " + command.target);
    const auto& region = it->second;
    if (command.kind == "QueuePrefetch") {
        aliases_.emplace(command.target,
                         memory_->enqueue_prefetch(region.tensor, region.offset,
                                                   region.bytes, region.buffer));
        return;
    }
    const auto alias = memory_->reserve_prefetch(
        region.tensor, region.offset, region.bytes, region.buffer);
    if (!alias.token) throw std::runtime_error("prefetch residency capacity exhausted");
    memory_->start_prefetch(alias);
    aliases_.emplace(command.target, alias);
}

void StreamingProgramRunner::synchronize_memory() {
    if (!memory_) return;
    while (!memory_->idle() && simulator_->cycle < max_cycles_)
        memory_->step(++simulator_->cycle);
    if (!memory_->idle()) throw std::runtime_error("stream memory drain timed out");
    timing_.total_cycles = simulator_->cycle;
}

void StreamingProgramRunner::execute_launch(const LaunchDef& launch) {
    if (!simulator_ || simulator_->dag)
        throw std::logic_error("kernel graph overlap in streaming runner");
    if (!tokens_.all_completed(launch.wait_tokens))
        throw std::logic_error("kernel wait token is not completed");
    timing_.state_history.at(launch.id).push_back("QUEUED");
    transport_.send_json(make_typed_json(
        "RequestKernel", "{\"launch_id\":" + json_quote(launch.id) +
                             ",\"stream_id\":" + json_quote(stream_id_) + "}"));

    const auto first = transport_.receive();
    require_type(first, "KernelBegin");
    const auto wire = decode_message<KernelBeginWire>(first);
    bool bindings_match = wire.tensor_bindings.size() == launch.bindings.size();
    if (bindings_match) {
        for (std::size_t i = 0; i < wire.tensor_bindings.size(); ++i) {
            if (wire.tensor_bindings[i][0] != launch.bindings[i].first ||
                wire.tensor_bindings[i][1] != launch.bindings[i].second) {
                bindings_match = false;
                break;
            }
        }
    }
    if (wire.launch_id != launch.id || wire.stream_id != stream_id_ ||
        wire.signal_token != launch.signal_token ||
        wire.wait_tokens != launch.wait_tokens ||
        !bindings_match ||
        wire.read_aliases != launch.read_aliases)
        throw std::invalid_argument("KernelBegin does not match LaunchDef");
    KernelHeader header{wire.launch_id,
                        wire.stream_id,
                        wire.wait_tokens,
                        wire.signal_token,
                        {},
                        wire.read_aliases};
    for (const auto& binding : wire.tensor_bindings)
        header.tensor_bindings.emplace_back(binding[0], binding[1]);
    KernelGraphBuilder builder(registry_, config_);
    builder.begin(std::move(header));
    try {
        while (true) {
            const auto message = transport_.receive();
            if (message.type == "CTA") {
                builder.add_cta(decode_message<CTA>(message));
            } else if (message.type == "Instruction") {
                builder.add_instruction(decode_message<Instruction>(message));
            } else if (message.type == "PrefetchWindow") {
                builder.add_prefetch_window(decode_message<PrefetchWindow>(message));
            } else if (message.type == "KernelEnd") {
                const auto end = decode_message<KernelEndWire>(message);
                KernelEnd contract{end.op_count, end.cta_count, end.outputs,
                                   end.partial_outputs};
                auto built = builder.finish(contract);
                std::vector<MemorySystem::ReadAlias> bindings;
                for (const auto& name : launch.read_aliases) {
                    const auto it = aliases_.find(name);
                    if (it == aliases_.end())
                        throw std::invalid_argument("launch binds inactive read alias");
                    bindings.push_back(it->second);
                }
                memory_->bind_read_aliases(std::move(bindings));

                std::unique_ptr<KernelPrefetchSession> panel;
                if (!built.prefetch_windows.empty()) {
                    panel = std::make_unique<KernelPrefetchSession>(
                        *memory_, built.prefetch_windows, built.dag->nodes.size(),
                        launch.bindings, next_prefetch_epoch_, simulator_->cycle);
                }
                KernelTiming launch_timing;
                launch_timing.id = launch.id;
                launch_timing.state = "RUNNING";
                timing_.state_history.at(launch.id).push_back("RUNNING");
                launch_timing.launch_cycle = simulator_->cycle;
                const auto traffic_before = memory_->tier_counters();
                launch_timing.background_at_launch =
                    memory_->pending_background_requests();
                for (int i = 0; i < launch_overhead_cycles_; ++i) {
                    if (simulator_->cycle >= max_cycles_)
                        throw std::runtime_error("launch overhead timeout");
                    memory_->step(++simulator_->cycle);
                }
                launch_timing.start_cycle = simulator_->cycle;
                launch_timing.l2_lines_before = l2_->resident_line_count();
                const auto background_before = memory_->background_cycles();
                simulator_->attach_graph(*built.dag);
                if (panel)
                    simulator_->node_completion_observer =
                        [&](int node, int now) { panel->complete(node, now); };
                if (!simulator_->run_active_graph(max_cycles_, false))
                    throw std::runtime_error("kernel execution timeout");
                simulator_->node_completion_observer = {};
                if (panel) launch_timing.prefetch_windows = panel->finish();
                launch_timing.l2_lines_after = l2_->resident_line_count();
                launch_timing.device_complete_cycle = simulator_->cycle;
                simulator_->verify_graph_retired();
                KernelExecutionSession::detach(*simulator_);
                built.dag.reset();
                launch_timing.complete_cycle =
                    std::max(simulator_->cycle, simulator_->memory_drain_cycle);
                launch_timing.background_compute_cycles =
                    simulator_->background_cycles_at_device_complete - background_before;
                launch_timing.device_cycles = launch_timing.device_complete_cycle -
                                              launch_timing.start_cycle;
                for (const auto& [tier, after] : memory_->tier_counters()) {
                    const auto& before = traffic_before.at(tier);
                    auto& delta = launch_timing.memory[memory_tier_name(tier)];
                    delta.read_bytes = after.read_bytes - before.read_bytes;
                    delta.write_bytes = after.write_bytes - before.write_bytes;
                    delta.read_completions =
                        after.read_completions - before.read_completions;
                    delta.read_latency_cycles =
                        after.read_latency_cycles - before.read_latency_cycles;
                    delta.unadmitted_alias_read_bytes =
                        after.unadmitted_alias_read_bytes -
                        before.unadmitted_alias_read_bytes;
                    delta.gauges = after.device.gauges;
                    for (const auto& [name, value] : after.device.cumulative)
                        delta.counters.emplace(name,
                                               value - before.device.cumulative.at(name));
                    for (std::size_t unit = 0;
                         unit < after.device.unit_busy_ticks.size(); ++unit) {
                        const auto ticks = after.device.unit_busy_ticks[unit] -
                                           before.device.unit_busy_ticks.at(unit);
                        if (ticks) {
                            delta.active_units.push_back(unit);
                            delta.unit_busy_ticks.emplace(unit, ticks);
                        }
                    }
                }
                launch_timing.background_at_completion =
                    memory_->pending_background_requests();
                launch_timing.state = "COMPLETED";
                timing_.state_history.at(launch.id).push_back("COMPLETED");
                timing_.per_kernel.push_back(launch_timing);
                simulator_->cycle = launch_timing.complete_cycle;
                timing_.total_cycles = simulator_->cycle;
                tokens_.complete(launch.signal_token);
                tokens_.release_if_unreferenced(launch.signal_token);
                for (const auto& wait : launch.wait_tokens) tokens_.consume(wait);
                transport_.send_json(make_typed_json(
                    "KernelDone", "{\"launch_id\":" + json_quote(launch.id) +
                                     ",\"stream_id\":" + json_quote(stream_id_) +
                                     ",\"complete_cycle\":" +
                                     std::to_string(launch_timing.complete_cycle) +
                                     ",\"device_complete_cycle\":" +
                                     std::to_string(launch_timing.device_complete_cycle) +
                                     "}"));
                return;
            } else {
                throw std::invalid_argument("unexpected kernel message " + message.type);
            }
        }
    } catch (...) {
        builder.abort();
        if (simulator_->dag) {
            simulator_->node_completion_observer = {};
            simulator_->detach_graph();
        }
        throw;
    }
}

void StreamingProgramRunner::execute_command(const CommandDef& command) {
    if (command.kind == "LaunchKernel") {
        const auto it = launches_.find(command.target);
        if (it == launches_.end()) throw std::invalid_argument("unknown kernel launch");
        execute_launch(it->second);
    } else if (command.kind == "Prefetch" || command.kind == "QueuePrefetch") {
        start_prefetch(command);
    } else if (command.kind == "Release") {
        const auto it = aliases_.find(command.target);
        if (it == aliases_.end()) throw std::invalid_argument("release inactive alias");
        memory_->release_prefetch(it->second);
        aliases_.erase(it);
        timing_.prefetch_release_cycles.emplace(command.target, simulator_->cycle);
    } else if (command.kind == "StreamSynchronize") {
        if (!command.target.empty() && command.target != stream_id_)
            throw std::invalid_argument("synchronize targets another stream");
        synchronize_memory();
    } else if (command.kind == "RecordEvent") {
        if (command.target.empty() || events_.contains(command.target))
            throw std::invalid_argument("empty/duplicate event");
        events_[command.target] = timing_.total_cycles;
    } else if (command.kind == "WaitEvent") {
        const auto it = events_.find(command.target);
        if (it == events_.end() || it->second > timing_.total_cycles)
            throw std::invalid_argument("wait before record or unknown event");
    } else {
        throw std::invalid_argument("unsupported stream command " + command.kind);
    }
}

ProgramTiming StreamingProgramRunner::run() {
    try {
        receive_program_manifest();
        std::set<std::string> submitted;
        for (const auto& command : commands_) {
            if (command.kind == "LaunchKernel" &&
                !submitted.insert(command.target).second)
                throw std::invalid_argument("duplicate kernel launch command");
            execute_command(command);
        }
        if (submitted.size() != launches_.size())
            throw std::invalid_argument("unsubmitted kernel launch");
        const auto end = transport_.receive();
        require_type(end, "ProgramEnd");
        synchronize_memory();
        timing_.memory_bandwidth = memory_->bandwidth_samples();
        timing_.background_cycles = memory_->background_cycles();
        timing_.prefetch_capacity_stalls = memory_->prefetch_capacity_stalls();
        timing_.prefetch_canceled = memory_->prefetch_canceled();
        for (const auto& [tier, stats] : memory_->tier_counters()) {
            const std::string name = memory_tier_name(tier);
            timing_.traffic_bytes.emplace(name + "_read", stats.read_bytes);
            timing_.traffic_bytes.emplace(name + "_write", stats.write_bytes);
        }
        timing_.persistent_gpu = timing_.persistent_memory = timing_.persistent_l2 =
            timing_.stable_allocations = true;
        transport_.send_json(make_typed_json(
            "ProgramDone", "{\"symbol\":" + json_quote(symbol_) +
                              ",\"total_cycles\":" +
                              std::to_string(timing_.total_cycles) + "}"));
        return timing_;
    } catch (const std::exception& error) {
        send_error(error);
        throw;
    }
}
} // namespace GTSim::ExecutionIR
