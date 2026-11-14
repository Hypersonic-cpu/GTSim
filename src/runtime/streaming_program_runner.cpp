#include "runtime/streaming_program_runner.h"

#include "memory/base.h"
#include "kernel_prefetch_session.h"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <experimental/scope>
#include <set>
#include <unordered_set>

namespace GTSim::ExecutionIR {

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
    std::string id, stream_id;
    std::uint64_t sequence{};
    std::vector<std::array<int, 2>> tensor_bindings;
    std::vector<std::string> read_aliases;
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
    std::string launch_id, stream_id;
    std::uint64_t sequence{};
    std::vector<std::string> read_aliases;
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
struct InstructionBatchWire {
    std::vector<Instruction> instructions;
};

namespace {

DataType parse_dtype(const std::string& value) {
    if (value == "fp16") return DataType::FP16;
    if (value == "bf16") return DataType::BF16;
    if (value == "fp32") return DataType::FP32;
    if (value == "int32") return DataType::INT32;
    throw std::invalid_argument("unsupported program dtype " + value);
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
    : config_(std::move(config)),
      transport_(transport),
      max_cycles_(max_cycles),
      registry_(config_.memory.address_layout) {
    if (max_cycles_ <= 0)
        throw std::invalid_argument("max_cycles must be positive");
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
                wire.shape.size() != 2 || wire.strides.size() != 2 ||
                !wire.shape[0] || !wire.shape[1] || wire.strides[1] != 1 ||
                wire.strides[0] != wire.shape[1])
                throw std::invalid_argument("invalid stream TensorDef");
            const auto dtype = parse_dtype(wire.dtype);
            const auto tier = parse_memory_tier(wire.tier);
            auto policy = placement_policies.find(tier);
            if (policy == placement_policies.end())
                policy =
                    placement_policies
                        .emplace(tier,
                                 make_tensor_placement(config_.memory, tier))
                        .first;
            policy->second(registry_,
                           TensorMetadata{wire.id, wire.name, 0, wire.shape,
                                          wire.strides, dtype, tier,
                                          wire.preloaded, wire.read_only,
                                          wire.tile_major});
            const auto& allocated = registry_.at(wire.id);
            timing_.tensor_addresses[wire.tier].emplace(
                wire.name, allocated.global_base_address);
            continue;
        }
        if (message.type == "LaunchDef") {
            const auto wire = decode_message<LaunchDefWire>(message);
            if (wire.id.empty() || wire.stream_id != stream_id_ ||
                launches_.contains(wire.id))
                throw std::invalid_argument(
                    "invalid/duplicate stream LaunchDef");
            LaunchDef launch;
            launch.id = wire.id;
            launch.stream_id = wire.stream_id;
            launch.sequence = wire.sequence;
            launch.read_aliases = wire.read_aliases;
            for (const auto& binding : wire.tensor_bindings)
                launch.bindings.emplace_back(binding[0], binding[1]);
            launches_.emplace(launch.id, launch);
            timing_.state_history[launch.id].push_back("CREATED");
            continue;
        }
        if (message.type == "PrefetchDef") {
            const auto wire = decode_message<PrefetchDefWire>(message);
            if (wire.id.empty() || wire.bytes == 0 ||
                !prefetch_regions_
                     .emplace(wire.id,
                              PrefetchDef{wire.id, wire.tensor, wire.offset,
                                          wire.bytes, wire.buffer})
                     .second)
                throw std::invalid_argument("invalid/duplicate PrefetchDef");
            const auto& tensor = registry_.at(wire.tensor);
            if (wire.offset > tensor.byte_size() ||
                wire.bytes > tensor.byte_size() - wire.offset)
                throw std::invalid_argument("prefetch region exceeds tensor");
            continue;
        }
        if (message.type == "CommandDef") {
            const auto wire = decode_message<CommandDefWire>(message);
            if (wire.kind.empty())
                throw std::invalid_argument("empty CommandDef");
            commands_.push_back({wire.kind, wire.target, wire.after});
            continue;
        }
        if (message.type == "ProgramReady") {
            const auto ready = decode_message<ProgramBeginWire>(message);
            if (ready.protocol_version != 0 &&
                ready.protocol_version !=
                    static_cast<int>(kStreamProtocolVersion))
                throw std::invalid_argument("ProgramReady version mismatch");
            break;
        }
        throw std::invalid_argument("unexpected manifest message " +
                                    message.type);
    }
    if (launches_.empty() || commands_.empty())
        throw std::invalid_argument("empty streaming program manifest");
    std::unordered_set<std::uint64_t> sequences;
    for (const auto& [id, launch] : launches_) {
        if (!sequences.insert(launch.sequence).second)
            throw std::invalid_argument("duplicate launch sequence");
        (void)id;
    }
    validate_commands();
    simulator_ = std::make_unique<Simulator>(config_, registry_);
    auto* gpu = simulator_->simulated_gpu;
    memory_ = gpu->memory_system.get();
    l2_ = gpu->l2;
}

void StreamingProgramRunner::validate_commands() const {
    std::unordered_set<std::string> submitted, active_aliases,
        started_prefetches, recorded;
    bool has_launch = false;
    for (const auto& command : commands_) {
        if (command.kind == "LaunchKernel") {
            if (!launches_.contains(command.target) ||
                !submitted.insert(command.target).second)
                throw std::invalid_argument("unknown/duplicate kernel launch");
            const auto& launch = launches_.at(command.target);
            for (const auto& alias : launch.read_aliases)
                if (!active_aliases.contains(alias))
                    throw std::invalid_argument(
                        "launch binds an inactive read alias");
            has_launch = true;
        } else if (command.kind == "Prefetch" ||
                   command.kind == "QueuePrefetch") {
            if (!prefetch_regions_.contains(command.target) ||
                !started_prefetches.insert(command.target).second)
                throw std::invalid_argument(
                    "unknown or repeated prefetch region");
            active_aliases.insert(command.target);
        } else if (command.kind == "Release") {
            if (!has_launch || !active_aliases.erase(command.target))
                throw std::invalid_argument(
                    "release requires an active prefetch and an initialized "
                    "stream");
        } else if (command.kind == "RecordEvent") {
            if (command.target.empty() ||
                !recorded.insert(command.target).second)
                throw std::invalid_argument("empty/duplicate event");
        } else if (command.kind == "WaitEvent") {
            if (!recorded.contains(command.target))
                throw std::invalid_argument(
                    "wait before record or unknown event");
        } else if (command.kind == "StreamSynchronize") {
            if (!command.target.empty() && command.target != stream_id_)
                throw std::invalid_argument(
                    "synchronize targets another stream");
            if (command.target.empty() && !has_launch &&
                !active_aliases.empty())
                throw std::invalid_argument(
                    "prefetch synchronization requires an initialized stream");
        } else {
            throw std::invalid_argument("unsupported stream command");
        }
    }
    if (submitted.size() != launches_.size())
        throw std::invalid_argument("unsubmitted kernel launch");
}

void StreamingProgramRunner::start_prefetch(const CommandDef& command) {
    assert(memory_);
    const auto& region = prefetch_regions_.at(command.target);
    if (command.kind == "QueuePrefetch") {
        const auto inserted = aliases_.emplace(
            command.target,
            memory_->enqueue_prefetch(region.tensor, region.offset,
                                      region.bytes, region.buffer)).second;
        assert(inserted);
        (void)inserted;
        return;
    }
    const auto alias = memory_->reserve_prefetch(region.tensor, region.offset,
                                                 region.bytes, region.buffer);
    if (!alias.token)
        throw std::runtime_error("prefetch residency capacity exhausted");
    memory_->start_prefetch(alias);
    const auto inserted = aliases_.emplace(command.target, alias).second;
    assert(inserted);
    (void)inserted;
}

void StreamingProgramRunner::synchronize_memory() {
    if (!memory_) return;
    while (!memory_->idle() && simulator_->cycle < max_cycles_)
        memory_->step(++simulator_->cycle);
    if (!memory_->idle())
        throw std::runtime_error("stream memory drain timed out");
    timing_.total_cycles = simulator_->cycle;
}

void StreamingProgramRunner::execute_launch(const LaunchDef& launch) {
    assert(simulator_ && !simulator_->dag);
    assert(launch.sequence == stream_state_.completed_launches);
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
        wire.sequence != launch.sequence || !bindings_match ||
        wire.read_aliases != launch.read_aliases)
        throw std::invalid_argument("KernelBegin does not match LaunchDef");
    KernelHeader header{
        wire.launch_id, wire.stream_id, wire.sequence, {}, wire.read_aliases};
    for (const auto& binding : wire.tensor_bindings)
        header.tensor_bindings.emplace_back(binding[0], binding[1]);
    KernelGraphBuilder builder(registry_, config_);
    builder.begin(std::move(header));
    [[maybe_unused]] auto cleanup = std::experimental::scope_exit([&] {
        builder.abort();
        simulator_->node_completion_observer = {};
        if (simulator_->dag) simulator_->detach_graph();
    });
    while (true) {
        const auto message = transport_.receive();
        if (message.type == "CTA") {
            builder.add_cta(decode_message<CTA>(message));
        } else if (message.type == "InstructionBatch") {
            const auto batch = decode_message<InstructionBatchWire>(message);
            assert(!batch.instructions.empty() &&
                   batch.instructions.size() <= 512);
            for (const auto& instruction : batch.instructions)
                builder.add_instruction(instruction);
        } else if (message.type == "PrefetchWindow") {
            builder.add_prefetch_window(
                decode_message<PrefetchWindow>(message));
        } else if (message.type == "KernelEnd") {
            const auto end = decode_message<KernelEndWire>(message);
            KernelEnd contract{end.op_count, end.cta_count, end.outputs,
                               end.partial_outputs};
            auto built = builder.finish(contract);
            std::vector<MemorySystem::ReadAlias> bindings;
            for (const auto& name : launch.read_aliases) {
                bindings.push_back(aliases_.at(name));
            }
            memory_->bind_read_aliases(std::move(bindings));
            memory_->prepare_node_read_aliases(built.dag->nodes.size());

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
                simulator_->node_completion_observer = [&](int node, int now) {
                    panel->complete(node, now);
                };
            if (!simulator_->run_active_graph(max_cycles_, false))
                throw std::runtime_error("kernel execution timeout");
            simulator_->node_completion_observer = {};
            if (panel) launch_timing.prefetch_windows = panel->finish();
            launch_timing.l2_lines_after = l2_->resident_line_count();
            launch_timing.device_complete_cycle = simulator_->cycle;
            simulator_->verify_retired();
            KernelExecutionSession::detach(*simulator_);
            built.dag.reset();
            launch_timing.complete_cycle =
                std::max(simulator_->cycle, simulator_->memory_drain_cycle);
            launch_timing.background_compute_cycles =
                simulator_->background_cycles_at_device_complete -
                background_before;
            launch_timing.device_cycles =
                launch_timing.device_complete_cycle - launch_timing.start_cycle;
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
                    delta.counters.emplace(
                        name, value - before.device.cumulative.at(name));
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
            ++stream_state_.completed_launches;
            stream_state_.last_completion_cycle = launch_timing.complete_cycle;
            transport_.send_json(make_typed_json(
                "KernelDone",
                "{\"launch_id\":" + json_quote(launch.id) + ",\"stream_id\":" +
                    json_quote(stream_id_) + ",\"complete_cycle\":" +
                    std::to_string(launch_timing.complete_cycle) +
                    ",\"device_complete_cycle\":" +
                    std::to_string(launch_timing.device_complete_cycle) + "}"));
            return;
        } else {
            throw std::invalid_argument("unexpected kernel message " +
                                        message.type);
        }
    }
}

void StreamingProgramRunner::execute_command(const CommandDef& command) {
    if (command.kind == "LaunchKernel") {
        execute_launch(launches_.at(command.target));
    } else if (command.kind == "Prefetch" || command.kind == "QueuePrefetch") {
        start_prefetch(command);
    } else if (command.kind == "Release") {
        const auto alias = aliases_.at(command.target);
        memory_->release_prefetch(alias);
        aliases_.erase(command.target);
        timing_.prefetch_release_cycles.emplace(command.target,
                                                simulator_->cycle);
    } else if (command.kind == "StreamSynchronize") {
        synchronize_memory();
    } else if (command.kind == "RecordEvent") {
        const auto inserted =
            events_.emplace(command.target, timing_.total_cycles).second;
        assert(inserted);
        (void)inserted;
        timing_.events[command.target] = timing_.total_cycles;
    } else if (command.kind == "WaitEvent") {
        const auto event_cycle = events_.at(command.target);
        assert(event_cycle <= timing_.total_cycles);
        (void)event_cycle;
    } else {
        assert(false && "unsupported stream command");
    }
}

ProgramTiming StreamingProgramRunner::run() {
    try {
        receive_program_manifest();
        for (const auto& command : commands_) execute_command(command);
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
        timing_.persistent_gpu = timing_.persistent_memory =
            timing_.persistent_l2 = timing_.stable_allocations = true;
        transport_.send_json(make_typed_json(
            "ProgramDone", "{\"symbol\":" + json_quote(symbol_) +
                               ",\"total_cycles\":" +
                               std::to_string(timing_.total_cycles) + "}"));
        return timing_;
    } catch (const std::exception& error) {
        try {
            transport_.send_json(make_typed_json(
                "Error", "{\"code\":\"execution_error\",\"message\":" +
                             json_quote(error.what()) + "}"));
        } catch (...) {
        }
        throw;
    }
}
} // namespace GTSim::ExecutionIR
