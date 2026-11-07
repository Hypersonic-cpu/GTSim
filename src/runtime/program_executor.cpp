#include "runtime/execution_program.h"
#include <set>
#include "kernel_prefetch_session.h"

namespace GTSim::ExecutionIR {
static DataType parse_dtype(const std::string& value) {
    if (value == "fp16") return DataType::FP16;
    if (value == "bf16") return DataType::BF16;
    if (value == "fp32") return DataType::FP32;
    if (value == "int32") return DataType::INT32;
    throw std::invalid_argument("unsupported program dtype " + value);
}

ProgramTiming execute(const Program& p, const std::filesystem::path& directory,
                      const SimulatorConfig& config, int max_cycles) {
    if (p.streams.size() != 1 || p.streams.front().empty() ||
        p.launch_overhead_cycles < 0)
        throw std::invalid_argument("exactly one ordered stream and "
                                    "nonnegative launch overhead required");
    TensorRegistry registry(config.memory.address_layout);
    std::map<MemoryTier, TensorPlacement> placement_policies;
    ProgramTiming timing;
    std::uint64_t next_prefetch_epoch{};
    std::set<std::string> names;
    std::map<int, std::tuple<std::uint64_t, std::vector<std::uint64_t>,
                             std::vector<std::uint64_t>, DataType>>
        allocations;
    for (const auto& t : p.tensors) {
        if (t.id < 0 || t.name.empty() || !names.insert(t.name).second ||
            t.shape.size() != 2 ||
            t.strides.size() != 2 || !t.shape[0] || !t.shape[1] ||
            t.strides[1] != 1 || t.strides[0] != t.shape[1])
            throw std::invalid_argument("invalid program tensor");
        const auto dtype = parse_dtype(t.dtype);
        const auto tier = parse_memory_tier(t.tier);
        auto policy = placement_policies.find(tier);
        if (policy == placement_policies.end())
            policy = placement_policies.emplace(
                tier, make_tensor_placement(config.memory, tier)).first;
        policy->second(registry,
                       TensorMetadata{t.id, t.name, 0, t.shape, t.strides,
                                      dtype, tier, t.preloaded, t.read_only,
                                      t.tile_major});
        const auto& allocated = registry.at(t.id);
        timing.tensor_addresses[t.tier].emplace(t.name, allocated.global_base_address);
        allocations.emplace(t.id, std::tuple{allocated.global_base_address,
                                             allocated.shape, allocated.strides,
                                             allocated.dtype});
    }

    std::map<std::string, const Launch*> launches;
    for (const auto& l : p.launches) {
        if (l.id.empty() || launches.contains(l.id))
            throw std::invalid_argument("duplicate/empty launch identity");
        launches[l.id] = &l;
        timing.state_history[l.id].push_back("CREATED");
    }
    std::map<std::string, PrefetchRegion> regions;
    if (p.prefetches)
        for (const auto& region : *p.prefetches)
            if (region.id.empty() || !regions.emplace(region.id, region).second)
                throw std::invalid_argument("duplicate/empty prefetch region ID");
    std::set<std::string> active_aliases, started_prefetches;
    bool has_launch = false;
    std::set<std::string> submitted, recorded;
    for (const auto& c : p.commands) {
        if (c.kind == "LaunchKernel") {
            if (!launches.contains(c.target) ||
                !submitted.insert(c.target).second)
                throw std::invalid_argument("unknown/duplicate kernel launch");
            const auto& launch = *launches.at(c.target);
            if (launch.read_aliases)
                for (const auto& alias : *launch.read_aliases)
                    if (!active_aliases.contains(alias))
                        throw std::invalid_argument("launch binds an inactive read alias");
            timing.state_history[c.target].push_back("QUEUED");
            has_launch = true;
        } else if ((c.kind == "Prefetch" || c.kind == "QueuePrefetch")) {
            if (!regions.contains(c.target) || !started_prefetches.insert(c.target).second)
                throw std::invalid_argument("unknown or repeated prefetch region");
            active_aliases.insert(c.target);
        } else if (c.kind == "Release") {
            if (!has_launch || !active_aliases.erase(c.target))
                throw std::invalid_argument("release requires an active prefetch and an initialized stream");
        } else if (c.kind == "RecordEvent") {
            if (c.target.empty() || !recorded.insert(c.target).second)
                throw std::invalid_argument("empty/duplicate event");
        } else if (c.kind == "WaitEvent") {
            if (!recorded.contains(c.target))
                throw std::invalid_argument(
                    "wait before record or unknown event");
        } else if (c.kind == "StreamSynchronize" && c.target.empty()) {
            if (!has_launch && !active_aliases.empty())
                throw std::invalid_argument("prefetch synchronization requires an initialized stream");
        } else {
            throw std::invalid_argument("unsupported stream command");
        }
    }
    if (submitted.size() != launches.size())
        throw std::invalid_argument("unsubmitted kernel launch");

    std::unique_ptr<DAG> current_dag;
    std::unique_ptr<DAG> registry_owner;
    DAG* active_dag = nullptr;
    std::unique_ptr<Simulator> context;
    GPU* gpu = nullptr;
    MemorySystem* memory = nullptr;
    L2Cache* l2 = nullptr;
    std::map<std::string, MemorySystem::ReadAlias> aliases;
    std::vector<Command> initial_prefetches;
    const auto prefetch = [&](const Command& command) {
        const auto& region = regions.at(command.target);
        if (command.kind == "QueuePrefetch") {
            aliases.emplace(command.target, memory->enqueue_prefetch(
                                                region.tensor, region.offset,
                                                region.bytes, region.buffer));
            return;
        }
        const auto alias = memory->reserve_prefetch(
            region.tensor, region.offset, region.bytes, region.buffer);
        if (!alias.token) throw std::runtime_error("prefetch residency capacity exhausted");
        memory->start_prefetch(alias);
        aliases.emplace(command.target, alias);
    };
    const auto synchronize = [&] {
        if (!context) return;
        while (!memory->idle() && context->cycle < max_cycles)
            memory->step(++context->cycle);
        if (!memory->idle()) throw std::runtime_error("stream memory drain timed out");
        timing.total_cycles = context->cycle;
    };
    for (const auto& c : p.commands) {
        if (c.kind == "LaunchKernel") {
            const auto& specification = *launches.at(c.target);
            auto kernel = read_kernel(directory / specification.kernel_file);
            std::vector<std::pair<int, int>> bindings;
            for (const auto& b : specification.bindings)
                bindings.emplace_back(b[0], b[1]);
            auto next_dag = materialize(kernel, config, registry, bindings);
            if (!context) {
                registry_owner = std::move(next_dag);
                active_dag = registry_owner.get();
                context = std::make_unique<Simulator>(active_dag, config);
                gpu = context->simulated_gpu;
                memory = gpu->memory_system.get();
                l2 = gpu->l2;
                for (const auto& id : initial_prefetches) prefetch(id);
                initial_prefetches.clear();
            } else {
                // Rebind on the existing GPU. The prior DAG remains alive until
                // KernelExecutionSession has retired it and installed this one.
                context->bind_kernel(next_dag.get());
                if (active_dag == registry_owner.get())
                    registry_owner->clear_nodes();
                else
                    current_dag.reset();
                current_dag = std::move(next_dag);
                active_dag = current_dag.get();
            }

            std::vector<MemorySystem::ReadAlias> bindings_for_launch;
            if (specification.read_aliases)
                for (const auto& alias : *specification.read_aliases)
                    bindings_for_launch.push_back(aliases.at(alias));
            memory->bind_read_aliases(std::move(bindings_for_launch));
            std::unique_ptr<KernelPrefetchSession> panel;
            if (kernel.prefetch_windows && !kernel.prefetch_windows->empty()) {
                panel = std::make_unique<KernelPrefetchSession>(*memory, kernel, bindings,
                                                               next_prefetch_epoch, context->cycle);
                context->node_completion_observer = [&](int node, int now) { panel->complete(node, now); };
            }
            KernelTiming launch;
            launch.background_at_launch = memory->pending_background_requests();
            launch.id = c.target;
            if (timing.state_history.at(c.target).back() != "QUEUED")
                throw std::logic_error("invalid launch transition");
            launch.state = "RUNNING";
            timing.state_history[c.target].push_back("RUNNING");
            launch.launch_cycle = context->cycle;
            const auto traffic_before = memory->tier_counters();
            for (int i = 0; i < p.launch_overhead_cycles; ++i) {
                if (context->cycle >= max_cycles)
                    throw std::runtime_error("launch overhead timeout");
                context->simulated_gpu->memory_system->step(++context->cycle);
            }
            // Launch overhead advances the stream clock but admits no CTA work.
            launch.start_cycle = context->cycle;
            launch.l2_lines_before = l2->resident_line_count();
            const auto background_before = memory->background_cycles();
            if (!context->run(max_cycles, false))
                throw std::runtime_error("kernel execution timeout");
            context->node_completion_observer = {};
            if (panel) launch.prefetch_windows = panel->finish();
            launch.l2_lines_after = l2->resident_line_count();
            launch.device_complete_cycle = context->cycle;
            KernelExecutionSession::retire(*context);
            launch.complete_cycle =
                std::max(context->cycle, context->memory_drain_cycle);
            launch.background_compute_cycles =
                context->background_cycles_at_device_complete - background_before;
            launch.device_cycles =
                launch.device_complete_cycle - launch.start_cycle;
            for (const auto& [tier, after] : memory->tier_counters()) {
                const auto& before = traffic_before.at(tier);
                auto& delta = launch.memory[memory_tier_name(tier)];
                delta.read_bytes = after.read_bytes - before.read_bytes;
                delta.write_bytes = after.write_bytes - before.write_bytes;
                delta.read_completions = after.read_completions - before.read_completions;
                delta.read_latency_cycles = after.read_latency_cycles - before.read_latency_cycles;
                delta.unadmitted_alias_read_bytes =
                    after.unadmitted_alias_read_bytes - before.unadmitted_alias_read_bytes;
                delta.gauges = after.device.gauges;
                for (const auto& [name, value] : after.device.cumulative)
                    delta.counters.emplace(name, value - before.device.cumulative.at(name));
                for (std::size_t unit = 0; unit < after.device.unit_busy_ticks.size(); ++unit)
                    if (const auto ticks =
                            after.device.unit_busy_ticks[unit] -
                            before.device.unit_busy_ticks.at(unit)) {
                        delta.active_units.push_back(unit);
                        delta.unit_busy_ticks.emplace(unit, ticks);
                    }
            }
            launch.background_at_completion = memory->pending_background_requests();
            launch.state = "COMPLETED";
            timing.state_history[c.target].push_back("COMPLETED");
            timing.per_kernel.push_back(launch);
            context->cycle = launch.complete_cycle;
            timing.total_cycles = launch.complete_cycle;
            if (context->simulated_gpu != gpu ||
                gpu->memory_system.get() != memory || gpu->l2 != l2)
                throw std::logic_error("GPU context was recreated");
        } else if ((c.kind == "Prefetch" || c.kind == "QueuePrefetch")) {
            if (memory) prefetch(c);
            else initial_prefetches.push_back(c);
        } else if (c.kind == "Release") {
            memory->release_prefetch(aliases.at(c.target));
            aliases.erase(c.target);
            timing.prefetch_release_cycles.emplace(c.target, context->cycle);
        } else if (c.kind == "StreamSynchronize") {
            synchronize();
        } else if (c.kind == "RecordEvent")
            timing.events[c.target] = timing.total_cycles;
        else if (c.kind == "WaitEvent" &&
                 timing.events.at(c.target) > timing.total_cycles)
            throw std::logic_error("event completion clock is in the future");
    }
    if (!initial_prefetches.empty())
        throw std::invalid_argument("prefetch program requires a kernel launch");
    if (context) {
        synchronize();
        timing.memory_bandwidth = memory->bandwidth_samples();
        timing.background_cycles = memory->background_cycles();
        timing.prefetch_capacity_stalls = memory->prefetch_capacity_stalls();
        timing.prefetch_canceled = memory->prefetch_canceled();
        for (const auto& [tier, stats] : memory->tier_counters()) {
            const std::string name = memory_tier_name(tier);
            timing.traffic_bytes.emplace(name + "_read", stats.read_bytes);
            timing.traffic_bytes.emplace(name + "_write", stats.write_bytes);
        }
        for (const auto& [id, expected] : allocations) {
            const auto& actual = registry.at(id);
            if (expected != std::tuple{actual.global_base_address, actual.shape,
                                       actual.strides, actual.dtype})
                throw std::logic_error(
                    "program allocation changed across launches");
        }
    }
    // Simulator owns GPU references into its current DAG, so destroy it first.
    context.reset();
    current_dag.reset();
    registry_owner.reset();
    timing.persistent_gpu = timing.persistent_memory = timing.persistent_l2 =
        timing.stable_allocations = true;
    return timing;
}
} // namespace GTSim::ExecutionIR
