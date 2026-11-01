#include "runtime/execution_program.h"
#include <set>

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
    std::set<std::string> names;
    std::map<int, std::tuple<std::uint64_t, std::vector<std::uint64_t>,
                             std::vector<std::uint64_t>, DataType>>
        allocations;
    for (const auto& t : p.tensors) {
        if (t.id < 0 || t.name.empty() || !names.insert(t.name).second ||
            t.tier != "HBM" || t.shape.size() != 2 || t.strides.size() != 2 ||
            !t.shape[0] || !t.shape[1] || t.strides[1] != 1 ||
            t.strides[0] != t.shape[1])
            throw std::invalid_argument("invalid program tensor");
        const auto dtype = parse_dtype(t.dtype);
        const auto& allocated =
            registry.allocate_tensor(t.id, t.name, t.shape, t.strides, dtype,
                                     MemoryTier::HBM, t.preloaded, t.read_only);
        allocations.emplace(t.id, std::tuple{allocated.global_base_address,
                                             allocated.shape, allocated.strides,
                                             allocated.dtype});
    }

    ProgramTiming timing;
    std::map<std::string, const Launch*> launches;
    for (const auto& l : p.launches) {
        if (l.id.empty() || launches.contains(l.id))
            throw std::invalid_argument("duplicate/empty launch identity");
        launches[l.id] = &l;
        timing.state_history[l.id].push_back("CREATED");
    }
    std::set<std::string> submitted, recorded;
    for (const auto& c : p.commands) {
        if (c.kind == "LaunchKernel") {
            if (!launches.contains(c.target) ||
                !submitted.insert(c.target).second)
                throw std::invalid_argument("unknown/duplicate kernel launch");
            timing.state_history[c.target].push_back("QUEUED");
        } else if (c.kind == "RecordEvent") {
            if (c.target.empty() || !recorded.insert(c.target).second)
                throw std::invalid_argument("empty/duplicate event");
        } else if (c.kind == "WaitEvent") {
            if (!recorded.contains(c.target))
                throw std::invalid_argument(
                    "wait before record or unknown event");
        } else if (c.kind != "StreamSynchronize" || !c.target.empty())
            throw std::invalid_argument("unsupported stream command");
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

            KernelTiming launch;
            launch.id = c.target;
            if (timing.state_history.at(c.target).back() != "QUEUED")
                throw std::logic_error("invalid launch transition");
            launch.state = "RUNNING";
            timing.state_history[c.target].push_back("RUNNING");
            launch.launch_cycle = context->cycle;
            const auto traffic_before = memory->statistics(context->cycle);
            for (int i = 0; i < p.launch_overhead_cycles; ++i) {
                if (context->cycle >= max_cycles)
                    throw std::runtime_error("launch overhead timeout");
                context->simulated_gpu->memory_system->step(++context->cycle);
            }
            // Launch overhead advances the stream clock but admits no CTA work.
            launch.start_cycle = context->cycle;
            launch.l2_lines_before = l2->resident_line_count();
            if (!context->run(max_cycles, false))
                throw std::runtime_error("kernel execution timeout");
            launch.l2_lines_after = l2->resident_line_count();
            launch.device_complete_cycle = context->cycle;
            KernelExecutionSession::retire(*context);
            launch.complete_cycle =
                std::max(context->cycle, context->memory_drain_cycle);
            launch.device_cycles =
                launch.device_complete_cycle - launch.start_cycle;
            const auto traffic_after =
                memory->statistics(context->memory_drain_cycle);
            launch.hbm_read_bytes =
                traffic_after.hbm.read_bytes - traffic_before.hbm.read_bytes;
            launch.hbm_write_bytes =
                traffic_after.hbm.write_bytes - traffic_before.hbm.write_bytes;
            launch.hbf_read_bytes =
                traffic_after.hbf.read_bytes - traffic_before.hbf.read_bytes;
            launch.hbf_write_bytes =
                traffic_after.hbf.write_bytes - traffic_before.hbf.write_bytes;
            launch.state = "COMPLETED";
            timing.state_history[c.target].push_back("COMPLETED");
            timing.per_kernel.push_back(launch);
            context->cycle = launch.complete_cycle;
            timing.total_cycles = launch.complete_cycle;
            if (context->simulated_gpu != gpu ||
                gpu->memory_system.get() != memory || gpu->l2 != l2)
                throw std::logic_error("GPU context was recreated");
        } else if (c.kind == "RecordEvent")
            timing.events[c.target] = timing.total_cycles;
        else if (c.kind == "WaitEvent" &&
                 timing.events.at(c.target) > timing.total_cycles)
            throw std::logic_error("event completion clock is in the future");
    }
    if (context) {
        const auto stats = memory->statistics(timing.total_cycles);
        timing.traffic_bytes = {{"HBM_read", stats.hbm.read_bytes},
                                {"HBM_write", stats.hbm.write_bytes},
                                {"HBF_read", stats.hbf.read_bytes},
                                {"HBF_write", stats.hbf.write_bytes}};
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
