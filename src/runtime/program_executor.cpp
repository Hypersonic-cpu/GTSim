#include "runtime/execution_program.h"
#include <set>

namespace GTSim::ExecutionIR {
ProgramTiming execute(const Program& p, const std::filesystem::path& directory,
                      const SimulatorConfig& config, int max_cycles) {
    if (p.streams.size() != 1 || p.streams.front().empty() ||
        p.launch_overhead_cycles < 0)
        throw std::invalid_argument("exactly one ordered stream and "
                                    "nonnegative launch overhead required");
    TensorRegistry registry(config.memory.address_layout);
    std::set<std::string> names;
    for (const auto& t : p.tensors) {
        if (t.id < 0 || t.name.empty() || !names.insert(t.name).second ||
            t.dtype != "fp16" || t.tier != "HBM" || t.shape.size() != 2 ||
            t.strides.size() != 2 || !t.shape[0] || !t.shape[1] ||
            t.strides[1] != 1 || t.strides[0] != t.shape[1])
            throw std::invalid_argument("invalid program tensor");
        registry.allocate_tensor(t.id, t.name, t.shape, t.strides,
                                 DataType::FP16, MemoryTier::HBM, t.preloaded,
                                 t.read_only);
    }
    ProgramTiming timing;
    std::map<std::string, std::size_t> launches;
    std::vector<std::unique_ptr<DAG>> dags;
    for (const auto& l : p.launches) {
        if (l.id.empty() || launches.contains(l.id))
            throw std::invalid_argument("duplicate/empty launch identity");
        launches[l.id] = dags.size();
        timing.state_history[l.id].push_back("CREATED");
        auto kernel = read_kernel(directory / l.kernel_file);
        std::vector<std::pair<int, int>> bindings;
        for (const auto& b : l.bindings) bindings.emplace_back(b[0], b[1]);
        dags.push_back(materialize(kernel, config, registry, bindings));
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
    std::unique_ptr<Simulator> context;
    GPU* gpu = nullptr;
    MemorySystem* memory = nullptr;
    L2Cache* l2 = nullptr;
    for (const auto& c : p.commands) {
        if (c.kind == "LaunchKernel") {
            auto* dag = dags.at(launches.at(c.target)).get();
            if (!context) {
                context = std::make_unique<Simulator>(dag, config);
                gpu = context->simulated_gpu;
                memory = gpu->memory_system.get();
                l2 = gpu->l2;
            } else
                context->bind_kernel(dag);
            KernelTiming launch;
            launch.id = c.target;
            if (timing.state_history.at(c.target).back() != "QUEUED")
                throw std::logic_error("invalid launch transition");
            launch.state = "RUNNING";
            timing.state_history[c.target].push_back("RUNNING");
            launch.launch_cycle = context->cycle;
            for (int i = 0; i < p.launch_overhead_cycles; ++i) {
                if (context->cycle >= max_cycles)
                    throw std::runtime_error("launch overhead timeout");
                context->simulated_gpu->memory_system->step(++context->cycle);
            }
            // No nodes may run during launch overhead. It is accounted below
            // by advancing idle memory time before admission, not scheduling.
            launch.start_cycle = context->cycle;
            launch.l2_lines_before = l2->resident_line_count();
            const auto bytes_before =
                memory->statistics(context->cycle).hbm.read_bytes;
            if (!context->run(max_cycles, false))
                throw std::runtime_error("kernel execution timeout");
            launch.l2_lines_after = l2->resident_line_count();
            launch.hbm_read_bytes =
                memory->statistics(context->memory_drain_cycle).hbm.read_bytes -
                bytes_before;
            launch.device_complete_cycle = context->cycle;
            launch.complete_cycle =
                std::max(context->cycle, context->memory_drain_cycle);
            launch.device_cycles = context->cycle - launch.start_cycle;
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
        for (const auto& dag : dags)
            for (const auto& [id, tensor] : registry.entries()) {
                const auto& bound = dag->tensor_registry.at(id);
                if (bound.global_base_address != tensor.global_base_address ||
                    bound.shape != tensor.shape)
                    throw std::logic_error(
                        "program allocation changed across launches");
            }
    }
    timing.persistent_gpu = timing.persistent_memory = timing.persistent_l2 =
        timing.stable_allocations = true;
    return timing;
}
} // namespace GTSim::ExecutionIR
