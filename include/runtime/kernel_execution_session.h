#pragma once
namespace GTSim {
class Simulator;
class DAG;
// Owns launch-local scheduling/admission lifecycle; never owns the GPU/memory.
class KernelExecutionSession {
public:
    static void initialize(Simulator& simulator, DAG& dag,
                           bool replace_schedulers);
    static void retire(const Simulator& simulator);
    static void rebind(Simulator& simulator, DAG& dag);
};
} // namespace GTSim
