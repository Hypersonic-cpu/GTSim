#pragma once
namespace GTSim {
class Simulator;
class DAG;
// Owns launch-local scheduling/admission lifecycle; never owns the GPU/memory.
class KernelExecutionSession {
public:
    static void attach(Simulator& simulator, DAG& dag);
    static void verify_retired(const Simulator& simulator);
    static void detach(Simulator& simulator);
};
} // namespace GTSim
