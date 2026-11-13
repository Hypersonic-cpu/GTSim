#pragma once

#include "execution_ir/kernel_graph.h"
#include "simulator.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace GTSim::ExecutionIR {

struct KernelHeader {
    std::string launch_id;
    std::string stream_id;
    std::uint64_t sequence{};
    std::vector<std::pair<int, int>> tensor_bindings;
    std::vector<std::string> read_aliases;
};

struct KernelEnd {
    std::size_t op_count{};
    std::size_t cta_count{};
    std::vector<int> outputs;
    std::vector<int> partial_outputs;
};

struct BuiltKernel {
    KernelHeader header;
    std::unique_ptr<DAG> dag;
    std::vector<PrefetchWindow> prefetch_windows;
};

// Receives one Kernel at a time and materializes each Instruction directly
// into its DAG.  The builder is deliberately not reusable concurrently and
// owns no Program-level TensorRegistry.
class KernelGraphBuilder {
public:
    KernelGraphBuilder(const TensorRegistry& global_tensors,
                       const SimulatorConfig& config);

    void begin(KernelHeader header);
    void add_cta(const CTA& cta);
    void add_instruction(const Instruction& op);
    void add_prefetch_window(PrefetchWindow window);
    BuiltKernel finish(const KernelEnd& end);
    void abort() noexcept;

private:
    const TensorRegistry& global_tensors_;
    const SimulatorConfig& config_;
    KernelHeader header_;
    std::unique_ptr<DAG> dag_;
    std::map<int, int> tensor_ids_;
    std::map<int, int> cta_sms_;
    std::map<int, int> cta_warps_;
    std::map<int, int> warp_bases_;
    std::vector<int> sm_next_subpartition_;
    int next_warp_id_{};
    std::vector<PrefetchWindow> prefetch_windows_;
    std::set<int> read_allocations_, write_allocations_;
    std::map<int, std::uint64_t> output_areas_;
    std::set<int> prefetch_consumers_;

    void require_started() const;
    static DataType parse_dtype(const std::string& value);
    static const std::map<std::string, std::string>& supported_ops();
};

} // namespace GTSim::ExecutionIR
