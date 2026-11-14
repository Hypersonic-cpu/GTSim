#pragma once

#include "execution_ir/kernel_graph.h"
#include "simulator.h"

#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <cstdint>
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
    std::vector<int> tensor_ids_;
    std::vector<int> cta_sms_;
    std::vector<int> cta_warps_;
    std::vector<int> warp_bases_;
    std::vector<int> sm_next_subpartition_;
    int next_warp_id_{};
    std::vector<PrefetchWindow> prefetch_windows_;
    std::vector<std::uint8_t> allocation_modes_;
    std::vector<std::uint64_t> output_areas_;
    std::vector<std::uint8_t> prefetch_consumers_;

    void require_started() const;
    static DataType parse_dtype(const std::string& value);
    static const std::unordered_map<std::string_view, std::string_view>&
    supported_ops();
    bool bound_tensor(int local_id) const;
    int global_tensor_id(int local_id) const;
    void clear_bindings() noexcept;
};

} // namespace GTSim::ExecutionIR
