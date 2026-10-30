#ifndef GTSIM_WORKLOAD_PLACEMENT_H
#define GTSIM_WORKLOAD_PLACEMENT_H

#include "simulator.h"

namespace GTSim {

struct WorkloadMemoryOptions {
    TensorPlacementProfile placement{TensorPlacementProfile::AllHBM};
    MemoryAddressLayout layout = [] {
        MemoryAddressLayout value;
        value.enabled = true;
        value.hbf_capacity_bytes = 0;
        return value;
    }();
};

inline WorkloadMemoryOptions workload_memory_options(const SimulatorConfig& config) {
    WorkloadMemoryOptions options;
    options.placement = config.memory.placement;
    if (config.memory.address_layout.enabled) options.layout = config.memory.address_layout;
    if (options.placement != TensorPlacementProfile::AllHBM &&
        (!config.memory.address_layout.enabled || config.memory.topology != MemoryTopology::CoLocated)) {
        throw std::invalid_argument("mixed placement requires explicit CoLocated device segments");
    }
    return options;
}

inline void print_workload_memory(const TensorRegistry& tensors, TensorPlacementProfile placement) {
    const auto& layout = tensors.address_layout();
    std::cout << "Placement profile: " << (placement == TensorPlacementProfile::AllHBM ? "all_hbm" :
        placement == TensorPlacementProfile::GEMMBInHBF ? "gemm_b_hbf" : "fa3_q_hbf (synthetic per-invocation input)") << '\n';
    for (auto tier : {MemoryTier::HBM, MemoryTier::HBF}) {
        std::cout << (tier == MemoryTier::HBM ? "HBM" : "HBF") << " segment: [" << layout.base(tier)
                  << ", " << layout.base(tier) + layout.capacity(tier) << "); reserved bytes: "
                  << tensors.reserved_bytes(tier) << '\n';
    }
    for (const auto& [id, tensor] : tensors.entries()) {
        std::cout << "Tensor " << id << " " << tensor.name << " base " << tensor.global_base_address
                  << " bytes " << tensor.byte_size() << " tier "
                  << (tensor.initial_memory_tier == MemoryTier::HBM ? "HBM" : "HBF")
                  << " preloaded " << tensor.preloaded << " read_only " << tensor.read_only << '\n';
    }
}

} // namespace GTSim
#endif
