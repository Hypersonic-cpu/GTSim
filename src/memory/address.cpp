#include "memory/base.h"
#include "hbf_placement.h"
namespace GTSim {
TensorPlacement make_hbf_tensor_placement(const MemorySystemConfig& config) {
    std::optional<HBFMegapageAllocator> placement;
    const auto& controller = config.hbf.controller;
    if (config.hbf.align_tensors_to_command_group &&
        controller.geometry == HBFGeometryProfile::GroupedPlanes)
        placement.emplace(config.address_layout.hbf_base, controller);
    return [placement = std::move(placement)](TensorRegistry& registry,
                                              TensorMetadata tensor) mutable {
        if (!tensor.preloaded || !tensor.read_only)
            throw std::invalid_argument("HBF program tensors require preloaded read-only data");
        if (placement) {
            tensor.global_base_address = placement->allocate(tensor.byte_size()).base;
            registry.register_tensor(std::move(tensor));
        } else {
            allocate_contiguous_tensor(registry, std::move(tensor));
        }
    };
}

}
