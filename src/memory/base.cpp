#include "memory.h"
#include "hbm_address.h"

namespace GTSim {

std::unique_ptr<MemoryBackend> make_bank_plane_hbf_backend(
    const HBFConfig&, double, std::uint64_t);

std::unique_ptr<MemoryBackend> make_eventlab_backend(const HBFConfig&, double, std::uint64_t);
std::unique_ptr<MemoryBackend> make_flashlab_backend(const HBFConfig&, double, std::uint64_t);

TensorPlacement make_bank_plane_tensor_placement(const MemorySystemConfig&);

void allocate_contiguous_tensor(TensorRegistry& registry, TensorMetadata tensor) {
    registry.allocate_tensor(
        tensor.id, std::move(tensor.name), std::move(tensor.shape),
        std::move(tensor.strides), tensor.dtype, tensor.initial_memory_tier,
        tensor.preloaded, tensor.read_only, tensor.tile_major);
}

TensorPlacement make_tensor_placement(const MemorySystemConfig& config, MemoryTier tier) {
    switch (tier) {
    case MemoryTier::HBM:
        return allocate_contiguous_tensor;
    case MemoryTier::HBF:
        if (!config.address_layout.enabled ||
            config.topology == MemoryTopology::FullHBM ||
            config.hbf.backend == HBFBackend::Naive)
            throw std::invalid_argument("HBF program tensors require an active HBFine segment");
        if (config.hbf.backend == HBFBackend::BankPlane) return make_bank_plane_tensor_placement(config);
        return [](TensorRegistry& r, TensorMetadata t) {
            if (!t.preloaded || !t.read_only) throw std::invalid_argument("HBF weights require preload and readonly");
            allocate_contiguous_tensor(r, std::move(t));
        };
    }
    throw std::invalid_argument("unknown placement tier");
}

std::unique_ptr<MemoryBackend> make_hbf_backend(const HBFConfig& config,
                                              double frequency, std::uint64_t base) {
    if (config.backend == HBFBackend::EventLab) return make_eventlab_backend(config, frequency, base);
    if (config.backend == HBFBackend::FlashLab) return make_flashlab_backend(config, frequency, base);
    if (config.backend == HBFBackend::Naive) return std::make_unique<HBFModel>(config, frequency);
    if (config.backend == HBFBackend::BankPlane) return make_bank_plane_hbf_backend(config, frequency, base);
    throw std::invalid_argument("unknown HBF backend");
}

} // namespace GTSim
