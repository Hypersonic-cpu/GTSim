#include "memory/system.h"
namespace GTSim {
MemorySystem::MemorySystem(int cache_size_bytes, int line_size_bytes, int hit_latency_cycles,
                 int l2_bandwidth_bytes_per_cycle,
                 int l2_write_bandwidth_bytes_per_cycle, int l2_queue_depth,
                 bool bypass_cache, const MemorySystemConfig& memory_config,
                 double core_frequency_mhz, const TensorRegistry& tensors)
        : memory_config_(memory_config),
          tensors_(tensors),
          l2_(std::make_unique<L2Cache>(cache_size_bytes, line_size_bytes,
                                        hit_latency_cycles,
                                        l2_bandwidth_bytes_per_cycle,
                                        l2_write_bandwidth_bytes_per_cycle,
                                        l2_queue_depth, bypass_cache, tensors)) {
        validate_memory_system_config(memory_config_);
        if (memory_config_.address_layout.enabled &&
            tensors_.address_layout() != memory_config_.address_layout) {
            throw std::invalid_argument("memory devices and TensorRegistry require identical explicit address layouts");
        }
        if (!std::isfinite(core_frequency_mhz) || core_frequency_mhz <= 0.0) {
            throw std::invalid_argument("core frequency must be positive");
        }
        switch (memory_config_.topology) {
            case MemoryTopology::FullHBM: build_full_hbm(core_frequency_mhz); break;
            case MemoryTopology::FullHBF: build_full_hbf(core_frequency_mhz); break;
            case MemoryTopology::CoLocated: build_co_located(core_frequency_mhz); break;
            case MemoryTopology::Cascaded: build_cascaded(core_frequency_mhz); break;
            default: throw std::invalid_argument("unsupported memory topology");
        }
        if (tensors_.address_layout().enabled) {
            if ((memory_config_.topology == MemoryTopology::FullHBM && tensors_.contains_tier(MemoryTier::HBF)) ||
                (memory_config_.topology == MemoryTopology::FullHBF && tensors_.contains_tier(MemoryTier::HBM)))
                throw std::invalid_argument("segmented allocations require a compatible explicit placement profile");
            for (const auto& [id, tensor] : tensors_.entries()) {
                (void)id;
                if (hbf_ && tensor.initial_memory_tier == MemoryTier::HBF && tensor.preloaded) {
                    const auto bytes = (tensor.byte_size() + 4095) / 4096 * 4096;
                    hbf_->preload_region(tensor.global_base_address, bytes, tensor.read_only);
                }
            }
        }
        report_placement_override();
    }
}
