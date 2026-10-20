#ifndef MEM_CONF_H
#define MEM_CONF_H

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace GTSim {

enum class MemoryTopology { FullHBM, FullHBF, CoLocated, Cascaded };

struct HBMConfig {
    double read_bw_GBps{3100.0};
    double write_max_bw_GBps{1550.0};
    int read_latency_cycles{270};
    int write_latency_cycles{270};
    int queue_capacity{65536};
    std::uint64_t capacity_bytes{}; // Zero means unspecified by this abstract profile.
};

struct HBFConfig {
    double read_bw_GBps{524.288};
    double write_bw_GBps{27.96};
    int read_latency_cycles{7920};
    int write_latency_cycles{148500};
    int queue_capacity{65536};
    std::uint64_t capacity_bytes{};
};

struct D2DLinkConfig {
    double bandwidth_GBps{3100.0};
    int latency_cycles{10};
    int queue_capacity{65536};
};

struct MemorySystemConfig {
    MemoryTopology topology{MemoryTopology::FullHBM};
    HBMConfig hbm{};
    HBFConfig hbf{};
    D2DLinkConfig gpu_to_hbm{3100.0, 10, 65536};
    D2DLinkConfig gpu_to_hbf{1500.0, 10, 65536};
    D2DLinkConfig hbm_to_hbf{1500.0, 10, 65536};
};

inline double bytes_per_cycle_from_GBps(double bandwidth_GBps,
                                        double core_frequency_mhz) {
    if (bandwidth_GBps <= 0.0 || core_frequency_mhz <= 0.0) {
        return 0.0;
    }
    return bandwidth_GBps * 1000.0 / core_frequency_mhz;
}

inline const char* memory_topology_name(MemoryTopology topology) {
    switch (topology) {
    case MemoryTopology::FullHBM: return "FullHBM";
    case MemoryTopology::FullHBF: return "FullHBF";
    case MemoryTopology::CoLocated: return "CoLocated";
    case MemoryTopology::Cascaded: return "Cascaded";
    }
    return "Unknown";
}

inline void validate_memory_system_config(const MemorySystemConfig& config) {
    const auto valid_bandwidth = [](double value) {
        return std::isfinite(value) && value > 0.0;
    };
    const auto valid_tier_bandwidth = [](double value, bool active) {
        return std::isfinite(value) && (active ? value > 0.0 : value >= 0.0);
    };
    const auto valid_latency = [](int value) { return value >= 0; };
    const auto valid_capacity = [](int value) { return value > 0; };

    const bool hbm_active = config.topology != MemoryTopology::FullHBF;
    const bool hbf_active = config.topology != MemoryTopology::FullHBM;
    if (!valid_tier_bandwidth(config.hbm.read_bw_GBps, hbm_active) ||
        !valid_tier_bandwidth(config.hbm.write_max_bw_GBps, hbm_active) ||
        !valid_latency(config.hbm.read_latency_cycles) ||
        !valid_latency(config.hbm.write_latency_cycles) ||
        !valid_capacity(config.hbm.queue_capacity)) {
        throw std::invalid_argument("invalid HBM memory configuration");
    }
    if (!valid_tier_bandwidth(config.hbf.read_bw_GBps, hbf_active) ||
        !valid_tier_bandwidth(config.hbf.write_bw_GBps, hbf_active) ||
        !valid_latency(config.hbf.read_latency_cycles) ||
        !valid_latency(config.hbf.write_latency_cycles) ||
        !valid_capacity(config.hbf.queue_capacity)) {
        throw std::invalid_argument("invalid HBF memory configuration");
    }
    const D2DLinkConfig* links[] = {&config.gpu_to_hbm, &config.gpu_to_hbf,
                                    &config.hbm_to_hbf};
    for (const auto* link : links) {
        if (!valid_bandwidth(link->bandwidth_GBps) ||
            !valid_latency(link->latency_cycles) ||
            !valid_capacity(link->queue_capacity)) {
            throw std::invalid_argument("invalid D2D link configuration");
        }
    }
}

} // namespace GTSim

#endif // MEM_CONF_H
