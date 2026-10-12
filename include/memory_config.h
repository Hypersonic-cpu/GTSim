#ifndef MEMORY_CONFIG_H
#define MEMORY_CONFIG_H

#include <cmath>
#include <stdexcept>

namespace GTSim {

enum class MemoryTopology {
    FullHBM,
    FullHBF,
    CoLocated,
    Cascaded
};

struct HBMConfig {
    // read_bw_gbps is the shared data-path ceiling; writes also obey
    // write_max_bw_gbps. Bandwidth units are decimal GB/s.
    double read_bw_gbps{3100.0};
    double write_max_bw_gbps{1550.0};
    int read_latency_cycles{270};
    int write_latency_cycles{270};
    int queue_capacity{65536};
};

struct HBFConfig {
    // Read and write bandwidths use independent service paths, in decimal GB/s.
    double read_bw_gbps{1500.0};
    double write_bw_gbps{100.0};
    int read_latency_cycles{40000};
    int write_latency_cycles{150000};
    int queue_capacity{65536};
};

struct D2DLinkConfig {
    // Bandwidth is decimal GB/s; latency is measured in GPU core cycles.
    double bandwidth_gbps{3100.0};
    int latency_cycles{10};
    int queue_capacity{65536};
};

struct MemorySystemConfig {
    // H100-friendly defaults keep architecture-only invocations usable;
    // topology YAMLs provide the explicit experiment settings.
    MemoryTopology topology{MemoryTopology::FullHBM};
    HBMConfig hbm{};
    HBFConfig hbf{};
    D2DLinkConfig gpu_to_hbm{3100.0, 10, 65536};
    D2DLinkConfig gpu_to_hbf{1500.0, 10, 65536};
    D2DLinkConfig hbm_to_hbf{1500.0, 10, 65536};
};

inline double bytes_per_cycle_from_gbps(double bandwidth_gbps,
                                        double core_frequency_mhz) {
    if (bandwidth_gbps <= 0.0 || core_frequency_mhz <= 0.0) {
        return 0.0;
    }
    return bandwidth_gbps * 1000.0 / core_frequency_mhz;
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
    const auto valid_latency = [](int value) { return value >= 0; };
    const auto valid_capacity = [](int value) { return value > 0; };

    if (!valid_bandwidth(config.hbm.read_bw_gbps) ||
        !valid_bandwidth(config.hbm.write_max_bw_gbps) ||
        !valid_latency(config.hbm.read_latency_cycles) ||
        !valid_latency(config.hbm.write_latency_cycles) ||
        !valid_capacity(config.hbm.queue_capacity)) {
        throw std::invalid_argument("invalid HBM memory configuration");
    }
    if (!valid_bandwidth(config.hbf.read_bw_gbps) ||
        !valid_bandwidth(config.hbf.write_bw_gbps) ||
        !valid_latency(config.hbf.read_latency_cycles) ||
        !valid_latency(config.hbf.write_latency_cycles) ||
        !valid_capacity(config.hbf.queue_capacity)) {
        throw std::invalid_argument("invalid HBF memory configuration");
    }
    const D2DLinkConfig* links[] = {&config.gpu_to_hbm,
                                    &config.gpu_to_hbf,
                                    &config.hbm_to_hbf};
    for (const auto* link : links) {
        if (!valid_bandwidth(link->bandwidth_gbps) ||
            !valid_latency(link->latency_cycles) ||
            !valid_capacity(link->queue_capacity)) {
            throw std::invalid_argument("invalid D2D link configuration");
        }
    }
}

} // namespace GTSim

#endif // MEMORY_CONFIG_H
