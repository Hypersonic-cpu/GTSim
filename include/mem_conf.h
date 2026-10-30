#ifndef MEM_CONF_H
#define MEM_CONF_H

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <string>
#include "memory_layout.h"
#include <hbfine/config.hpp>

namespace GTSim {

enum class MemoryTopology { FullHBM, FullHBF, CoLocated, Cascaded };
enum class HBMBackend { Naive, Ramulator2 };
enum class TensorPlacementProfile { AllHBM, GEMMBInHBF, FA3QInHBF };
enum class HBFBackend { Naive, BankPlane };
using HBFGeometryProfile = hbfine::GeometryProfile;
using HBFControllerConfig = hbfine::Config;

struct RamulatorHBMConfig {
    std::string config_file;
    int stacks{5};
    int logical_mc_groups_per_stack{2};
    int channels_per_stack{16};
    int pseudochannels_per_channel{2};
    int interleave_bytes{128};
};

struct HBMConfig {
    double read_bw_GBps{3100.0};
    double write_max_bw_GBps{1550.0};
    int read_latency_cycles{270};
    int write_latency_cycles{270};
    int queue_capacity{65536};
    std::uint64_t capacity_bytes{}; // Zero means unspecified by this abstract profile.
    HBMBackend backend{HBMBackend::Naive};
    RamulatorHBMConfig ramulator{};
    bool model_channels{false};
    int stacks{5};
    int channels_per_stack{16};
    std::uint64_t capacity_bytes_per_stack{16ULL << 30};
};

struct HBFConfig {
    double read_bw_GBps{524.288};
    double write_bw_GBps{27.96};
    int read_latency_cycles{7920};
    int write_latency_cycles{148500};
    int queue_capacity{65536};
    std::uint64_t capacity_bytes{};
    HBFBackend backend{HBFBackend::Naive};
    HBFControllerConfig controller{};
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
    MemoryAddressLayout address_layout{};
    TensorPlacementProfile placement{TensorPlacementProfile::AllHBM};
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

inline void validate_memory_system_config(const MemorySystemConfig& config, double core_frequency_mhz = 0) {
    config.address_layout.validate();
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
    if (config.hbm.backend == HBMBackend::Ramulator2) {
        const auto& ram = config.hbm.ramulator;
        if ((config.topology != MemoryTopology::FullHBM && config.topology != MemoryTopology::CoLocated) ||
            ram.config_file.empty() || ram.stacks < 1 || ram.stacks > 5 || ram.logical_mc_groups_per_stack != 2 ||
            ram.channels_per_stack != 16 || ram.pseudochannels_per_channel != 2 ||
            ram.interleave_bytes != 128) {
            throw std::invalid_argument(
                "Ramulator2 requires FullHBM/CoLocated and 1-5 H100 HBM3 stacks with 128B interleave");
        }
    }
    if (config.hbm.model_channels && (config.hbm.stacks < 1 || config.hbm.stacks > 5 ||
                                       config.hbm.channels_per_stack != 16)) {
        throw std::invalid_argument("channelized HBM requires 1-5 stacks and 16 channels per stack");
    }
    if (config.address_layout.enabled && (config.hbm.model_channels || config.hbm.backend == HBMBackend::Ramulator2) &&
        (config.address_layout.hbm_base != 0 ||
         config.address_layout.hbm_capacity_bytes != static_cast<std::uint64_t>(
             config.hbm.backend == HBMBackend::Ramulator2 ? config.hbm.ramulator.stacks : config.hbm.stacks) * config.hbm.capacity_bytes_per_stack)) {
        throw std::invalid_argument("HBM physical channels require a zero-based segment matching stack capacity");
    }
    if (config.hbf.backend == HBFBackend::BankPlane) {
        const auto& c = config.hbf.controller;
        if (!hbf_active) throw std::invalid_argument("detailed HBF requires an active HBF device");
        hbfine::validate_config(c);
        if (core_frequency_mhz > 0 &&
            (config.hbf.read_latency_cycles != std::llround(c.sense_latency_us * core_frequency_mhz) ||
             config.hbf.write_latency_cycles != std::llround(c.program_latency_us * core_frequency_mhz))) {
            throw std::invalid_argument("HBF cycle latencies disagree with physical sense/program times");
        }

        const auto matches = [](double actual, double expected) {
            return std::isfinite(actual) && std::abs(actual - expected) <= 1e-9 * std::max(1.0, expected);
        };
        if (!matches(config.hbf.read_bw_GBps, c.stacks * c.read_target_GBps_per_stack) ||
            !matches(config.hbf.write_bw_GBps, c.stacks * c.write_target_GBps_per_stack) ||
            (config.topology == MemoryTopology::CoLocated &&
             config.gpu_to_hbf.bandwidth_GBps > c.stacks * c.ucie_ceiling_GBps_per_stack)) {
            throw std::invalid_argument("HBF aggregate targets or physical link contradict per-stack configuration");
        }
        if (config.topology == MemoryTopology::CoLocated && config.hbm.stacks + c.stacks != 5) {
            throw std::invalid_argument("CoLocated detailed HBF uses a fixed five-stack package");
        }
        if (!config.address_layout.enabled ||
            config.address_layout.hbf_capacity_bytes != static_cast<std::uint64_t>(c.stacks) * hbfine::capacity_bytes_per_stack(c) ||
            config.address_layout.hbm_capacity_bytes != static_cast<std::uint64_t>(config.hbm.stacks) * config.hbm.capacity_bytes_per_stack) {
            throw std::invalid_argument("device segment capacities must match HBM/HBF stack counts");
        }
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
