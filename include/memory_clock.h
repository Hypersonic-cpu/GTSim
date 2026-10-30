#ifndef GTSIM_MEMORY_CLOCK_H
#define GTSIM_MEMORY_CLOCK_H

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace GTSim {

// Integer phase conversion avoids rounding each GPU cycle independently.
// Frequency resolution is 1kHz; time resolution is the backend's tick in ps.
class MemoryClock {
public:
    MemoryClock(double core_frequency_mhz, int tick_ps) {
        if (!std::isfinite(core_frequency_mhz) || core_frequency_mhz <= 0 ||
            core_frequency_mhz > 1000000 || tick_ps <= 0) {
            throw std::invalid_argument("invalid memory clock domain");
        }
        const auto frequency_khz = static_cast<std::uint64_t>(
            std::llround(core_frequency_mhz * 1000.0));
        if (!frequency_khz) throw std::invalid_argument("core frequency is too small");
        denominator_ = frequency_khz * static_cast<std::uint64_t>(tick_ps);
    }

    std::uint64_t ticks_at(int core_cycle) const {
        if (core_cycle < 0) throw std::invalid_argument("negative core cycle");
        return static_cast<std::uint64_t>(core_cycle) * 1000000000ULL / denominator_;
    }

    std::uint64_t first_edge_at_or_after(int core_cycle) const {
        if (core_cycle < 0) throw std::invalid_argument("negative core cycle");
        const auto numerator = static_cast<std::uint64_t>(core_cycle) * 1000000000ULL;
        return numerator / denominator_ + (numerator % denominator_ != 0);
    }

private:
    std::uint64_t denominator_{};
};

} // namespace GTSim
#endif
