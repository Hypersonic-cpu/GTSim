#include "memory.h"

namespace GTSim {

#ifdef GTSIM_ENABLE_RAMULATOR2
std::unique_ptr<MemoryBackend> make_ramulator_hbm_backend(
    const HBMConfig& config, double core_frequency_mhz);
#endif

std::unique_ptr<MemoryBackend> make_hbm_backend(const HBMConfig& config,
                                              double core_frequency_mhz) {
    switch (config.backend) {
    case HBMBackend::Naive:
        return std::make_unique<HBMModel>(config, core_frequency_mhz);
    case HBMBackend::Ramulator2:
#ifdef GTSIM_ENABLE_RAMULATOR2
        return make_ramulator_hbm_backend(config, core_frequency_mhz);
#else
        throw std::runtime_error(
            "Ramulator2 backend requested; rebuild with -DGTSIM_ENABLE_RAMULATOR2=ON");
#endif
    }
    throw std::invalid_argument("unknown HBM backend");
}

} // namespace GTSim
