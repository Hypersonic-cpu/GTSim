#pragma once
#include <hbfine/address.hpp>
namespace GTSim {
using HBFAddress = hbfine::Address;
inline auto decode_hbf_address(std::uint64_t address, std::uint64_t base, const hbfine::Config& c) {
    return hbfine::decode_address(address, base, c);
}
inline auto encode_hbf_address(const HBFAddress& address, std::uint64_t base, const hbfine::Config& c) {
    return hbfine::encode_address(address, base, c);
}
inline double hbf_array_read_ceiling_GBps(const hbfine::Config& c) { return hbfine::array_read_ceiling_GBps(c); }
inline double hbf_array_write_ceiling_GBps(const hbfine::Config& c) { return hbfine::array_write_ceiling_GBps(c); }
} // namespace GTSim
