#ifndef GTSIM_HBM_ADDRESS_H
#define GTSIM_HBM_ADDRESS_H

#include <cstdint>
#include <stdexcept>

namespace GTSim {

struct HBMAddress {
    int stack;
    int logical_mc_group;
    int channel;
    int pseudochannel;
    int sid;
    int bank_group;
    int bank;
    int row;
    int column;
    int byte_offset;
};

// Reversible 128B striping across the 80 physical H100 HBM3 channels.
// This is an explicit modeling assumption, not NVIDIA's private address hash.
// After stripping the channel: byte[4:0], PC, burst-column[4:0], bank[1:0],
// bank-group[1:0], SID, row[13:0]. Column addresses are BL8 burst indices.
inline HBMAddress decode_h100_hbm_address(std::uint64_t address) {
    constexpr std::uint64_t capacity = 80ULL << 30;
    if (address >= capacity) {
        throw std::out_of_range("physical address exceeds modeled 80GiB HBM capacity");
    }
    const int channel = static_cast<int>((address / 128) % 80);
    std::uint64_t local = (address / (128 * 80)) * 128 + address % 128;
    const auto take = [&local](int bits) {
        const int value = static_cast<int>(local & ((1ULL << bits) - 1));
        local >>= bits;
        return value;
    };
    HBMAddress result{};
    result.channel = channel;
    result.stack = channel / 16;
    result.logical_mc_group = result.stack * 2 + (channel % 16) / 8;
    result.byte_offset = take(5);
    result.pseudochannel = take(1);
    result.column = take(5);
    result.bank = take(2);
    result.bank_group = take(2);
    result.sid = take(1);
    result.row = take(14);
    return result;
}

inline std::uint64_t encode_h100_hbm_address(const HBMAddress& address) {
    std::uint64_t local = address.row;
    local = (local << 1) | address.sid;
    local = (local << 2) | address.bank_group;
    local = (local << 2) | address.bank;
    local = (local << 5) | address.column;
    local = (local << 1) | address.pseudochannel;
    local = (local << 5) | address.byte_offset;
    return ((local / 128) * 80 + address.channel) * 128 + local % 128;
}

} // namespace GTSim
#endif
