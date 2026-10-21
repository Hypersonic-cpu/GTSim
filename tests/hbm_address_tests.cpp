#include "hbm_address.h"
#include "memory_clock.h"
#include <array>
#include <vector>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace GTSim;
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
    try {
        std::array<int, 80> channels{};
        std::array<int, 160> pseudochannels{};
        std::array<int, 10> groups{};
        for (std::uint64_t address = 0; address < 128 * 80 * 16; address += 32) {
            const auto mapped = decode_h100_hbm_address(address);
            require(encode_h100_hbm_address(mapped) == address, "address mapping aliases bytes");
            require(mapped.stack == mapped.channel / 16, "wrong stack");
            ++channels[mapped.channel];
            ++pseudochannels[mapped.channel * 2 + mapped.pseudochannel];
            ++groups[mapped.logical_mc_group];
        }
        for (auto count : channels) require(count == channels[0] && count > 0, "channel imbalance");
        for (auto count : pseudochannels) require(count == pseudochannels[0] && count > 0, "PC imbalance");
        for (auto count : groups) require(count == groups[0] && count > 0, "MC group imbalance");
        std::mt19937_64 random(20261021);
        for (int i = 0; i < 100000; ++i) {
            const auto address = random() % (80ULL << 30);
            const auto mapped = decode_h100_hbm_address(address);
            require(encode_h100_hbm_address(mapped) == address, "high address alias or byte loss");
            require(encode_h100_hbm_address(decode_h100_hbm_address(address)) == address,
                    "mapping is not deterministic");
        }
        for (const auto address : {0ULL, (80ULL << 30) - 1}) {
            require(encode_h100_hbm_address(decode_h100_hbm_address(address)) == address,
                    "capacity boundary mismatch");
        }
        bool rejected = false;
        try { decode_h100_hbm_address(80ULL << 30); }
        catch (const std::out_of_range&) { rejected = true; }
        require(rejected, "out of range address was silently truncated");

        for (int stacks : {2, 4, 5}) {
            std::vector<int> distribution(stacks * 16);
            const auto capacity = static_cast<std::uint64_t>(stacks) * (16ULL << 30);
            for (std::uint64_t address = 0; address < 128ULL * distribution.size() * 16; address += 128) {
                const auto decoded = decode_h100_hbm_address(address, stacks);
                require(encode_h100_hbm_address(decoded, stacks) == address, "scaled topology mapping alias");
                ++distribution.at(decoded.channel);
            }
            for (auto count : distribution) require(count == distribution[0], "scaled channel imbalance");
            for (int i = 0; i < 10000; ++i) {
                const auto address = random() % capacity;
                require(encode_h100_hbm_address(decode_h100_hbm_address(address, stacks), stacks) == address,
                        "scaled capacity mapping alias");
            }
        }

        const MemoryClock exact(1000, 500);
        require(exact.ticks_at(0) == 0 && exact.ticks_at(1) == 2 &&
                exact.ticks_at(100000000) == 200000000, "integer clock conversion");
        require(exact.first_edge_at_or_after(1) == 2, "coincident edge was delayed");
        const MemoryClock fractional(1980, 383);
        require(fractional.ticks_at(1) == 1 && fractional.first_edge_at_or_after(1) == 2,
                "fractional arrival edge was rounded down");
        std::uint64_t accumulated = 0;
        for (int i = 1; i <= 100000; ++i) {
            accumulated += fractional.ticks_at(i) - fractional.ticks_at(i - 1);
        }
        require(accumulated == 100000ULL * 1000000000 / (1980000ULL * 383),
                "fractional clock lost phase");
        require(MemoryClock(1375.5, 383).ticks_at(100000000) ==
                    100000000ULL * 1000000000 / (1375500ULL * 383),
                "fractional core frequency conversion");
        std::cout << "HBM address and clock checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
