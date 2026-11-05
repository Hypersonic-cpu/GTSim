#include "memory.h"

#include <cassert>
#include <set>

using namespace GTSim;

int main() {
    const TileMajorLayout example{2, 4, 8};
    assert(example.offset(0, 0, 8) == 0);
    assert(example.offset(1, 0, 8) == 1);
    assert(example.offset(2, 0, 8) == 8);
    assert(example.offset(0, 4, 8) == 32);
    std::vector<int> physical(64, -1);
    for (int k = 0; k < 8; ++k)
        for (int n = 0; n < 8; ++n) {
            const auto pos = example.offset(k, n, 8);
            assert(physical.at(pos) == -1);
            physical[pos] = k * 8 + n;
        }
    for (int n = 0; n < 8; ++n) {
        int row_major = 0, tile_major = 0;
        for (int k = 0; k < 8; ++k) {
            row_major += (k + 1) * (k * 8 + n);
            const auto index = 32 * (n / 4) + 8 * (k / 2) + 2 * (n % 4) + k % 2;
            tile_major += (k + 1) * physical[index];
        }
        assert(tile_major == row_major);
    }

    const TileMajorLayout layout{16, 128, 32};
    constexpr std::uint64_t base = 0x10000;
    TensorMetadata weights{1,
                           "weights",
                           base,
                           {64, 256},
                           {256, 1},
                           DataType::BF16,
                           MemoryTier::HBF,
                           true,
                           true,
                           layout};
    std::set<std::uint64_t> offsets;
    for (int k = 0; k < 64; ++k)
        for (int n = 0; n < 256; ++n) {
            const auto offset = layout.offset(k, n, 256) * 2;
            assert((offset >= 32 * 256 * 2) == (k >= 32));
            assert(offsets.insert(offset).second);
        }
    assert(offsets.size() == 64 * 256 && *offsets.rbegin() == 64 * 256 * 2 - 2);

    for (const int width : {64, 128, 256}) {
        DAGNode node(0, "test", "LD", "cp.dram2sram_tma", 0, {}, 0,
                     Tile(48, 0, 16, width), DataType::BF16);
        node.matrix_id = 1;
        std::set<std::uint64_t> lines;
        for (int i = 0; i < compute_dram_subops(node, 16); ++i)
            for (const auto& key :
                 compute_dram_subop_lines(node, weights, 16, i, 128))
                assert(lines.insert(key.global_line_addr).second);
        std::set<std::uint64_t> expected;
        for (int c = 0; c < width / 4; ++c)
            expected.insert(base + 5 * 4096 + (c / 32) * 2 * 4096 +
                            (c % 32) * 128);
        assert(lines == expected);
    }
}
