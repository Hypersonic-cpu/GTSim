#include "hbf_placement.h"
#include <hbfine/controller.hpp>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <array>
#include <cassert>
#include <iostream>

int main() {
    hbfine::Config c;
    c.geometry = hbfine::GeometryProfile::GroupedPlanes;
    c.stacks = 2;
    c.dies_per_stack = 8;
    c.planes_per_die = 96;
    c.planes_per_command_group = 32;
    c.banks_per_channel = 48;
    c.pages_per_block = 192;
    c.blocks_per_plane = 342;
    c.read_target_GBps_per_stack = 768;
    constexpr std::uint64_t base = 4096;
    constexpr std::uint64_t mp = 131072;
    constexpr std::uint64_t hp = 48 * mp;
    GTSim::HBFMegapageAllocator allocate(base, c);
    const auto a = allocate.allocate(mp + 64);
    const auto b = allocate.allocate(2 * mp);
    assert(a.map(mp).bytes == 64 && b.base == base + 2 * mp);
    for (std::uint64_t offset :
         std::array<std::uint64_t, 6>{0, 63, 4096, mp - 1, hp, hp * 192}) {
        auto p = hbfine::decode_address(base + offset, base, c);
        assert(hbfine::encode_address(p, base, c) == base + offset);
    }
    const auto capacity = 2 * hbfine::capacity_bytes_per_stack(c);
    for (std::uint64_t offset = 0; offset < capacity; offset += 81767) {
        assert(hbfine::encode_address(
                   hbfine::decode_address(base + offset, base, c), base, c) ==
               base + offset);
    }
    const auto last = hbfine::decode_address(base + capacity - 1, base, c);
    assert(hbfine::encode_address(last, base, c) == base + capacity - 1);
    const auto wrap = allocate.allocate(hp);
    assert(wrap.map(hp - 4 * mp).address == base + hp);
    assert(hbfine::decode_address(base + hp, base, c).page == 1);
    hbfine::Controller controller(c, base, 16);
    controller.preload_region(base, 2 * hp);
    assert(controller.try_submit({0, base, 64, hbfine::Operation::Read}, 0));
    assert(controller.try_submit({1, base + 4096, 64, hbfine::Operation::Read},
                                 0));
    assert(
        controller.try_submit({2, base + mp, 64, hbfine::Operation::Read}, 0));
    assert(
        controller.try_submit({3, base + hp, 64, hbfine::Operation::Read}, 0));
    assert(controller.try_submit(
        {4, base + 2 * mp, 64, hbfine::Operation::Read}, 0));
    auto first = controller.advance(4100000);
    assert(first.size() == 4);
    auto next = controller.advance(8100000);
    assert(next.size() == 1 && next[0].request_id == 3);
    const auto& stats = controller.statistics();
    assert(stats.senses == 128 && stats.sensed_bytes == 4 * mp);
    assert(stats.host_read_bytes == 320 && !stats.failed_requests);
    assert(stats.bank_busy_ps[0] == 8000000 &&
           stats.bank_busy_ps[1] == 4000000);
    std::cout
        << "4 group senses, " << stats.sensed_bytes << " NAND bytes, "
        << stats.host_read_bytes << " host bytes, "
        << "concurrent groups and serialized same-group offsets verified\n";
}
