#include <hbfine/address.hpp>
#include <hbfine/controller.hpp>
#include <cassert>
#include <iostream>

int main() {
    hbfine::Config c;
    c.geometry = hbfine::GeometryProfile::GroupedPlanes;
    c.stacks = 2;
    c.dies_per_stack = 8;
    c.planes_per_die = 96;
    c.planes_per_command_group = 1;
    c.banks_per_channel = 48;
    c.pages_per_block = 192;
    c.blocks_per_plane = 342;
    c.read_target_GBps_per_stack = 768;
    hbfine::validate_config(c);
    constexpr std::uint64_t base = 4096;
    assert(hbfine::scheduling_unit_count(c) == 1536);
    for (std::uint64_t index : {0ULL, 1ULL, 31ULL, 32ULL, 1535ULL, 1536ULL}) {
        const auto address = base + index * 4096;
        const auto decoded = hbfine::decode_address(address, base, c);
        const auto mapped = hbfine::map_read(decoded, address, base, c);
        assert(hbfine::encode_address(decoded, base, c) == address);
        assert(mapped.plane_count == 1 && mapped.buffer_key == address);
        assert(mapped.unit == index % 1536);
    }
    hbfine::Controller controller(c, base, 16);
    controller.preload_region(base, 4096 * 1537);
    assert(controller.try_submit({0, base, 64, hbfine::Operation::Read}, 0));
    assert(controller.try_submit({1, base + 4096, 64, hbfine::Operation::Read},
                                 0));
    assert(controller.try_submit(
        {2, base + 4096 * 1536, 64, hbfine::Operation::Read}, 0));
    auto first = controller.advance(4100000);
    assert(first.size() == 2);
    auto next = controller.advance(8100000);
    assert(next.size() == 1 && next[0].request_id == 2);
    const auto& stats = controller.statistics();
    assert(stats.sense_commands == 3 && stats.senses == 3 &&
           stats.sensed_bytes == 12288);
    assert(stats.unique_sense_addresses == 3 && stats.host_read_bytes == 192);
    std::cout << "4 KiB per Sense, round-trip addressing, independent planes "
                 "and same-plane serialization verified\n";
}
