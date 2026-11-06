#include <hbfine/controller.hpp>

#include <cassert>
#include <tuple>
#include <vector>
#include <iostream>

struct Run {
    hbfine::Statistics stats;
    std::vector<std::pair<std::uint64_t, hbfine::Time>> completed;
};

Run simulate(hbfine::Time interval, bool grouped) {
    hbfine::Config config;
    if (grouped) {
        config.geometry = hbfine::GeometryProfile::GroupedPlanes;
        config.stacks = 2;
        config.dies_per_stack = 8;
        config.planes_per_die = 96;
        config.planes_per_command_group = 32;
        config.banks_per_channel = 48;
        config.pages_per_block = 192;
        config.blocks_per_plane = 342;
        config.read_target_GBps_per_stack = 768;
    }
    config.enable_read_sram = true;
    config.read_sram_access_latency_ns = 20;
    config.read_sram_bandwidth_GBps_per_stack = 1536;
    config.statistics_interval_ps = interval;
    hbfine::Controller device(config, 0);
    device.preload_region(0, 12288);
    assert(device.try_submit({0, 0, 64, hbfine::Operation::Read}, 0));
    assert(device.try_submit(
        {1, 4096, 128, hbfine::Operation::Read, -1, hbfine::ReadPath::Prefetch},
        0));
    assert(device.try_submit(
        {2, 8192, 64, hbfine::Operation::Read, -1, hbfine::ReadPath::Resident},
        0));
    Run run;
    for (hbfine::Time now = 0; now <= 6000000; now += 1000)
        for (const auto& done : device.advance(now))
            run.completed.emplace_back(done.request_id, now);
    run.stats = device.statistics();
    return run;
}

int main() {
    for (const bool grouped : {false, true}) {
        const auto disabled = simulate(0, grouped);
        const auto enabled = simulate(1000000, grouped);
        assert(disabled.stats.bandwidth_bins.empty());
        assert(disabled.completed == enabled.completed &&
               enabled.completed.size() == 3);
        assert(disabled.stats.sensed_bytes == enabled.stats.sensed_bytes);
        std::uint64_t media = 0, demand = 0, fill = 0, resident = 0,
                      commands = 0;
        for (const auto& bin : enabled.stats.bandwidth_bins) {
            assert(bin.end_ps - bin.start_ps == 1000000);
            media += bin.media_read_bytes;
            demand += bin.host_read_bytes;
            fill += bin.prefetch_fill_bytes;
            resident += bin.resident_read_bytes;
            commands += bin.sense_commands;
        }
        assert(media == (grouped ? 131072 : 8192) && demand == 128 &&
               fill == 128 && resident == 64);
        assert(commands == (grouped ? 1 : 2) &&
               media == enabled.stats.sensed_bytes);
    }
    std::cout << "Bandwidth sampling preserves completion times and conserves "
                 "media/demand/prefetch/resident bytes\n";
}
