#include "hbf_address.h"
#include "memory_backend.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace GTSim;

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    HBFConfig config;
    config.backend = HBFBackend::BankPlane;
    auto& c = config.controller;
    c.enable_read_sram = true;
    c.read_sram_access_latency_ns = 20;
    c.read_sram_bandwidth_GBps_per_stack = 1536;
    c.sram_bytes_per_plane = 4096;
    c.base_die_sram_bytes_per_stack = 65536;
    c.write_pages_per_channel = 1;

    auto device = make_hbf_backend(config, 1980, 0);
    device->preload_region(0, hbfine::capacity_bytes_per_stack(c), true);
    const auto first = device->reserve_prefetch(0, 4096);
    require(first != 0, "plane reservation failed");
    std::vector<std::uint64_t> spill;
    for (int i = 0; i < 16; ++i) {
        const auto token = device->reserve_prefetch(0, 4096);
        require(token != 0, "base die fallback failed");
        spill.push_back(token);
    }
    require(device->reserve_prefetch(0, 4096) == 0, "capacity overcommit");
    device->start_prefetch(first);
    device->step(0);
    require(device->pending_requests() == 0 &&
                device->pending_background_requests() == 1 &&
                !device->prefetch_ready(first),
            "ownership or early residency");
    require(
        device->try_submit(
            {1, 0, 128, MemoryOperation::Read, MemoryTier::HBF, {}, first}, 1),
        "fallback admission");
    auto at = 1;
    for (; at < 20000 && (device->pending_requests() ||
                          device->pending_background_requests());
         ++at)
        device->step(at);
    require(device->prefetch_ready(first), "prefetch failed to drain");
    auto snapshot = device->device_counters();
    require(snapshot.cumulative.at("fallback_read_bytes") == 128 &&
                snapshot.cumulative.at("prefetch_bytes") == 4096 &&
                device->hbf_statistics().host_read_bytes == 128 &&
                snapshot.cumulative.at("sensed_pages") == 1,
            "shared sense or traffic mismatch");

    require(
        device->try_submit(
            {2, 0, 128, MemoryOperation::Read, MemoryTier::HBF, {}, first}, at),
        "resident admission");
    device->release_prefetch(first);
    require(device->reserve_prefetch(0, 4096) == 0,
            "released live resident data");
    for (; at < 21000 && device->pending_requests(); ++at) device->step(at);
    const auto reused = device->reserve_prefetch(0, 4096);
    require(reused != 0, "capacity not reclaimed after consumer");
    snapshot = device->device_counters();
    require(snapshot.cumulative.at("resident_read_bytes") == 128 &&
                snapshot.cumulative.at("sensed_pages") == 1,
            "resident read sensed NAND");

    // An unaliased request still uses the ordinary read path.
    require(device->try_submit(
                {3, 0, 128, MemoryOperation::Read, MemoryTier::HBF}, at),
            "ordinary read admission");
    for (; at < 22000 && device->pending_requests(); ++at) device->step(at);
    require(device->device_counters().cumulative.at("resident_read_bytes") ==
                128,
            "ordinary read used an alias");

    device->start_prefetch(reused);
    device->release_prefetch(reused);
    require(device->reserve_prefetch(0, 4096) == 0, "released filling SRAM");
    for (; at < 23000 && device->pending_background_requests(); ++at)
        device->step(at);
    const auto after_fill = device->reserve_prefetch(0, 4096);
    require(after_fill != 0, "capacity not reclaimed after a released fill");
    device->release_prefetch(after_fill);
    for (const auto token : spill) device->release_prefetch(token);
    auto shared = make_hbf_backend(config, 1980, 0);
    shared->preload_region(0, hbfine::capacity_bytes_per_stack(c), true);
    auto address = hbfine::decode_address(0, 0, c);
    address.page = 1;
    const auto other_page = hbfine::encode_address(address, 0, c);
    const auto background = shared->reserve_prefetch(0, 4096);
    require(background != 0, "contention reservation failed");
    shared->start_prefetch(background);
    shared->step(0);
    require(
        shared->try_submit(
            {4, other_page, 128, MemoryOperation::Read, MemoryTier::HBF}, 1),
        "contending read admission");
    int completion = -1;
    for (int cycle = 1; cycle < 20000 && completion < 0; ++cycle)
        if (!shared->step(cycle).empty()) completion = cycle;
    require(completion >= 2 * 7920 && shared->hbf_statistics().senses == 2,
            "foreground and background bypassed shared NAND sense engine");
    shared->release_prefetch(background);

    // Separate slots retain equal per-plane/base-die quotas and tile leases.
    for (const int count : {2, 4}) {
        auto banked_config = config;
        banked_config.controller.sram_bytes_per_plane = 16384;
        auto banked = make_hbf_backend(banked_config, 1980, 0);
        banked->preload_region(0, hbfine::capacity_bytes_per_stack(c), true);
        const PrefetchBuffer current{0, count, 0}, other{1, count, 1};
        const PrefetchBuffer next{0, count, static_cast<std::uint64_t>(count)};
        const auto first_slot = banked->reserve_prefetch(0, 4096, current);
        const auto second_slot = banked->reserve_prefetch(0, 4096, other);
        require(first_slot && second_slot, "independent buffer reservation");
        require(!banked->reserve_prefetch(0, 4096, next),
                "overwrote a live buffer");
        banked->start_prefetch(first_slot);
        banked->step(0);
        banked->release_prefetch(first_slot);
        require(!banked->reserve_prefetch(0, 4096, next),
                "overwrote an in-flight fill");
        int cycle = 1;
        while (banked->pending_background_requests() && cycle < 20000)
            banked->step(cycle++);
        std::vector<std::uint64_t> leases;
        for (int page = 0; page < (16384 + 65536) / count / 4096; ++page) {
            const auto lease = banked->reserve_prefetch(0, 4096, next);
            require(lease != 0, "fixed buffer quota unexpectedly unavailable");
            leases.push_back(lease);
        }
        require(!banked->reserve_prefetch(0, 4096, next),
                "buffer quota overcommit");
        for (const auto lease : leases) banked->release_prefetch(lease);
        banked->release_prefetch(second_slot);
    }

    // A one-page plane belongs to one bank as a whole, not two half-pages.
    auto small_config = config;
    small_config.controller.sram_bytes_per_plane = 4096;
    small_config.controller.base_die_sram_bytes_per_stack = 65536;
    auto small = make_hbf_backend(small_config, 1980, 0);
    small->preload_region(0, hbfine::capacity_bytes_per_stack(c), true);
    std::vector<std::uint64_t> small_leases;
    for (int page = 0; page < 9; ++page) {
        auto token = small->reserve_prefetch(0, 4096, PrefetchBuffer{0, 2, 0});
        require(token != 0, "one-page plane quota lost in partition");
        small_leases.push_back(token);
    }
    require(!small->reserve_prefetch(0, 4096, PrefetchBuffer{0, 2, 0}),
            "one-page plane bank exceeded quota");
    for (const auto token : small_leases) small->release_prefetch(token);

    std::cout << "HBF SRAM capacity, fallback, shared sense, alias isolation "
                 "and deferred release passed\n";
}
