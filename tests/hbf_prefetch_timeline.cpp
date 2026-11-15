#include "memory/base.h"

#include <cassert>
#include <iostream>

using namespace GTSim;

int main() {
    HBFConfig config;
    config.backend = HBFBackend::HBFine;
    config.controller.enable_read_sram = true;
    config.controller.sram_bytes_per_plane = 4096;
    config.controller.base_die_sram_bytes_per_stack = 65536;
    config.controller.read_sram_bandwidth_GBps_per_stack = 1536;
    config.controller.read_sram_access_latency_ns = 20;
    config.controller.write_pages_per_channel = 1;
    config.prefetch_engine.queue_capacity = 1;

    auto backend = make_hbf_backend(config, 1980.0, 0);
    backend->preload_region(0, 8192, true);
    constexpr int router_completion = 100;

    const auto enqueue0 = router_completion;
    const auto expert0 = backend->enqueue_prefetch(0, 4096);
    assert(expert0 != 0);
    assert(backend->prefetch_alias(expert0) == 0);

    int admission0 = -1;
    for (int cycle = enqueue0; cycle < enqueue0 + 4 && admission0 < 0;
         ++cycle) {
        backend->advance_prefetch(cycle);
        admission0 = backend->prefetch_admission_cycle(expert0);
    }
    assert(admission0 >= enqueue0);

    const auto enqueue1 = admission0;
    const auto expert1 = backend->enqueue_prefetch(4096, 4096);
    assert(expert1 != 0);
    int admission1 = -1;
    for (int cycle = enqueue1; cycle < enqueue1 + 4 && admission1 < 0;
         ++cycle) {
        backend->advance_prefetch(cycle);
        admission1 = backend->prefetch_admission_cycle(expert1);
    }
    assert(admission1 >= enqueue1);

    int ready0 = -1;
    int ready1 = -1;
    for (int cycle = admission1; cycle < 100000 && (ready0 < 0 || ready1 < 0);
         ++cycle) {
        backend->step(cycle);
        if (ready0 < 0 && backend->prefetch_ready(expert0)) ready0 = cycle;
        if (ready1 < 0 && backend->prefetch_ready(expert1)) ready1 = cycle;
    }
    assert(ready0 >= admission0 && ready1 >= admission1);
    backend->release_prefetch(expert0);
    backend->release_prefetch(expert1);

    std::cout << "prefetch_timeline router=" << router_completion
              << " expert0=" << enqueue0 << "," << admission0 << ","
              << ready0 << " expert1=" << enqueue1 << "," << admission1
              << "," << ready1 << '\n';
}
