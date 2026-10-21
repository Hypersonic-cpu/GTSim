#include "memory.h"
#include "hbf_address.h"
#include <iostream>
#include <random>
#include <set>

using namespace GTSim;
namespace {
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> void rejects(F action, const char* message) {
    bool caught = false;
    try { action(); } catch (const std::exception&) { caught = true; }
    require(caught, message);
}

MemorySystemConfig profile(HBFGeometryProfile geometry = HBFGeometryProfile::HBFlexResearch) {
    MemorySystemConfig m;
    m.topology = MemoryTopology::CoLocated;
    m.hbm.stacks = 4;
    m.hbm.model_channels = true;
    m.hbf.backend = HBFBackend::BankPlane;
    m.hbf.read_bw_GBps = 488;
    m.hbf.write_bw_GBps = 27.2;
    m.hbf.controller.geometry = geometry;
    if (geometry == HBFGeometryProfile::OCPReference) {
        m.hbf.controller.banks_per_channel = 16;
        m.hbf.controller.planes_per_bank = 2;
    }
    m.address_layout.enabled = true;
    m.address_layout.hbm_capacity_bytes = 64ULL << 30;
    m.address_layout.hbf_base = 64ULL << 30;
    m.address_layout.hbf_capacity_bytes = 512ULL << 30;
    m.address_layout.hbm_allocation_offset = 0;
    validate_memory_system_config(m);
    return m;
}

std::uint64_t location(const MemorySystemConfig& m, int bank, int page = 0,
                       int plane_in_bank = 0, int channel = 0, int block = 0) {
    HBFAddress a;
    a.channel = channel; a.bank = bank; a.page = page; a.block = block;
    const int physical = (channel * m.hbf.controller.banks_per_channel + bank) *
                         m.hbf.controller.planes_per_bank + plane_in_bank;
    a.die = physical / 32; a.plane = physical % 32;
    return encode_hbf_address(a, m.address_layout.hbf_base, m.hbf.controller);
}

std::unique_ptr<MemoryBackend> device(const MemorySystemConfig& m, bool preload = true, double mhz = 1000) {
    validate_memory_system_config(m);
    auto result = make_hbf_backend(m.hbf, mhz, m.address_layout.hbf_base);
    if (preload) result->preload_region(m.address_layout.hbf_base, m.address_layout.hbf_capacity_bytes, true);
    return result;
}

void mapping_and_constraints() {
    std::mt19937_64 random(1021);
    for (auto geometry : {HBFGeometryProfile::OCPReference, HBFGeometryProfile::HBFlexResearch}) {
        auto m = profile(geometry);
        for (int stripe : {4096, 8192, 16384}) {
            m.hbf.controller.host_stripe_bytes = stripe;
            m.hbf.controller.virtual_axi_ports_per_channel = 4;
            for (int i = 0; i < 20000; ++i) {
                const auto address = m.address_layout.hbf_base + random() % m.address_layout.hbf_capacity_bytes;
                const auto a = decode_hbf_address(address, m.address_layout.hbf_base, m.hbf.controller);
                require(encode_hbf_address(a, m.address_layout.hbf_base, m.hbf.controller) == address,
                        "HBF address is not a bijection");
                const auto page = address / 4096 * 4096;
                const auto first = decode_hbf_address(page, m.address_layout.hbf_base, m.hbf.controller);
                const auto last = decode_hbf_address(page + 4095, m.address_layout.hbf_base, m.hbf.controller);
                require(first.channel == last.channel && first.stack == last.stack && first.plane == last.plane,
                        "a Core Die page was split between physical channels");
            }
        }
        m.hbf.controller.planes_per_bank = 3;
        rejects([&] { validate_memory_system_config(m); }, "contradictory Banks/Planes accepted");
    }
    require(std::abs(hbf_array_read_ceiling_GBps(profile().hbf.controller) - 524.288) < 1e-9,
            "research sense ceiling is incorrect");
    require(std::abs(hbf_array_read_ceiling_GBps(profile(HBFGeometryProfile::OCPReference).hbf.controller) - 262.144) < 1e-9,
            "page buffers incorrectly doubled the OCP sense ceiling");
    auto bad = profile(); bad.hbf.controller.allow_program = true;
    rejects([&] { validate_memory_system_config(bad); }, "programs accepted without an explicit erase/timeout assumption");
    bad = profile(); bad.hbf.read_bw_GBps *= 2;
    rejects([&] { validate_memory_system_config(bad); }, "aggregate targets contradict device accepted");
    bad = profile(); bad.hbf.controller.host_stripe_bytes = 64;
    rejects([&] { validate_memory_system_config(bad); }, "cross-channel subpage striping accepted");
}

void sense_order_and_buffers() {
    auto m = profile(HBFGeometryProfile::OCPReference);
    auto backend = device(m);
    for (int i = 0; i < 3; ++i) {
        auto address = i == 1 ? location(m, 0, 0, 1) : location(m, i == 2 ? 1 : 0);
        require(backend->try_submit({static_cast<std::uint64_t>(i), address, 64, MemoryOperation::Read, MemoryTier::HBF}, 0), "read admission");
    }
    auto first = backend->step(5000);
    require(first.size() == 2, "independent Bank did not overlap, or two buffers created extra sense parallelism");
    auto second = backend->step(9000);
    require(second.size() == 1 && second[0].request_id == 1, "same-Bank different-Plane sense was not serialized");
    require(backend->hbf_statistics().senses == 3, "sense coalescing mismatch");

    m = profile(); backend = device(m);
    for (int i = 0; i < 3; ++i) {
        require(backend->try_submit({static_cast<std::uint64_t>(i), location(m, 0, i), 64,
                                    MemoryOperation::Read, MemoryTier::HBF}, i * 5000), "cache read admission");
        require(backend->step((i + 1) * 5000).size() == 1, "cache miss failed to complete");
    }
    require(backend->hbf_statistics().cache_evictions == 1, "two-page Bank cache eviction is incorrect");
    require(backend->try_submit({10, location(m, 0, 2), 64, MemoryOperation::Read, MemoryTier::HBF}, 15000), "cache hit admission");
    require(backend->step(15010).size() == 1 && backend->hbf_statistics().senses == 3,
            "cache hit unnecessarily sensed the array");

    m = profile(); m.hbf.controller.read_target_GBps_per_stack = 1.024; m.hbf.read_bw_GBps = 1.024; // 64B/us per Channel
    backend = device(m);
    for (int i = 0; i < 3; ++i) require(backend->try_submit({static_cast<std::uint64_t>(i), location(m, 0, i), 4096,
                        MemoryOperation::Read, MemoryTier::HBF}, 0), "pinned cache admission");
    backend->step(12000);
    require(backend->hbf_statistics().senses == 2, "in-flight page data was evicted from a pinned buffer");
    const auto drain_cycles = static_cast<int>(std::ceil(1000 * (10 + 3 * 4096 /
        (m.hbf.controller.read_target_GBps_per_stack / 16 * 1000))));
    require(backend->step(drain_cycles).size() == 3 && backend->pending_requests() == 0,
            "pinned buffers did not wake blocked reads");
}

void admission_clock_and_axi() {
    auto m = profile(); m.hbf.controller.channel_queue_capacity = 64;
    auto backend = device(m);
    require(backend->try_submit({1, location(m, 0), 4096, MemoryOperation::Read, MemoryTier::HBF}, 0), "full-page admission");
    require(!backend->try_submit({2, location(m, 1), 64, MemoryOperation::Read, MemoryTier::HBF}, 0), "channel queue did not backpressure");
    require(backend->step(10000).size() == 1, "rejected request was silently admitted");
    require(backend->try_submit({2, location(m, 1), 64, MemoryOperation::Read, MemoryTier::HBF}, 10000), "retry rejected after capacity freed");
    require(backend->step(20000).size() == 1 && backend->pending_requests() == 0, "retry conservation");
    require(backend->hbf_statistics().accepted_requests == 2 && backend->hbf_statistics().completed_requests == 2,
            "failed admission changed request accounting");

    backend = device(m, true, 1980);
    backend->try_submit({3, location(m, 0), 64, MemoryOperation::Read, MemoryTier::HBF}, 0);
    require(backend->step(7924).empty() && backend->step(7925).size() == 1, "microsecond/core clock conversion is wrong");

    m = profile(); backend = device(m);
    backend->try_submit({0, location(m, 1), 64, MemoryOperation::Read, MemoryTier::HBF}, 0);
    backend->step(5000);
    MemoryRoutingHint same; same.axi_id = 7;
    backend->try_submit({1, location(m, 0), 64, MemoryOperation::Read, MemoryTier::HBF, same}, 5000);
    backend->try_submit({2, location(m, 1), 64, MemoryOperation::Read, MemoryTier::HBF, same}, 5000);
    require(backend->step(6000).empty(), "same AXI ID cache hit overtook a miss");
    const auto ordered = backend->step(10000);
    require(ordered.size() == 2 && ordered[0].request_id == 1 && ordered[1].request_id == 2, "AXI ID order changed");
    backend->try_submit({3, location(m, 0, 1), 64, MemoryOperation::Read, MemoryTier::HBF}, 10000);
    backend->try_submit({4, location(m, 1), 64, MemoryOperation::Read, MemoryTier::HBF}, 10000);
    const auto bypass = backend->step(11000);
    require(bypass.size() == 1 && bypass[0].request_id == 4, "independent AXI IDs were globally serialized");
    backend->step(15000);
}

void writes_and_failures() {
    auto m = profile();
    m.hbf.controller.allow_program = true;
    // Artificial UNIT TEST timings, not claimed HBF device parameters.
    m.hbf.controller.erase_latency_us = 10;
    m.hbf.controller.write_timeout_us = 10;
    auto backend = device(m, false);
    for (int i = 0; i < 64; ++i) require(backend->try_submit({static_cast<std::uint64_t>(i), location(m, 0) + i * 64,
                            64, MemoryOperation::Write, MemoryTier::HBF}, 0), "write piece admission");
    require(backend->step(80000).empty(), "write acknowledged before program data reached Core Die");
    const auto programmed = backend->step(100000);
    require(programmed.size() == 64, "4KiB write accumulation lost an ID");
    for (const auto& response : programmed) require(response.completed, "legal fresh-block write failed");
    require(backend->hbf_statistics().erases == 1 && backend->hbf_statistics().programs == 1, "page-0 auto-erase/program accounting");
    backend->try_submit({100, location(m, 0, 1), 4096, MemoryOperation::Writeback, MemoryTier::HBF}, 100000);
    require(backend->step(200000).at(0).completed, "next sequential page writeback failed");
    backend->try_submit({101, location(m, 0, 3), 4096, MemoryOperation::Write, MemoryTier::HBF}, 200000);
    require(!backend->step(210000).at(0).completed, "skipped block page accepted");
    backend->try_submit({102, location(m, 0), 4096, MemoryOperation::Write, MemoryTier::HBF}, 210000);
    require(!backend->step(220000).at(0).completed, "arbitrary rewrite implicitly erased valid NAND");
    require(backend->hbf_statistics().programs == 2 && backend->hbf_statistics().erases == 1, "illegal writes modified NAND");
    backend->try_submit({103, location(m, 0, 0, 0, 0, 1), 64, MemoryOperation::Write, MemoryTier::HBF}, 220000);
    require(!backend->step(240000).at(0).completed, "partial page did not time out");
    require(backend->pending_requests() == 0, "failed write did not drain");

    m.hbf.controller.write_timeout_us = 0.05;
    backend = device(m, false);
    backend->try_submit({0, location(m, 0), 4096, MemoryOperation::Write, MemoryTier::HBF}, 0);
    const auto late = backend->step(5000);
    require(late.size() == 1 && !late[0].completed && backend->pending_requests() == 0,
            "timeout dropped or duplicated pieces still in the channel serializer");
    require(backend->hbf_statistics().programs == 0, "partial/expired page programmed");

    m = profile(); backend = device(m, false);
    backend->try_submit({0, location(m, 0), 64, MemoryOperation::Read, MemoryTier::HBF}, 0);
    require(!backend->step(100).at(0).completed, "uninitialized NAND read fabricated valid data");
    rejects([&] { backend->try_submit({1, location(m, 0), 4096, MemoryOperation::Write, MemoryTier::HBF}, 100); },
            "read-only experiment admitted an HBF write");
    require(backend->hbf_statistics().accepted_requests == 1, "invalid write changed admission accounting");
}

void shared_path_failures() {
    auto m = profile();
    TensorRegistry registry(m.address_layout);
    registry.allocate_tensor(0, "input", {64}, {1}, DataType::FP16, MemoryTier::HBF, true, true);
    MemorySystem memory(0, 128, 0, 4096, 128, 64, true, m, 1000, registry);
    rejects([&] { memory.l2()->handle_response({999, m.address_layout.hbf_base,
                        MemoryTier::HBF, false, 16, MemoryOperation::Writeback, 3}, 0); },
            "failed writeback without L2 waiter was silently discarded");
    auto wrong_layout = m.address_layout; wrong_layout.hbm_allocation_offset = 4096;
    TensorRegistry mismatched(wrong_layout);
    rejects([&] { MemorySystem invalid(0, 128, 0, 4096, 128, 64, true, m, 1000, mismatched); },
            "controller and TensorRegistry accepted mismatched layouts");
}

void optional_internal_limits() {
    auto m = profile();
    m.hbf.controller.core_transfer_GBps_per_die = 4.096; // 1us/page, synthetic test only
    m.hbf.controller.ecc_GBps_per_stack = 4.096;
    auto backend = device(m);
    backend->try_submit({0, location(m, 0), 64, MemoryOperation::Read, MemoryTier::HBF}, 0);
    backend->try_submit({1, location(m, 0, 0, 0, 1), 64, MemoryOperation::Read, MemoryTier::HBF}, 0);
    require(backend->step(6000).empty(), "optional core/ECC resource was bypassed");
    require(backend->step(6500).size() == 1 && backend->step(7500).size() == 1,
            "shared ECC and independent die transfers modeled incorrectly");
    const auto stats = backend->hbf_statistics();
    require(stats.ecc_busy_ps[0] == 2000000 && stats.events_processed < 100,
            "internal bandwidth accounting or sparse event scheduling failed");
}
}

int main() {
    try {
        mapping_and_constraints();
        sense_order_and_buffers();
        admission_clock_and_axi();
        writes_and_failures();
        optional_internal_limits();
        shared_path_failures();
        std::cout << "HBF mapping, Bank/Plane, caches, AXI, NAND legality and conservation checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
