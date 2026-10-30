#include "arch_conf.h"
#include "hbf_address.h"
#include "memory_backend.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

using namespace GTSim;
namespace {
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void matched_profiles(const char* simple, const char* detailed) {
    SimulatorConfig a, b;
    ArchConfYamlReader(simple).parse(a);
    ArchConfYamlReader(detailed).parse(b);
    require(a.memory.hbm.backend == HBMBackend::Naive && b.memory.hbm.backend == HBMBackend::Naive,
            "HBM must stay Simple in this comparison");
    require(a.memory.hbf.backend == HBFBackend::Naive && b.memory.hbf.backend == HBFBackend::BankPlane,
            "HBF selection mismatch");
    b.memory.hbf.backend = HBFBackend::Naive;
    require(glz::write_json(a).value() == glz::write_json(b).value(), "matched profiles differ beyond HBF selection");
    b.memory.hbf.backend = HBFBackend::BankPlane;
    const auto& c = b.memory.hbf.controller;
    require(c.dies_per_stack == 8 && c.banks_per_channel == 16 && c.planes_per_bank == 1 && c.stacks == 2,
            "incorrect 8-Hi geometry");
    require(hbfine::capacity_bytes_per_stack(c) == (256ULL << 30), "8-Hi capacity mismatch");
    require(std::abs(hbf_array_read_ceiling_GBps(c) - 262.144) < 1e-9, "8-Hi sense ceiling mismatch");
    require(std::abs(hbf_array_write_ceiling_GBps(c) - 13.9810133333333) < 1e-8,
            "8-Hi program ceiling mismatch");
    auto device = make_hbf_backend(b.memory.hbf, b.core_frequency_mhz, b.memory.address_layout.hbf_base);
    device->preload_region(b.memory.address_layout.hbf_base, b.memory.address_layout.hbf_capacity_bytes, true);
    require(device->try_submit({0, b.memory.address_layout.hbf_base, 128, MemoryOperation::Read, MemoryTier::HBF}, 0),
            "cold read rejected");
    require(device->step(5651).empty(), "cold response preceded 4us sense plus two physical channel beats");
    require(device->step(5652).size() == 1, "cold read exceeded expected sense plus transfer time");
    auto stats = device->hbf_statistics();
    require(stats.senses == 1 && stats.sensed_bytes == 4096 && stats.host_read_bytes == 128,
            "page/host traffic distinction lost");
    require(stats.plane_reads.size() == 512 && stats.bank_reads.size() == 512,
            "8-Hi resources not scaled");
    require(device->try_submit({1, b.memory.address_layout.hbf_base, 128, MemoryOperation::Read, MemoryTier::HBF}, 5652),
            "page hit rejected");
    require(device->step(5664).size() == 1 && device->hbf_statistics().senses == 1,
            "page-buffer hit incurred a second sense");
    // Synthetic write-only test supplies erase/timeout policy explicitly.
    // Production kernels remain immutable and issue zero HBF writes.
    auto writable = b.memory.hbf;
    writable.controller.allow_program = true;
    writable.controller.erase_latency_us = 10;
    writable.controller.write_timeout_us = 10;
    auto writer = make_hbf_backend(writable, b.core_frequency_mhz, b.memory.address_layout.hbf_base);
    require(writer->try_submit({0, b.memory.address_layout.hbf_base, 4096, MemoryOperation::Write, MemoryTier::HBF}, 0),
            "synthetic complete-page program rejected");
    require(writer->step(120000).empty(), "program acknowledged before its physical operations completed");
    require(writer->step(130000).size() == 1, "synthetic program did not finish");
    stats = writer->hbf_statistics();
    require(stats.programs == 1 && stats.host_write_bytes == 4096 &&
            std::accumulate(stats.plane_program_busy_ps.begin(), stats.plane_program_busy_ps.end(), 0ULL) == 75000000,
            "program latency does not equal 75us");
    auto bad = b;
    ++bad.memory.hbf.read_latency_cycles;
    bool rejected = false;
    try { validate_memory_system_config(bad.memory, bad.core_frequency_mhz); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "physical and cycle latency mismatch accepted");
    std::cout << "Matched 8-Hi profiles: cold128B=5652 cycles, sense=4us, program=75us, per-stack ceilings=262.144/13.981013 GB/s\n";
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 3, "expected matched Simple and HBFine profiles");
        matched_profiles(argv[1], argv[2]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
