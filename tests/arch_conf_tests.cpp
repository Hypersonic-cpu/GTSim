#include "arch_conf.h"

#include <chrono>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

class TemporaryYaml {
public:
    explicit TemporaryYaml(const std::string& contents) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("gtsim-arch-conf-" + std::to_string(stamp) + ".yaml");
        std::ofstream output(path_);
        if (!output) throw std::runtime_error("cannot create temporary YAML test file");
        output << contents;
        if (!output) throw std::runtime_error("cannot write temporary YAML test file");
    }

    ~TemporaryYaml() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

template <typename Function>
void require_throws(Function&& function, const std::string& message) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        static_assert(glz::reflectable<GTSim::SimulatorConfig>);
        require(argc == 6, "expected the architecture YAML and four memory YAML paths");

        auto config = GTSim::make_h100_sxm_variants(argv[1]);
        require(config.num_sms == 132, "sm_count was not mapped to num_sms");
        require(config.core_frequency_mhz == 1980.0, "core frequency was not loaded");
        require(config.sram_bandwidth_bytes_per_cycle == 128, "SRAM bandwidth was not loaded");
        require(config.sram_latency_cycles == 33, "SRAM latency was not loaded");
        require(config.l2_bandwidth_bytes_per_cycle == 4300, "L2 bandwidth was not loaded");
        require(config.l2_hit_latency_cycles == 380, "L2 latency was not mapped");
        require(config.memory.topology == GTSim::MemoryTopology::FullHBM,
                "default memory topology is not FullHBM");
        require(config.tma_issue_rate_bytes_per_cycle == 100, "TMA issue rate was not loaded");
        require(config.tma_setup_latency_cycles == 100, "TMA setup latency was not loaded");
        require(config.tma_issue_interval_cycles == 1, "TMA issue interval was not mapped");
        require(config.l2_write_bandwidth_bytes_per_cycle == 2150,
                "omitted patch field did not preserve the base profile");
        require(config.tensor_core_throughput == 128,
                "descriptive hardware metadata unexpectedly changed simulator throughput");

        const std::array<GTSim::MemoryTopology, 4> expected_topologies{
            GTSim::MemoryTopology::FullHBM, GTSim::MemoryTopology::FullHBF,
            GTSim::MemoryTopology::CoLocated, GTSim::MemoryTopology::Cascaded};
        const std::array<double, 4> expected_hbm_read_bw{3100.0, 0.0,
                                                          1240.0, 1240.0};
        const std::array<double, 4> expected_hbf_read_bw{0.0, 2621.44,
                                                          1572.864, 1572.864};
        for (int index = 2; index < argc; ++index) {
            auto memory_patch = config;
            GTSim::ArchConfYamlReader(argv[index]).parse(memory_patch);
            const auto profile_index = static_cast<std::size_t>(index - 2);
            require(memory_patch.memory.topology ==
                        expected_topologies[profile_index],
                    "memory topology YAML enum did not parse");
            require(memory_patch.memory.hbm.read_bw_GBps ==
                        expected_hbm_read_bw[profile_index] &&
                        memory_patch.memory.hbf.read_bw_GBps ==
                            expected_hbf_read_bw[profile_index],
                    "memory controller parameters were not loaded");
        }
        GTSim::ArchConfYamlReader(argv[2]).parse(config);
        require(config.memory.hbm.write_max_bw_GBps == 1550.0,
                "HBM maximum write bandwidth was not loaded");
        require(config.memory.hbf.write_latency_cycles == 148500,
                "HBF write latency was not loaded");

        TemporaryYaml partial(R"yaml(
name: test patch
source: unit test
sm_count: 8
l2:
  bandwidth_bytes_per_cycle: 2048
dram:
  bandwidth_GBps: 900.5
  latency_cycles: 42
scheduler:
  startup_delay_cycles: 12
)yaml");
        auto patched = GTSim::make_h100_sxm_config();
        patched.l2_write_bandwidth_bytes_per_cycle = 512;
        GTSim::ArchConfYamlReader(partial.path()).parse(patched);
        require(patched.num_sms == 8, "partial patch did not override sm_count");
        require(patched.l2_bandwidth_bytes_per_cycle == 2048,
                "nested L2 patch did not override bandwidth");
        require(patched.memory.hbm.read_bw_GBps == 900.5 &&
                    patched.memory.hbm.write_max_bw_GBps == 900.5,
                "partial patch did not preserve floating-point values");
        require(patched.memory.hbm.read_latency_cycles == 42 &&
                    patched.memory.hbm.write_latency_cycles == 42,
                "legacy DRAM latency did not map to HBM latency");
        require(patched.startup_delay_cycles == 12, "scheduler field was not mapped");
        require(patched.l2_write_bandwidth_bytes_per_cycle == 512,
                "omitted field did not preserve the caller's value");

        TemporaryYaml typed_patch(R"yaml(
schedule_policy: rr
workload_type: fa3
noc_enabled: true
noc_topology: mesh
noc_mesh_sm_mapping: [0, 2, 4]
tmem:
  read_latency_cycles: 18
  write_latency_cycles: 23
)yaml");
        GTSim::ArchConfYamlReader(typed_patch.path()).parse(patched);
        require(patched.schedule_policy == GTSim::SchedulePolicy::RR,
                "enum field was not deserialized");
        require(patched.workload_type == GTSim::WorkloadType::FA3,
                "workload enum field was not deserialized");
        require(patched.noc_enabled, "boolean field was not deserialized");
        require(patched.noc_topology == GTSim::NoCTopology::MESH,
                "NoC topology enum was not deserialized");
        require(patched.noc_mesh_sm_mapping == std::vector<int>({0, 2, 4}),
                "vector field was not deserialized");
        require(patched.tmem_read_latency_cycles == 18 &&
                    patched.tmem_write_latency_cycles == 23,
                "nested TMEM fields were not deserialized");

        TemporaryYaml memory_override(R"yaml(
topology: CoLocated
hbm:
  read_bw_GBps: 2500.5
hbf:
  read_bw_GBps: 1572.864
  write_bw_GBps: 83.88
d2dlink:
  gpu_to_hbm:
    bandwidth_GBps: 1000
)yaml");
        auto sequential = GTSim::make_h100_sxm_config();
        GTSim::ArchConfYamlReader(argv[2]).parse(sequential);
        GTSim::ArchConfYamlReader(memory_override.path()).parse(sequential);
        require(sequential.memory.topology == GTSim::MemoryTopology::CoLocated,
                "topology patch did not override the previous memory config");
        require(sequential.memory.hbm.read_bw_GBps == 2500.5 &&
                    sequential.memory.hbm.write_max_bw_GBps == 1550.0,
                "sequential memory patch reset unspecified HBM fields");
        require(sequential.memory.hbf.read_bw_GBps == 1572.864 &&
                    sequential.memory.hbf.write_bw_GBps == 83.88,
                "topology patch did not configure the newly active HBF tier");
        require(sequential.memory.gpu_to_hbm.bandwidth_GBps == 1000.0 &&
                    sequential.memory.gpu_to_hbm.latency_cycles == 10,
                "nested D2D link patch did not preserve omitted link fields");

        TemporaryYaml unknown("l2:\n  unknown_bandwidth: 7\n");
        const int old_l2_bandwidth = patched.l2_bandwidth_bytes_per_cycle;
        require_throws([&] { GTSim::ArchConfYamlReader(unknown.path()).parse(patched); },
                       "unknown YAML field was accepted");
        require(patched.l2_bandwidth_bytes_per_cycle == old_l2_bandwidth,
                "failed patch partially modified the configuration");

        TemporaryYaml duplicate("l2:\n  bandwidth_bytes_per_cycle: 7\n  bandwidth_bytes_per_cycle: 8\n");
        require_throws([&] { GTSim::ArchConfYamlReader(duplicate.path()).parse(patched); },
                       "duplicate YAML key was accepted");
        require(patched.l2_bandwidth_bytes_per_cycle == old_l2_bandwidth,
                "duplicate-key failure partially modified the configuration");

        TemporaryYaml alias_duplicate("sm_count: 8\nnum_sms: 9\n");
        require_throws([&] { GTSim::ArchConfYamlReader(alias_duplicate.path()).parse(patched); },
                       "duplicate assignment through aliases was accepted");

        TemporaryYaml partial_failure("sm_count: 99\nsram:\n  latency_cycles: invalid\n");
        const int old_sm_count = patched.num_sms;
        require_throws([&] { GTSim::ArchConfYamlReader(partial_failure.path()).parse(patched); },
                       "invalid later field was accepted");
        require(patched.num_sms == old_sm_count,
                "type failure partially modified the configuration");

        TemporaryYaml wrong_type("sram:\n  latency_cycles: fast\n");
        require_throws([&] { GTSim::ArchConfYamlReader(wrong_type.path()).parse(patched); },
                       "wrong YAML field type was accepted");

        TemporaryYaml invalid_enum("schedule_policy: random\n");
        require_throws([&] { GTSim::ArchConfYamlReader(invalid_enum.path()).parse(patched); },
                       "invalid enum value was accepted");

        TemporaryYaml invalid_topology("topology: SomeOtherTopology\n");
        require_throws([&] { GTSim::ArchConfYamlReader(invalid_topology.path()).parse(patched); },
                       "invalid memory topology was accepted");

        TemporaryYaml invalid_bandwidth(
            "topology: CoLocated\nhbf:\n  read_bw_GBps: 0\n");
        require_throws([&] { GTSim::ArchConfYamlReader(invalid_bandwidth.path()).parse(patched); },
                       "invalid memory bandwidth was accepted");

        TemporaryYaml unknown_memory("hbm:\n  bandwidth_typo_GBps: 7\n");
        require_throws([&] { GTSim::ArchConfYamlReader(unknown_memory.path()).parse(patched); },
                       "unknown memory field was accepted");

        TemporaryYaml invalid_backend("hbm:\n  backend: typo\n");
        require_throws([&] { GTSim::ArchConfYamlReader(invalid_backend.path()).parse(patched); },
                       "unknown HBM backend was accepted");
        TemporaryYaml incomplete_ramulator("topology: FullHBM\nhbm:\n  backend: ramulator2\n");
        require_throws([&] { GTSim::ArchConfYamlReader(incomplete_ramulator.path()).parse(patched); },
                       "Ramulator without an explicit timing profile was accepted");

        TemporaryYaml malformed("l2: [\n");
        require_throws([&] { GTSim::ArchConfYamlReader(malformed.path()).parse(patched); },
                       "malformed YAML was accepted");

        TemporaryYaml empty_unknown("not_a_section: {}\n");
        require_throws([&] { GTSim::ArchConfYamlReader(empty_unknown.path()).parse(patched); },
                       "unknown empty mapping was accepted");

        std::cout << "Architecture config parser tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Architecture config test failure: " << error.what() << '\n';
        return 1;
    }
}
