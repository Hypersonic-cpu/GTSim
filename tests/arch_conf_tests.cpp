#include "arch_conf.h"

#include <chrono>
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
        require(argc == 2, "expected the H100 YAML path");

        const auto config = GTSim::make_h100_sxm_variants(argv[1]);
        require(config.num_sms == 132, "sm_count was not mapped to num_sms");
        require(config.core_frequency_mhz == 1980.0, "core frequency was not loaded");
        require(config.dram_frequency_mhz == 2619.0, "memory frequency was not mapped");
        require(config.sram_bandwidth_bytes_per_cycle == 128, "SRAM bandwidth was not loaded");
        require(config.sram_latency_cycles == 33, "SRAM latency was not loaded");
        require(config.l2_bandwidth_bytes_per_cycle == 4300, "L2 bandwidth was not loaded");
        require(config.l2_hit_latency_cycles == 380, "L2 latency was not mapped");
        require(config.dram_bandwidth_gbps == 3100.0, "DRAM bandwidth was not loaded");
        require(config.l2_miss_penalty_cycles == 270, "DRAM latency was not mapped");
        require(config.tma_issue_rate_bytes_per_cycle == 100, "TMA issue rate was not loaded");
        require(config.tma_setup_latency_cycles == 100, "TMA setup latency was not loaded");
        require(config.tma_issue_interval_cycles == 1, "TMA issue interval was not mapped");
        require(config.l2_write_bandwidth_bytes_per_cycle == 2150,
                "omitted patch field did not preserve the base profile");
        require(config.tensor_core_throughput == 128,
                "descriptive hardware metadata unexpectedly changed simulator throughput");

        TemporaryYaml partial(R"yaml(
name: test patch
source: unit test
sm_count: 8
l2:
  bandwidth_bytes_per_cycle: 2048
dram:
  bandwidth_gbps: 900.5
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
        require(patched.dram_bandwidth_gbps == 900.5,
                "partial patch did not preserve floating-point values");
        require(patched.l2_miss_penalty_cycles == 42,
                "nested DRAM latency did not map to miss penalty");
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
