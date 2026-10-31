#include "arch_conf.h"
#include "runtime/execution_program.h"
#include <glaze/json/write.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 5 || std::string(argv[2]) != "--memory-config") {
        std::cerr << "Usage: gtsim_program architecture.yaml --memory-config "
                     "memory.yaml program.gtsir|--describe-config\n";
        return 1;
    }
    try {
        auto config = GTSim::make_h100_sxm_variants(argv[1]);
        GTSim::ArchConfYamlReader(argv[3]).parse(config);
        if (std::string(argv[4]) == "--describe-config") {
            std::cout << "{\"num_sms\":" << config.num_sms
                      << ",\"shared_memory_per_cta_bytes\":"
                      << config.shared_memory_per_cta_bytes
                      << ",\"shared_memory_per_sm_bytes\":"
                      << config.shared_memory_per_sm_bytes
                      << ",\"register_file_per_sm_bytes\":"
                      << config.register_file_per_sm_bytes
                      << ",\"max_warps_per_sm\":" << config.max_warps_per_sm
                      << "}\n";
            return 0;
        }
        config.silence_mode = true;
        const std::filesystem::path file = argv[4];
        auto program = GTSim::ExecutionIR::read_program(file);
        auto timing =
            GTSim::ExecutionIR::execute(program, file.parent_path(), config);
        std::cout << "Program timing JSON: " << glz::write_json(timing).value()
                  << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Execution program error: " << e.what() << '\n';
        return 2;
    }
}
