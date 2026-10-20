#include "arch_conf.h"
#include "tile_ir.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 5 || std::string(argv[2]) != "--memory-config") {
        std::cerr << "Usage: gtsim_gemm_ir architecture.yaml --memory-config memory.yaml kernel.gtsir\n";
        return 1;
    }
    try {
        auto config = GTSim::make_h100_sxm_variants(argv[1]);
        GTSim::ArchConfYamlReader(argv[3]).parse(config);
        if (std::string(argv[4]) == "--describe-config") {
            std::cout << GTSim::describe_tile_ir_hardware(config) << '\n';
            return 0;
        }
        auto dag = GTSim::materialize_tile_ir(GTSim::read_tile_ir(argv[4]), config);
        std::cout << "Generated DAG with " << dag->nodes.size() << " nodes\n";
        config.silence_mode = true;
        GTSim::Simulator simulator(dag.get(), config);
        return simulator.run(100000000, true) ? 0 : 3;
    } catch (const std::exception& e) {
        std::cerr << "Tile IR error: " << e.what() << '\n';
        return 2;
    }
}
