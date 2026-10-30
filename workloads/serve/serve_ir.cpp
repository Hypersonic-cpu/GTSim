#include "arch_conf.h"
#include "tile_ir.h"
#include <iostream>
#include <map>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: gtsim_serve_ir architecture-and-memory.yaml kernel.gtsir|--describe-config\n";
        return 1;
    }
    try {
        GTSim::SimulatorConfig config;
        GTSim::ArchConfYamlReader(argv[1]).parse(config);
        if (std::string(argv[2]) == "--describe-config") {
            std::cout << GTSim::describe_tile_ir_hardware(config) << '\n';
            return 0;
        }
        auto ir = GTSim::read_tile_ir(argv[2]);
        auto dag = GTSim::materialize_tile_ir(ir, config);
        std::cout << "Generated DAG with " << dag->nodes.size() << " nodes\n";
        config.silence_mode = true;
        GTSim::Simulator simulator(dag.get(), config);
        const bool complete = simulator.run(100000000, true);
        std::map<std::string, std::pair<int, int>> phases;
        for (const auto* node : dag->nodes) {
            const auto separator = node->name.find("__");
            if (separator == std::string::npos) continue;
            auto name = node->name.substr(0, separator);
            auto [position, inserted] = phases.try_emplace(name, node->start, node->end);
            if (!inserted) {
                position->second.first = std::min(position->second.first, node->start);
                position->second.second = std::max(position->second.second, node->end);
            }
        }
        for (const auto& [name, times] : phases)
            std::cout << "Stage timing: " << name << " first " << times.first << " complete " << times.second << '\n';
        return complete ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << "Serving Tile IR error: " << error.what() << '\n';
        return 2;
    }
}
