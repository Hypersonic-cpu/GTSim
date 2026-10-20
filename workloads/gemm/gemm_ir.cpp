#include "arch_conf.h"
#include "tile_ir.h"
#include <iostream>
#include <glaze/json/write.hpp>

int main(int argc, char* argv[]) {
    if ((argc != 5 && argc != 6) || std::string(argv[2]) != "--memory-config") {
        std::cerr << "Usage: gtsim_gemm_ir architecture.yaml --memory-config memory.yaml kernel.gtsir [--inspect]\n";
        return 1;
    }
    try {
        auto config = GTSim::make_h100_sxm_variants(argv[1]);
        GTSim::ArchConfYamlReader(argv[3]).parse(config);
        if (std::string(argv[4]) == "--describe-config") {
            std::cout << GTSim::describe_tile_ir_hardware(config) << '\n';
            return 0;
        }
        if (argc == 6 && std::string(argv[5]) != "--inspect")
            throw std::invalid_argument("unknown inspection option");
        auto ir = GTSim::read_tile_ir(argv[4]);
        auto dag = GTSim::materialize_tile_ir(ir, config);
        if (argc == 6) {
            // Inspect the materialized graph, including the addresses consumed
            // by L2, without simulating or changing the hardware configuration.
            std::cout << "{\"tensors\":[";
            bool first = true;
            for (const auto& input : ir.tensors) {
                const auto& tensor = dag->tensor_registry.at(input.id);
                if (!first) std::cout << ',';
                first = false;
                std::cout << "{\"id\":" << tensor.id << ",\"name\":"
                          << glz::write_json(tensor.name).value()
                          << ",\"base\":" << tensor.global_base_address
                          << ",\"bytes\":" << tensor.byte_size()
                          << ",\"device\":\""
                          << (tensor.initial_memory_tier == GTSim::MemoryTier::HBM ? "HBM" : "HBF") << "\"}";
            }
            std::cout << "],\"ctas\":[";
            first = true;
            for (const auto& cta : ir.ctas) {
                if (!first) std::cout << ',';
                first = false;
                std::cout << '[' << cta.id << ',' << cta.sm << ',' << cta.resources.warps
                          << ',' << cta.resources.shared_bytes << ',' << cta.resources.register_bytes << ']';
            }
            std::cout << "],\"nodes\":[";
            first = true;
            for (const auto* node : dag->nodes) {
                if (!first) std::cout << ',';
                first = false;
                std::cout << "{\"id\":" << node->id << ",\"name\":" << glz::write_json(node->name).value()
                          << ",\"op\":" << glz::write_json(node->op).value()
                          << ",\"pipeline\":" << glz::write_json(node->pipeline_type).value()
                          << ",\"warp\":" << node->warp_id << ",\"sm\":" << node->sm_id
                          << ",\"cta\":" << node->thread_block_id << ",\"tensor\":" << node->matrix_id
                          << ",\"leading_dim\":" << node->matrix_leading_dim
                          << ",\"dtype\":" << static_cast<int>(node->data_type)
                          << ",\"region\":[" << node->tile.r_off << ',' << node->tile.c_off
                          << ',' << node->tile.r << ',' << node->tile.c << ']'
                          << ",\"after_complete\":" << glz::write_json(node->depends_on).value()
                          << ",\"after_issue\":" << glz::write_json(node->issue_depends_on).value() << '}';
            }
            std::cout << "]}\n";
            return 0;
        }
        std::cout << "Generated DAG with " << dag->nodes.size() << " nodes\n";
        config.silence_mode = true;
        GTSim::Simulator simulator(dag.get(), config);
        return simulator.run(100000000, true) ? 0 : 3;
    } catch (const std::exception& e) {
        std::cerr << "Tile IR error: " << e.what() << '\n';
        return 2;
    }
}
