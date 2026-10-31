// Independent oracle utility. Production frontend never imports this builder.
#define GTSIM_WORKLOAD_LIBRARY
#include "../../workloads/gemm/gemm_h100.cpp"
#include "../../workloads/fa3/fa3_h100.cpp"
#include <memory>
#include <map>

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::invalid_argument("expected gemm M N K SMs or fa3 B Hq Hkv Q KV SMs");
        auto hardware = make_h100_sxm_config();
        KernelOptions options;
        auto memory = workload_memory_options(hardware);
        std::unique_ptr<DAG> dag;
        if (std::string(argv[1]) == "gemm" && argc == 6) {
            hardware.num_sms = std::stoi(argv[5]);
            dag.reset(build_gemm_h100(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), options, hardware, &memory));
        } else if (std::string(argv[1]) == "fa3" && argc == 8) {
            hardware.num_sms = std::stoi(argv[7]);
            dag.reset(build_fa3_h100(std::stoi(argv[5]), std::stoi(argv[6]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[2]), options, hardware, &memory));
        } else throw std::invalid_argument("invalid oracle arguments");
        std::map<int, CTAResources> ctas(dag->cta_resources.begin(), dag->cta_resources.end());
        for (const auto& [id, c] : ctas)
            std::cout << "cta " << id << ' ' << c.warps << ' ' << c.shared_bytes << ' ' << c.register_bytes << '\n';
        std::map<int, TensorMetadata> tensors(dag->tensor_registry.entries().begin(),dag->tensor_registry.entries().end());
        for (const auto& [id,t] : tensors) {
            std::cout << "tensor " << id << ' ' << t.name << ' ' << static_cast<int>(t.dtype) << ' ' << static_cast<int>(t.initial_memory_tier);
            for (auto d:t.shape) std::cout << ' ' << d;
            std::cout << " strides";
            for (auto d:t.strides) std::cout << ' ' << d;
            std::cout << '\n';
        }
        for (const auto* n : dag->nodes) {
            std::cout << "op " << n->id << ' ' << n->name << ' ' << n->pipeline_type << ' ' << n->op << ' '
                      << n->sm_id << ' ' << n->thread_block_id << ' ' << n->warp_id << ' ' << n->matrix_id << ' '
                      << static_cast<int>(n->data_type) << ' ' << static_cast<int>(n->layout) << ' '
                      << n->matrix_leading_dim << ' ' << n->tile.r_off << ' ' << n->tile.c_off << ' ' << n->tile.r << ' ' << n->tile.c;
            std::cout << " complete";
            for (auto d:n->depends_on) std::cout << ' ' << d;
            std::cout << " issue";
            for (auto d:n->issue_depends_on) std::cout << ' ' << d;
            std::cout << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
