#define GTSIM_WORKLOAD_LIBRARY
#include GTSIM_OLD_SOURCE
#include "../../workloads/gemm/gemm_h100.cpp"
#include <sstream>
#include <memory>
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    auto config = make_h100_sxm_variants(argv[1]);
    ArchConfYamlReader(argv[2]).parse(config);
    config.num_sms = 2;
    config.silence_mode = true;
    auto memory = workload_memory_options(config);
    for (int k : {64, 128, 192, 256}) {
        std::unique_ptr<DAG> old(
            build_gemm_before(512, 512, k, {}, config, &memory));
        std::unique_ptr<DAG> now(
            build_gemm_h100(512, 512, k, {}, config, &memory));
        if (old->nodes.size() != now->nodes.size()) return 3;
        int additions = 0;
        for (std::size_t i = 0; i < old->nodes.size(); ++i) {
            auto* a = old->nodes[i];
            auto* b = now->nodes[i];
            if (a->name != b->name || a->op != b->op ||
                a->issue_depends_on != b->issue_depends_on)
                return 4;
            for (int d : a->depends_on)
                if (std::find(b->depends_on.begin(), b->depends_on.end(), d) ==
                    b->depends_on.end())
                    return 5;
            additions += int(b->depends_on.size()) - int(a->depends_on.size());
        }
        int before = 0, after = 0;
        {
            Simulator s(old.get(), config);
            std::ostringstream sink;
            s.output_stream = &sink;
            if (!s.run(100000000, false)) return 6;
            before = s.cycle;
        }
        {
            Simulator s(now.get(), config);
            std::ostringstream sink;
            s.output_stream = &sink;
            if (!s.run(100000000, false)) return 7;
            after = s.cycle;
        }
        std::cout << "K=" << k << " old=" << before << " new=" << after
                  << " added_completion_edges=" << additions << "\n";
    }
}
