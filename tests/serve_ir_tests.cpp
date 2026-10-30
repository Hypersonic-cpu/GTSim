#include "arch_conf.h"
#include "tile_ir.h"
#include <iostream>
#include <set>
#include <sstream>

using namespace GTSim;

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    try {
        std::ostringstream text;
        text << "gtsim.tile_ir 2\n"
             << "tensor id=0 name=X shape=[8,16] strides=[16,1] dtype=bf16 tier=HBM\n"
             << "tensor id=1 name=W shape=[8,16] strides=[16,1] dtype=bf16 tier=HBF\n"
             << "tensor id=2 name=Y shape=[8,16] strides=[16,1] dtype=bf16 tier=HBM\n";
        for (int cta = 0; cta < 8; ++cta)
            text << "cta id=" << cta << " sm=0 warps=1 shared=512 registers=2048\n";
        for (int cta = 0; cta < 8; ++cta) {
            const int base = cta * 4;
            text << '%' << base << " = ld.dram2reg name=load" << cta << " cta=" << cta
                 << " warp=0 tensor=1 dtype=bf16 tile=[" << cta
                 << ",0,1,16] after_issue=[] after_complete=[] granularity=16 layout=row\n";
            text << '%' << base + 1 << " = ld.sram2reg name=shared" << cta << " cta=" << cta
                 << " warp=0 tensor=0 dtype=bf16 tile=[0,0,16,16] after_issue=[] after_complete=["
                 << base << "] granularity=16 layout=swizzled\n";
            text << '%' << base + 2 << " = mma.sync.m16n8k16 name=mma" << cta << " cta=" << cta
                 << " warp=0 tensor=0 dtype=fp32 tile=[0,0,16,8] after_issue=[] after_complete=["
                 << base + 1 << "] granularity=16 layout=row\n";
            text << '%' << base + 3 << " = st.reg2dram name=store" << cta << " cta=" << cta
                 << " warp=0 tensor=2 dtype=bf16 tile=[" << cta << ",0,1,16] after_issue=[] after_complete=["
                 << base + 2 << "] granularity=16 layout=row\n";
        }
        std::istringstream input(text.str());
        const auto ir = parse_tile_ir(input);
        auto config = make_h100_sxm_config();
        config.memory.topology = MemoryTopology::CoLocated;
        const auto dag = materialize_tile_ir(ir, config);
        std::set<int> identities;
        for (int cta = 0; cta < 8; ++cta) {
            const auto* global = dag->nodes[cta * 4];
            const auto* shared = dag->nodes[cta * 4 + 1];
            const auto* mma = dag->nodes[cta * 4 + 2];
            require(global->warp_id % 4 == cta % 4, "one-warp CTAs do not rotate across SM subpartitions");
            require(identities.insert(global->warp_id).second, "CTA warp identities alias");
            require(global->data_type == DataType::BF16 && global->memory_access_granularity_bytes == 16,
                    "BF16 vector load metadata lost");
            require(dag->tensor_registry.at(global->matrix_id).initial_memory_tier == MemoryTier::HBF,
                    "weight tier lost");
            require(shared->matrix_id == -1 && shared->layout == Layout::Swizzled && shared->matrix_leading_dim == 0,
                    "shared operand uses a global tensor address/stride");
            require(mma->op_type == OpType::MMA && mma->issue_group_kind == IssueGroupKind::NONE,
                    "Ampere warp MMA became a warpgroup instruction");
        }
        auto reject = [&](auto mutation) {
            auto changed = ir;
            mutation(changed);
            try { materialize_tile_ir(changed, config); }
            catch (const std::exception&) { return; }
            throw std::runtime_error("invalid serving IR was accepted");
        };
        reject([](auto& program) { program.operations[0].region = Tile(8, 0, 1, 16); });
        reject([](auto& program) { program.operations[0].granularity_bytes = 32; });
        reject([](auto& program) { program.operations[0].op = "unknown.compute"; });
        reject([](auto& program) { program.operations[0].op = "cp.dram2sram_tma"; });
        reject([](auto& program) { program.operations[2].region = Tile(0, 0, 16, 16); });
        reject([](auto& program) { program.ctas[0].resources.warps = 0; });
        reject([](auto& program) { program.ctas[0].resources.shared_bytes = 128; });
        std::cout << "Serving IR BF16, tiers, Ampere MMA, SRAM and warp mapping checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
