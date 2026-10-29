#define main gemm_cli_main
#include "../workloads/gemm/gemm_h100.cpp"
#undef main
#define main fa3_cli_main
#include "../workloads/fa3/fa3_h100.cpp"
#undef main
#include <memory>
#include <cmath>
#include <queue>

void require(bool value, const char* why) {
    if (!value) throw std::runtime_error(why);
}
void check_graph(DAG& dag, long long macs) {
    long long actual = 0;
    std::vector<int> indegree(dag.nodes.size());
    std::vector<std::vector<int>> children(dag.nodes.size());
    for (auto* n : dag.nodes) {
        if (n->op_type == OpType::WGMMA) actual += 16LL * 64 * 16;
        for (const auto* deps : {&n->depends_on, &n->issue_depends_on})
            for (int dep : *deps) {
                require(dep >= 0 && dep < static_cast<int>(dag.nodes.size()), "bad dependency identity");
                ++indegree[n->id]; children[dep].push_back(n->id);
            }
        if (n->op_type == OpType::CP_DRAM2SRAM_TMA || n->op_type == OpType::CP_SRAM2DRAM_TMA ||
            n->op_type == OpType::LD_DRAM2REG || n->op_type == OpType::ST_REG2DRAM) {
            const auto& t = dag.tensor_registry.at(n->matrix_id);
            require(n->data_type == t.dtype, "global access dtype mismatch");
            require(n->tile.r > 0 && n->tile.c > 0, "empty global operation");
            require(n->tile.r_off >= 0 && n->tile.c_off >= 0 &&
                n->tile.r_off + n->tile.r <= static_cast<int>(t.shape[0]) &&
                n->tile.c_off + n->tile.c <= static_cast<int>(t.shape[1]), "global access outside registered tensor");
        }
    }
    require(actual == macs, "tile/split changes arithmetic work");
    std::queue<int> ready;
    for (std::size_t i=0;i<indegree.size();++i) if(!indegree[i]) ready.push(static_cast<int>(i));
    std::size_t count=0;
    while(!ready.empty()) {
        int id=ready.front(); ready.pop(); ++count;
        for(int child:children[id]) if(--indegree[child]==0) ready.push(child);
    }
    require(count == dag.nodes.size(), "dependency cycle");
}
int main() {
    auto hardware = make_h100_sxm_config();
    for (int bm : {64,128}) for(int bn:{128,256}) for(int bk:{32,64,128}) {
        KernelOptions o; o.bm=bm; o.bn=bn; o.bk=bk;
        bool fits=o.resources(false).shared_bytes<=hardware.shared_memory_per_cta_bytes;
        try {
            std::unique_ptr<DAG> d(build_gemm_h100(128,256,1024,o,hardware));
            require(fits,"accepted oversized GEMM tile");
            check_graph(*d,128LL*256*1024);
        } catch(const std::invalid_argument&) { require(!fits,"rejected feasible GEMM tile"); }
    }
    {
        KernelOptions o; o.bn=128;
        std::unique_ptr<DAG> d(build_gemm_h100(128,1536,512,o,hardware));
        check_graph(*d,128LL*1536*512);
    }
    for(int split:{2,4,8}) {
        KernelOptions o; o.split=split;
        std::unique_ptr<DAG> g(build_gemm_h100(128,256,1024,o,hardware));
        check_graph(*g,128LL*256*1024);
        bool nonblocking_store=false;
        for(auto* n:g->nodes) if(n->op_type==OpType::LD_DRAM2REG) {
            for(int id:n->issue_depends_on) if(g->nodes[id]->op_type==OpType::ST_REG2DRAM) {
                require(std::find(n->depends_on.begin(),n->depends_on.end(),id)==n->depends_on.end(),"register reuse waits for memory store acknowledgement");
                nonblocking_store=true;
            }
        }
        require(nonblocking_store,"merge store issue-order edge missing");
        std::unique_ptr<DAG> f(build_fa3_h100(128,1024,2,1,1,o,hardware));
        check_graph(*f,2LL*2*128*1024*128);
        bool merge=false;
        for(auto*n:f->nodes) if(n->op=="split_normalize") merge=true;
        require(merge,"split attention has no charged merge");
    }
    {
        KernelOptions o; o.split=8;
        std::unique_ptr<DAG> d(build_fa3_h100(128,1536,2,1,1,o,hardware));
        check_graph(*d,2LL*2*128*1536*128);
        auto* tail=d->nodes_by_name.at("TB7_K_Load0_k1");
        require(tail->tile.c==64 && tail->tile.c_off==1472,"partition tail is not bounded correctly");
    }
    for(int q:{64,128}) for(int kv:{64,128,256}) for(int stages:{2,3,4}) {
        KernelOptions o; o.q_tile=q; o.kv_tile=kv; o.stages=stages;
        bool fits=o.resources(true).shared_bytes<=hardware.shared_memory_per_cta_bytes;
        try {
            std::unique_ptr<DAG> d(build_fa3_h100(128,1024,2,1,1,o,hardware));
            require(fits,"accepted oversized pipeline");
            check_graph(*d,2LL*2*128*1024*128);
        } catch(const std::invalid_argument&) { require(!fits,"rejected feasible tile/pipeline"); }
    }
    for(int stages:{3,4}) {
        KernelOptions o; o.bm=64;o.bn=128;o.stages=stages;
        std::unique_ptr<DAG> d(build_gemm_h100(128,256,1024,o,hardware));
        check_graph(*d,128LL*256*1024);
        auto* load=d->nodes_by_name.at("TB0_Tile0_TMA_LoadA_k"+std::to_string(stages));
        bool reuse=false;
        for(int id:load->depends_on) if(d->nodes[id]->name.find("_k0_")!=std::string::npos) reuse=true;
        require(reuse,"GEMM buffer is reused without consumer completion");
    }
    {
        KernelOptions o; o.bm=64; o.bn=128; o.order="m-major";
        std::unique_ptr<DAG> d(build_gemm_h100(128,256,256,o,hardware));
        auto* ld=d->nodes_by_name.at("TB1_Tile1_TMA_LoadA_k0");
        require(ld->sm_id==1 && ld->tile.r_off==64,"GEMM order does not change execution/SM mapping");
        auto* next=d->nodes_by_name.at("TB0_Tile0_TMA_LoadA_k1");
        require(!next->issue_depends_on.empty(),"TMA issue order missing");
    }
    {
        KernelOptions o; o.q_tile=64;o.kv_tile=64;o.stages=3;
        std::unique_ptr<DAG> d(build_fa3_h100(64,512,1,1,1,o,hardware));
        auto* k=d->nodes_by_name.at("TB0_K_Load0_k3");
        auto* v=d->nodes_by_name.at("TB0_V_Load0_k3");
        for(auto* load:{k,v}) {
            bool found=false;
            for(int id:load->depends_on) {
                auto& name=d->nodes[id]->name;
                if(name.find(load==k ? "_WGMMA_SS_" : "_WGMMA_RS_")!=std::string::npos &&
                   name.find("_k0_")!=std::string::npos) found=true;
            }
            require(found,"K/V lifetime not protected independently");
        }
    }
    for(int stages:{2,4}) {
        auto constrained=hardware; constrained.num_sms=1; constrained.max_concurrent_blocks_per_sm=2; constrained.silence_mode=true;
        KernelOptions o;o.q_tile=64;o.kv_tile=64;o.stages=stages;
        std::unique_ptr<DAG> d(build_fa3_h100(128,256,2,1,1,o,constrained));
        Simulator sim(d.get(),constrained);
        auto* sm=sim.simulated_gpu->sms.front();
        require(sm->resident_tb_ids.size()==static_cast<std::size_t>(stages==2 ? 2 : 1),"resource footprints do not constrain occupancy");
        require(sm->shared_used<=sm->shared_capacity && sm->registers_used<=sm->register_capacity,"resident resource overflow");
        require(sm->waiting_tb_queue.size()==static_cast<std::size_t>(stages==2 ? 2 : 3),"non-resident CTAs must wait in FIFO order");
        require(sim.run(1000000,false),"queued resource-constrained CTAs did not complete");
        require(sm->waiting_tb_queue.empty() && sm->resident_tb_ids.empty(),"CTA queues did not drain");
        require(sm->shared_used==0 && sm->registers_used==0 && sm->warps_used==0,"CTA retirement did not release resources");
    }
    for(const std::string order:{"query-major","kv-head-grouped"}) {
        KernelOptions o;o.order=order;
        std::unique_ptr<DAG> d(build_fa3_h100(256,256,4,2,1,o,hardware));
        check_graph(*d,2LL*4*256*256*128);
        require(d->nodes_by_name.at("TB1_Q_Load0_b0_h1")->tile.r_off==0,"CTA order not reflected in TB identity");
    }
    for (int split : {1, 2, 8}) {
        KernelOptions o; o.bn=128; o.split=split; o.io_dtype=DataType::BF16;
        std::unique_ptr<DAG> d(build_gemm_h100(1,256,512,o,hardware));
        check_graph(*d,64LL*256*512);
        require(d->tensor_registry.at(0).initial_memory_tier == MemoryTier::HBM &&
                d->tensor_registry.at(1).initial_memory_tier == MemoryTier::HBF &&
                d->tensor_registry.at(2).initial_memory_tier == MemoryTier::HBM,
                "decode GEMM activation/weight/output tier mismatch");
        require(d->nodes_by_name.at("TB0_Tile0_TMA_LoadA_k0")->tile.r == 1,
                "single-token GEMM must load one live input row");
    }
    // Decode uses a masked m64 tile: only one live row is loaded/stored.
    for (int split : {1, 2, 8}) {
        KernelOptions o; o.kv_tile=64; o.head_dim=256;
        o.io_dtype=DataType::BF16; o.split=split;
        std::unique_ptr<DAG> d(build_fa3_h100(1,512,4,1,1,o,hardware));
        check_graph(*d,2LL*4*64*512*256);
        require(d->nodes_by_name.at("TB0_Q_Load0_b0_h0")->tile.r == 1,
                "decode query must load exactly one row");
    }
    std::vector<double> logits(32), values(32);
    for(int i=0;i<32;++i) { logits[i]=std::sin(i)*30; values[i]=std::cos(i); }
    for(int split:{1,2,4,8}) {
        std::vector<std::vector<double>> out;
        std::vector<double> lse;
        double full_sum=0,full_weighted=0,m=*std::max_element(logits.begin(),logits.end());
        for(int i=0;i<32;++i) { double w=std::exp(logits[i]-m);full_sum+=w;full_weighted+=w*values[i]; }
        for(int s=0;s<split;++s) {
            int begin=s*32/split,end=(s+1)*32/split;
            double local_m=*std::max_element(logits.begin()+begin,logits.begin()+end),sum=0,weighted=0;
            for(int i=begin;i<end;++i) { double w=std::exp(logits[i]-local_m);sum+=w;weighted+=w*values[i]; }
            out.push_back({weighted/sum});lse.push_back(local_m+std::log(sum));
        }
        require(std::abs(merge_attention(out,lse)[0]-full_weighted/full_sum)<1e-12,"split softmax normalization is incorrect");
    }
    std::cout << "kernel_swopt_tests passed\n";
}
