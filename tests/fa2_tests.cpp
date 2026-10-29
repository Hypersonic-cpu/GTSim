#define GTSIM_FA2_NO_MAIN
#include "../workloads/fa2/fa2_a100.cpp"
#include <cmath>
#include <queue>

static void require(bool value,const char* why) {
    if(!value)throw std::runtime_error(why);
}
static void inspect(DAG& dag,long long expected_macs) {
    long long macs=0;
    std::vector<int> degree(dag.nodes.size());
    std::vector<std::vector<int>> children(dag.nodes.size());
    for(auto* n:dag.nodes) {
        require(n->op_type!=OpType::WGMMA && n->op_type!=OpType::CP_DRAM2SRAM_TMA &&
                n->op_type!=OpType::CP_SRAM2DRAM_TMA,"FA2 contains Hopper instructions");
        if(n->op_type==OpType::MMA) {
            require(n->issue_group_size<=1,"FA2 MMA is warp-group coupled");
            macs+=16*8*16;
        }
        for(const auto* deps:{&n->depends_on,&n->issue_depends_on})for(int dep:*deps) {
            require(dep>=0 && dep<static_cast<int>(dag.nodes.size()),"bad dependency");
            ++degree[n->id];children[dep].push_back(n->id);
        }
        if(n->op_type==OpType::CP_DRAM2SRAM || n->op_type==OpType::LD_DRAM2REG || n->op_type==OpType::ST_REG2DRAM) {
            const auto& t=dag.tensor_registry.at(n->matrix_id);
            require(n->data_type==t.dtype,"global dtype mismatch");
            require(n->tile.r>0 && n->tile.c>0 && n->tile.r_off>=0 && n->tile.c_off>=0 &&
                    n->tile.r_off+n->tile.r<=static_cast<int>(t.shape[0]) &&
                    n->tile.c_off+n->tile.c<=static_cast<int>(t.shape[1]),"global access out of bounds");
        }
    }
    require(macs==expected_macs,"FA2 padded MMA work mismatch");
    std::queue<int> ready;
    for(std::size_t i=0;i<degree.size();++i)if(!degree[i])ready.push(static_cast<int>(i));
    std::size_t count=0;
    while(!ready.empty()) {
        int i=ready.front();ready.pop();++count;
        for(int j:children[i])if(--degree[j]==0)ready.push(j);
    }
    require(count==dag.nodes.size(),"FA2 dependency cycle");
}
// Independent FP64 recurrence check: online block-softmax and LSE-weighted
// split merge must equal dense GQA attention, including empty/tail partitions.
static void numerical_reference() {
    constexpr int hq=4,hkv=2,seq=257,dim=8;
    auto q=[](int h,int d){return std::sin(h*7+d)*9;};
    auto k=[](int h,int t,int d){return std::cos(h*5+t*.3+d);};
    auto v=[](int h,int t,int d){return std::sin(h*3+t*.7+d);};
    for(int splits:{1,3,47})for(int h=0;h<hq;++h) {
        int kh=h/(hq/hkv),blocks=(seq+63)/64,per=(blocks+splits-1)/splits;
        std::vector<double> score(seq),dense(dim,0);
        for(int t=0;t<seq;++t)for(int d=0;d<dim;++d)score[t]+=q(h,d)*k(kh,t,d)/std::sqrt(dim);
        double max=*std::max_element(score.begin(),score.end()),den=0;
        for(int t=0;t<seq;++t) {
            double e=std::exp(score[t]-max);den+=e;
            for(int d=0;d<dim;++d)dense[d]+=e*v(kh,t,d);
        }
        for(auto& x:dense)x/=den;
        std::vector<std::vector<double>> partial(splits,std::vector<double>(dim));
        std::vector<double> lse(splits,-INFINITY);
        for(int s=0;s<splits;++s) {
            double m=-INFINITY,l=0;
            for(int block=std::min(blocks,(s+1)*per)-1;block>=s*per;--block) {
                int begin=block*64,end=std::min(seq,begin+64);
                double nm=std::max(m,*std::max_element(score.begin()+begin,score.begin()+end));
                double alpha=std::exp(m-nm);l*=alpha;
                for(auto& x:partial[s])x*=alpha;
                for(int t=begin;t<end;++t) {
                    double e=std::exp(score[t]-nm);l+=e;
                    for(int d=0;d<dim;++d)partial[s][d]+=e*v(kh,t,d);
                }
                m=nm;
            }
            if(l>0) {lse[s]=m+std::log(l);for(auto& x:partial[s])x/=l;}
        }
        double lm=*std::max_element(lse.begin(),lse.end()),ls=0;
        std::vector<double> result(dim);
        for(int s=0;s<splits;++s) {
            double weight=std::exp(lse[s]-lm);ls+=weight;
            for(int d=0;d<dim;++d)result[d]+=weight*partial[s][d];
        }
        for(int d=0;d<dim;++d)require(std::abs(result[d]/ls-dense[d])<1e-11,"online/split GQA reference differs from dense");
    }
}
int main(int argc,char** argv) {
    if(argc!=2)return 1;
    SimulatorConfig cfg;ArchConfYamlReader(argv[1]).parse(cfg);
    require(cfg.num_sms==108 && cfg.shared_memory_per_sm_bytes==164*1024 &&
            cfg.shared_memory_per_cta_bytes==163*1024 && cfg.l2_cache_size_bytes==40*1024*1024,
            "A100 PCIe 40GB capacity mismatch");
    for(int b:{1,2})for(int splits:{1,2,4}) {
        std::unique_ptr<DAG> d(build_fa2_a100(b,4,2,257,64,splits,cfg));
        inspect(*d,2LL*b*2*64*320*64);
    }
    {
        std::unique_ptr<DAG> d(build_fa2_a100(1,16,4,257,256,3,cfg));
        inspect(*d,2LL*4*64*320*256);
    }
    for(int batch : {1,2}) {
        std::unique_ptr<DAG> d(build_fa2_a100(batch,4,2,65,64,2,cfg));
        Simulator sim(d.get(),cfg);
        require(sim.run(1000000,false),"small-batch FA2 did not drain");
    }
    numerical_reference();
    std::cout<<"fa2_tests passed\n";
}
