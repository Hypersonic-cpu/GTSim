#include "memory/base.h"
#include <stdexcept>
#include <iostream>
#include <set>
#include <physical/simulation_session.hpp>
#include <cmath>
using namespace GTSim;
void check(bool b,const char* m) { if(!b) throw std::runtime_error(m); }
namespace f=hbfsim::physical;
void flash_frontier_contract(bool coalescing) {
 f::SimulationSessionConfig c; c.enable_hbm=false; c.hbf.host.mapping_mode=hbfsim::host::MappingMode::RawPhysical; c.hbf.device.stacks=4; c.hbf.device.blocks_per_plane=8;c.static_hbf_blocks_per_plane=1;
 c.hbf.host.read_buffer_coalescing_enabled = coalescing;
 f::SimulationSession a(c),b(c);
 f::SimulationBatchOptions o; o.frontier=std::vector<std::string>{};
 std::vector<f::SimulationTransaction> ts;
 for(int i=0;i<8;++i){f::SimulationTransaction t;t.id=std::to_string(i);t.target=f::SimulationTarget::HbfStatic;t.bytes=128;t.addr=(i%3)*4096;t.issue_ns=i*5;ts.push_back(t);}
 auto ref=a.run_batch("all",ts,o);double origin=0;int i=0;
 for(auto t:ts){t.issue_ns-=origin;auto r=b.run_batch(t.id,{t},o);origin=r.blocking_finish_ns;
 check(std::abs(ref.completions[i].finish_ns-r.completions[0].finish_ns)<1e-6,"FlashLab frontier changes concurrent timing"); ++i;}
 if(coalescing) {
     check(a.hbf_stats()->page_reads == 3, "grouped reads did not coalesce by page");
     check(b.hbf_stats()->page_reads == 3, "incremental reads did not coalesce by page");
 }
}
int main() {
    flash_frontier_contract(false);
    flash_frontier_contract(true);
    for(auto kind:{HBFBackend::HBFine,HBFBackend::EventLab,HBFBackend::FlashLab}) {
        HBFConfig c; c.backend=kind; c.queue_capacity=2;
        c.controller.stacks=4;
        auto b=make_hbf_backend(c,1980,1ULL<<30);
        b->preload_region(1ULL<<30,8192,true);
        check(b->supports_prefetch()==(kind==HBFBackend::HBFine),"capability mismatch");
        std::set<std::uint64_t> ids;
        for(int i=0;i<2;++i) check(b->try_submit({static_cast<std::uint64_t>(i+1),(1ULL<<30)+i*(kind==HBFBackend::FlashLab ? 128 : 4096)+64,128,MemoryOperation::Read,MemoryTier::HBF},0),"admission failed");
        check(!b->try_submit({3,1ULL<<30,64,MemoryOperation::Read,MemoryTier::HBF},0),"missing backpressure");
        check(b->step(0).empty(),"early completion");
        for(int cycle=1;cycle<100000 && ids.size()<2;++cycle) {
            for(auto r:b->step(cycle)) { check(r.completed,"failed read after preload"); check(ids.insert(r.request_id).second,"duplicate completion"); }
        }
        check(ids.size()==2 && !b->pending_requests(),"completion lost");
        if(kind==HBFBackend::FlashLab)
            check(b->device_counters().cumulative.at("nand_senses")==1,
                  "FlashLab adapter failed to coalesce in-flight reads of the same page");
        check(b->try_submit({3,(1ULL<<30)+128,64,MemoryOperation::Read,MemoryTier::HBF},100000),"dynamic arrival failed");
        bool done=false;
        for(int cycle=100000;cycle<200000 && !done;++cycle) for(auto r:b->step(cycle)) {check(r.request_id==3 && r.completed,"dynamic read failed");done=true;}
        check(done && !b->pending_requests(),"dynamic completion lost");
        std::cout<<"backend "<<static_cast<int>(kind)<<" passed\n";
    }
}
