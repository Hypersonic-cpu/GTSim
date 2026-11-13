#include "memory/base.h"
#include "memory/system.h"
#include <stdexcept>
#include <iostream>
#include <set>
#include <physical/simulation_session.hpp>
#include <physical/hbf/hbf_device.hpp>
#include <hbf_eventlab/runtime/simulation_session.hpp>
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
void flash_sram_native_contract() {
    f::SimulationSessionConfig c;
    c.enable_hbm=false;
    c.hbf.host.mapping_mode=hbfsim::host::MappingMode::RawPhysical;
    c.hbf.host.read_buffer_coalescing_enabled=true;
    c.hbf.device.blocks_per_plane=8;
    c.static_hbf_blocks_per_plane=1;
    f::SimulationSession session(c);
    f::SimulationBatchOptions options;
    options.frontier=std::vector<std::string>{};
    f::SimulationTransaction fill;
    fill.id="fill"; fill.target=f::SimulationTarget::HbfStatic;
    fill.bytes=4096; fill.read_data_path=f::ReadDataPath::Prefetch;
    auto receipt=session.run_batch("fill",{fill},options);
    check(receipt.completions.size()==1,"missing native fill receipt");
    check(receipt.completions[0].finish_ns>=4000,"NAND fill completed early");
    check(receipt.device_after.hbf.page_reads==1,"fill did not sense NAND");
    check(receipt.device_after.hbf.prefetch_fill_bytes==4096,"native fill byte conservation");
    check(session.hbf_stats()->hb_io_data_busy_ns==0,"prefetch crossed external HBIO");
    f::SimulationTransaction resident;
    resident.id="resident"; resident.target=f::SimulationTarget::HbfStatic;
    resident.bytes=128; resident.read_data_path=f::ReadDataPath::Resident;
    resident.issue_ns=receipt.completions[0].finish_ns;
    auto served=session.run_batch("resident",{resident},options);
    check(served.device_after.hbf.page_reads==1,"resident path sensed NAND");
    check(served.device_after.hbf.resident_read_bytes==128,"native resident byte conservation");
    check(session.hbf_stats()->hb_io_data_busy_ns>0,"resident payload did not cross HBIO");
    check(served.completions[0].finish_ns>=resident.issue_ns,"early resident receipt");
}

void shared_sram_adapter_contract(HBFBackend kind) {
    HBFConfig config;
    config.backend=kind;
    config.controller.enable_read_sram=true;
    config.controller.read_sram_access_latency_ns=20;
    config.controller.read_sram_bandwidth_GBps_per_stack=1536;
    config.controller.sram_bytes_per_plane=4096;
    config.controller.base_die_sram_bytes_per_stack=65536;
    auto backend=make_hbf_backend(config,1980,0);
    backend->preload_region(0,8192,true);
    auto token=backend->reserve_prefetch(0,4096,PrefetchBuffer{0,2,1});
    check(token!=0,"FlashLab shared SRAM reservation failed");
    backend->start_prefetch(token);
    backend->step(0);
    check(!backend->prefetch_ready(token),"FlashLab SRAM published before NAND fill");
    check(backend->try_submit({1,0,128,MemoryOperation::Read,MemoryTier::HBF,{},token},1),"FlashLab fallback rejected");
    int cycle=1;
    for(;cycle<100000 && (backend->pending_requests() || backend->pending_background_requests());++cycle)
        backend->step(cycle);
    check(backend->prefetch_ready(token),"FlashLab fill failed to publish");
    const auto senses=backend->device_counters().cumulative.at("nand_senses");
    check(backend->try_submit({2,0,128,MemoryOperation::Read,MemoryTier::HBF,{},token},cycle),"FlashLab resident rejected");
    backend->release_prefetch(token);
    for(;cycle<200000 && backend->pending_requests();++cycle) backend->step(cycle);
    const auto stats=backend->device_counters();
    check(stats.cumulative.at("nand_senses")==senses,"FlashLab resident sensed NAND");
    check(stats.cumulative.at("native_prefetch_fill_bytes")==4096,"FlashLab fill bypassed native path");
    check(stats.cumulative.at("native_resident_read_bytes")==128,"FlashLab resident bypassed native path");
    check(stats.gauges.at("sram_reserved_bytes")==0,"FlashLab SRAM lease was not reclaimed");
}

void event_stream_retirement_contract() {
    namespace e=hbf_eventlab;
    std::uint64_t reference_finish=0;
    for(bool retain:{true,false}) {
        e::SimulationConfiguration cfg;
        cfg.address.interleave_bytes=4096;
        cfg.host.writes=e::WriteCommandMode::Burst4096;
        cfg.host.retain_completed_results=retain;
        e::SimulationSession session(e::resolve_configuration(cfg));
        unsigned completed=0;
        session.set_completion_handler([&](const e::GlobalCompletion& result) {
            check(result.result.state==e::LogicalResult::Success,"stream request failed");
            ++completed;
        });
        for(unsigned id=1;id<=201;++id) {
            e::GlobalRequest request{};
            request.id=e::GlobalRequestId{id};
            request.op=id==1?e::Op::Write:e::Op::Read;
            request.bytes=id==1?4096:64;request.content=1;
            if(id!=1)request.expected_content=1;
            request.not_before=e::Tick{session.now().value+1};
            session.submit(request);session.advance_until_yield();
            const auto storage=session.host().storage();
            check(storage.requests==0 && storage.work==0,"terminal stream work not reclaimed");
            check(storage.results==(retain?id:0),"completed stream history grew unexpectedly");
        }
        check(completed==201,"stream retirement lost completion callbacks");
        if(retain)reference_finish=session.now().value;
        else {
            check(session.now().value==reference_finish,"result retirement changed timing");
            e::GlobalRequest duplicate{};duplicate.id=e::GlobalRequestId{1};
            duplicate.bytes=64;duplicate.op=e::Op::Read;
            duplicate.not_before=e::Tick{session.now().value+1};
            bool rejected=false;
            try {session.submit(duplicate);} catch(const std::invalid_argument&) {rejected=true;}
            check(rejected,"compact stream reused a retired ID");
        }
    }
}

void event_sram_native_contract() {
    namespace e=hbf_eventlab;
    e::SimulationConfiguration cfg;
    cfg.address.interleave_bytes=4096;
    cfg.host.writes=e::WriteCommandMode::Burst4096;
    e::SimulationSession session(e::resolve_configuration(cfg));
    bool failed=false;
    session.set_completion_handler([&](const e::GlobalCompletion& result) {
        failed |= result.result.state!=e::LogicalResult::Success;
    });
    e::GlobalRequest write{};
    write.id=e::GlobalRequestId{1}; write.op=e::Op::Write;
    write.bytes=4096;write.content=1;
    session.submit(write); session.advance_until_yield();
    check(!failed,"EventLab setup write failed");
    const auto before=session.device().statistics();
    e::GlobalRequest fill{};
    fill.id=e::GlobalRequestId{2}; fill.op=e::Op::Read;
    fill.bytes=128;fill.expected_content=1;
    fill.not_before=e::Tick{session.now().value+1};
    fill.read_data_path=e::ReadDataPath::Prefetch;
    session.submit(fill);session.advance_until_yield();
    const auto after=session.device().statistics();
    check(!failed,"EventLab native fill returned wrong data");
    check(after.senses==before.senses+1,"EventLab native fill did not sense NAND");
    check(after.external_request_bytes==before.external_request_bytes &&
          after.external_response_bytes==before.external_response_bytes,
          "EventLab prefetch crossed UCIe");
    check(after.prefetch_fill_bytes==128,"EventLab native SRAM fill bytes mismatch");
    auto resident=fill;resident.id=e::GlobalRequestId{3};
    resident.not_before=e::Tick{session.now().value+1};
    resident.read_data_path=e::ReadDataPath::Resident;
    session.submit(resident);session.advance_until_yield();
    const auto served=session.device().statistics();
    check(!failed && served.senses==after.senses,"EventLab resident read touched NAND or returned wrong data");
    check(served.resident_read_bytes==128,"EventLab resident bytes mismatch");
    check(served.external_response_bytes>after.external_response_bytes,"EventLab resident payload bypassed UCIe");
    auto missing=resident;missing.id=e::GlobalRequestId{4};missing.address=e::GlobalByteAddress{4096};
    missing.not_before=e::Tick{session.now().value+1};
    session.submit(missing);session.advance_until_yield();
    check(failed,"EventLab resident path masked uninitialized NAND state");
}

void event_preload_order_contract() {
    MemorySystemConfig config;
    config.topology=MemoryTopology::CoLocated;
    config.hbm.stacks=4;
    config.hbf.backend=HBFBackend::EventLab;
    config.hbf.read_bw_GBps=488;
    config.hbf.write_bw_GBps=27.2;
    config.address_layout.enabled=true;
    config.address_layout.hbm_capacity_bytes=64ULL<<30;
    config.address_layout.hbf_base=64ULL<<30;
    config.address_layout.hbf_capacity_bytes=512ULL<<30;
    TensorRegistry tensors(config.address_layout);
    tensors.allocate_tensor(0,"prefix",{512*2048},{1},DataType::FP16,MemoryTier::HBF,true,true);
    tensors.allocate_tensor(1,"next_page",{2048},{1},DataType::FP16,MemoryTier::HBF,true,true);
    // The second tensor begins at page 1 of the first physical bank. Reversing
    // these preload calls would violate the native block's program order.
    MemorySystem memory(4096,128,1,1024,512,64,false,config,1980,tensors);
    check(memory.foreground_idle(),"EventLab initialization did not drain");
    check(memory.tier_counters().at(MemoryTier::HBF).device.cumulative.at("setup_bytes")==513*4096,
          "EventLab ordered preload byte conservation");
}

void event_group_native_contract() {
    namespace e=hbf_eventlab;
    e::SimulationConfiguration cfg;
    cfg.device.spec_profile="grouped_planes";
    cfg.execution.media_timing.sense=e::Tick{4000000};
    cfg.device.geometry={1,1,2,32,4,1,4};
    cfg.address.interleave_bytes=4*4096;
    cfg.host.writes=e::WriteCommandMode::Burst4096;
    e::SimulationSession session(e::resolve_configuration(cfg));
    bool failed=false;
    std::map<std::uint64_t,std::uint64_t> finished;
    session.set_completion_handler([&](const e::GlobalCompletion& r) {
        failed|=r.result.state!=e::LogicalResult::Success;
        finished[r.result.id.value]=r.at.value;
    });
    for(unsigned page=0;page<24;++page) {
        e::GlobalRequest w{};w.id=e::GlobalRequestId{page+1};w.op=e::Op::Write;
        w.address=e::GlobalByteAddress{page*4096ULL};w.bytes=4096;w.content=1;
        w.not_before=e::Tick{session.now().value+1};
        session.submit(w);session.advance_until_yield();
    }
    check(!failed,"grouped EventLab setup failed");
    const auto at=session.now().value+1;
    std::vector<e::GlobalRequest> reads;
    for(auto page:{0,1,4,8}) {
        e::GlobalRequest r{};r.id=e::GlobalRequestId{100+reads.size()};r.op=e::Op::Read;
        r.address=e::GlobalByteAddress{static_cast<std::uint64_t>(page)*4096};r.bytes=64;
        r.expected_content=1;r.not_before=e::Tick{at};reads.push_back(r);
    }
    session.submit(reads);session.advance_until_yield();
    const auto stats=session.device().statistics();
    check(!failed,"grouped EventLab returned the wrong plane's data");
    check(stats.senses==3 && stats.sensed_pages==12,"EventLab group sense conservation");
    check(stats.cache_hits>=1,"EventLab sibling plane did not share a page buffer");
    check(finished.at(102)-finished.at(100)<2000000,"EventLab independent groups serialized");
    check(finished.at(103)>=at+8000000,"EventLab same-group page offsets sensed concurrently");
    const auto before=session.device().statistics().senses;
    for(auto page:{16,8,0}) {
        e::GlobalRequest r{};r.id=e::GlobalRequestId{200+static_cast<unsigned>(page)};r.op=e::Op::Read;
        r.address=e::GlobalByteAddress{static_cast<std::uint64_t>(page)*4096};r.bytes=64;r.expected_content=1;
        r.not_before=e::Tick{session.now().value+1};session.submit(r);session.advance_until_yield();
    }
    check(!failed && session.device().statistics().senses==before+2,"EventLab two-megapage buffer eviction");
}

void group_adapter_layout_contract(HBFBackend kind) {
    HBFConfig c;c.backend=kind;
    c.controller.geometry=HBFGeometryProfile::GroupedPlanes;
    c.controller.stacks=4;c.controller.dies_per_stack=8;c.controller.planes_per_die=96;
    c.controller.planes_per_command_group=32;c.controller.banks_per_channel=48;
    c.controller.blocks_per_plane=342;c.controller.pages_per_block=192;
    auto backend=make_hbf_backend(c,1980,0);
    const std::vector<std::pair<std::uint64_t,MemoryLocation>> cases={
        {0,{0,0,0}},{31,{0,0,31}},{32,{1,0,0}},
        {128,{0,1,0}},{1024,{0,0,32}},{3072,{0,0,0}}};
    for(const auto& [page,want]:cases) {
        auto got=backend->locate(page*4096);
        check(got.stack==want.stack && got.die==want.die && got.plane==want.plane,
              "EventLab megapage rotation or physical SRAM location mismatch");
    }
    backend->preload_region(0,4096,true);
    check(kind!=HBFBackend::EventLab || backend->device_counters().cumulative.at("setup_bytes")==128*1024,
          "EventLab grouped preload did not initialize padding lanes");
    check(backend->try_submit({1,0,128,MemoryOperation::Read,MemoryTier::HBF},0),"group read admission");
    bool done=false;
    for(int cycle=1;cycle<20000 && !done;++cycle)
        for(auto r:backend->step(cycle)) {check(r.completed,"group read failed");done=true;}
    const auto counters=backend->device_counters().cumulative;
    check(done && counters.at("nand_senses")==32 && counters.at("sense_commands")==1,
          "EventLab adapter did not issue a native coupled plane command");
}

void event_hyperpage_dispatch_contract() {
    for(const bool interleaved:{false,true}) {
        HBFConfig c;c.backend=HBFBackend::EventLab;
        auto& p=c.controller;
        p.geometry=HBFGeometryProfile::GroupedPlanes;
        p.stacks=4;p.dies_per_stack=8;p.planes_per_die=96;p.planes_per_command_group=32;
        p.banks_per_channel=48;p.blocks_per_plane=342;p.pages_per_block=192;
        p.enable_read_sram=true;p.read_sram_access_latency_ns=20;
        p.read_sram_bandwidth_GBps_per_stack=1536;
        p.sram_bytes_per_plane=8192;p.base_die_sram_bytes_per_stack=2097152;
        c.eventlab_fragments_per_channel=256;c.prefetch_hyperpage_order=interleaved;
        auto b=make_hbf_backend(c,1980,0);
        constexpr auto hyperpage=12ULL*1024*1024;
        b->preload_region(0,hyperpage,true);
        auto token=b->reserve_prefetch(0,hyperpage,PrefetchBuffer{0,2,1});
        check(token!=0,"hyperpage did not fit configured double-buffer slot");
        b->start_prefetch(token);b->step(1);b->step(2);
        const auto counts=b->device_counters().cumulative;
        check(counts.at("sense_commands")==static_cast<unsigned>(interleaved?96:1),
              "hyperpage queue did not expose independent native megaplanes");
        check(!b->prefetch_ready(token),"hyperpage SRAM became ready before NAND sense");
    }
}

void flash_group_native_contract() {
    f::SimulationSessionConfig c;
    c.enable_hbm=false;
    c.hbf.host.mapping_mode=hbfsim::host::MappingMode::RawPhysical;
    c.hbf.host.read_buffer_coalescing_enabled=true;
    auto& d=c.hbf.device;
    d.grouped_geometry=true; d.planes_per_command_group=4;
    d.stacks=2; d.channels_per_stack=2; d.planes_per_die=8;
    d.blocks_per_plane=8; d.pages_per_block=8;
    c.static_hbf_blocks_per_plane=1;
    f::SimulationSession a(c), b(c);
    for (std::uint64_t page=0;page<96;++page) {
        const auto loc=a.hbf_static_location(page*4096);
        const auto group=page/4;
        check(loc.stack==group%2 && loc.channel==(group/2)%2 &&
              loc.plane==((group/4)%2)*4+page%4 &&
              loc.page==page/32,"FlashLab grouped physical layout mismatch");
    }
    f::SimulationBatchOptions options;
    options.frontier=std::vector<std::string>{};
    std::vector<f::SimulationTransaction> requests;
    for(int i=0;i<4;++i) {
        f::SimulationTransaction t;
        t.id=std::to_string(i);t.target=f::SimulationTarget::HbfStatic;
        t.addr=i*4096;t.bytes=128;t.issue_ns=i;
        requests.push_back(t);
    }
    const auto together=a.run_batch("lanes",requests,options);
    check(a.hbf_stats()->group_sense_commands==1 && a.hbf_stats()->group_sensed_pages==4,
          "FlashLab megaplane did not jointly sense all lanes");
    double origin=0;
    for(int i=0;i<4;++i) {
        auto t=requests[i];t.issue_ns-=origin;
        const auto r=b.run_batch(t.id,{t},options);origin=r.blocking_finish_ns;
        check(std::abs(together.completions[i].finish_ns-r.completions[0].finish_ns)<1e-6,
              "FlashLab grouped incremental timing differs from batch");
    }
    check(b.hbf_stats()->group_sense_commands==1,"FlashLab sibling read re-sensed NAND");
    for(int page:{32,64,96,0}) {
        auto t=requests[0]; t.id="offset"+std::to_string(page);
        t.addr=page*4096;t.issue_ns=100000+page;
        (void)a.run_batch(t.id,{t},options);
    }
    check(a.hbf_stats()->group_sense_commands==5,"FlashLab raw two-slot eviction failed");
}

void flash_raw_temporal_contract() {
    hbfsim::physical::hbf::HbfDeviceConfig config;
    config.grouped_geometry=true;config.planes_per_die=4;
    config.planes_per_command_group=4;config.channels_per_stack=1;
    config.blocks_per_plane=8;config.pages_per_block=8;
    hbfsim::physical::hbf::HbfDevice device(config,true);
    std::uint64_t a=0,b=0,c=0;
    check(device.raw_reserve(0,0,4,a),"raw A reserve failed");
    check(device.raw_reserve(1,4,8,b),"raw B reserve failed");
    check(device.raw_touch(0,7,a),"raw A consumer lease failed");
    check(!device.raw_reserve(2,5,9,c),"raw fill invalidated a promised future consumer");
    check(device.raw_reserve(2,8,12,c),"raw C reserve failed");
    const auto pending=device.raw_page(0,3);
    check(pending && pending->ready_ns==4 && pending->generation==a,
          "future raw reservation hid an earlier live page");
    check(!device.raw_touch(2,11,c),"pending raw data became visible before sense finish");
    check(device.raw_touch(2,12,c),"raw C publication failed");
    check(!device.cache_resident(0,12),"raw register became decoded data without ECC");
    device.cache_fill(2,13,c);
    check(device.cache_hit(2,14),"decoded page did not retain its raw generation");
    device.advance_cache(14);
    check(device.raw_page(2,14).has_value(),"retirement lost live raw page");
}

int main() {
    flash_raw_temporal_contract();
    flash_group_native_contract();
    event_stream_retirement_contract();
    event_hyperpage_dispatch_contract();
    event_group_native_contract();
    group_adapter_layout_contract(HBFBackend::EventLab);
    group_adapter_layout_contract(HBFBackend::FlashLab);
    event_preload_order_contract();
    event_sram_native_contract();
    flash_sram_native_contract();
    shared_sram_adapter_contract(HBFBackend::FlashLab);
    shared_sram_adapter_contract(HBFBackend::EventLab);
    flash_frontier_contract(false);
    flash_frontier_contract(true);
    for(auto kind:{HBFBackend::HBFine,HBFBackend::EventLab,HBFBackend::FlashLab}) {
        HBFConfig c; c.backend=kind; c.queue_capacity=2;
        c.controller.stacks=4;
        auto b=make_hbf_backend(c,1980,1ULL<<30);
        const auto initial_counters=b->device_counters().cumulative;
        b->preload_region(1ULL<<30,8192,true);
        bool disabled=false;
        try { b->reserve_prefetch(1ULL<<30,4096); }
        catch(const std::invalid_argument&) { disabled=true; }
        check(disabled,"disabled SRAM accepted a prefetch reservation");
        bool raw_path=false;
        try { b->try_submit({999,1ULL<<30,64,MemoryOperation::Read,MemoryTier::HBF,{},0,MemoryReadPath::Resident},0); }
        catch(const std::invalid_argument&) { raw_path=true; }
        check(raw_path && !b->pending_requests(),"unmanaged SRAM path bypassed feature activation");
        check(b->supports_prefetch(),"capability mismatch");
        std::set<std::uint64_t> ids;
        for(int i=0;i<2;++i) check(b->try_submit({static_cast<std::uint64_t>(i+1),(1ULL<<30)+i*(kind==HBFBackend::FlashLab ? 128 : 4096)+64,128,MemoryOperation::Read,MemoryTier::HBF},0),"admission failed");
        check(!b->try_submit({3,1ULL<<30,64,MemoryOperation::Read,MemoryTier::HBF},0),"missing backpressure");
        check(b->step(0).empty(),"early completion");
        for(int cycle=1;cycle<100000 && ids.size()<2;++cycle) {
            for(auto r:b->step(cycle)) { check(r.completed,"failed read after preload"); check(ids.insert(r.request_id).second,"duplicate completion"); }
        }
        check(ids.size()==2 && !b->pending_requests(),"completion lost");
        for(const auto& [name,value]:b->device_counters().cumulative) {
            (void)value;
            check(initial_counters.contains(name),"backend counter schema changed after lazy initialization");
        }
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
