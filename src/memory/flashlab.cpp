#include "memory/base.h"
#include "memory/clock.h"
#include <physical/simulation_session.hpp>
#include <queue>
namespace GTSim {
namespace {
namespace f=hbfsim::physical;
class FlashLab final : public MemoryBackend {
    MemoryClock clock_;
    std::uint64_t base_, sequence_{}, static_bytes_{}, preload_end_{}, raw_bytes_{};
    f::SimulationSessionConfig native_config_;
    double origin_{};
    int capacity_;
    std::unique_ptr<f::SimulationSession> session_;
    struct Completion { double ps; MemRequest r; bool operator<(const Completion& b) const { return ps>b.ps; } };
    std::priority_queue<Completion> pending_;
    MemoryBackendStatistics stats_;
public:
    FlashLab(const HBFConfig& c,double frequency,std::uint64_t base):clock_(frequency,1),base_(base),capacity_(c.queue_capacity) {
        f::SimulationSessionConfig cfg;
        cfg.enable_hbm=false;
        cfg.hbf.host.read_buffer_coalescing_enabled = true;
        cfg.hbf.host.mapping_mode=hbfsim::host::MappingMode::RawPhysical;
        auto& d=cfg.hbf.device;
        d.stacks=c.controller.stacks; d.channels_per_stack=c.controller.host_channels_per_stack;
        d.dies_per_channel=1; d.planes_per_die=16;
        d.blocks_per_plane=c.controller.blocks_per_plane * (c.controller.banks_per_channel / 16);
        if(c.controller.geometry==HBFGeometryProfile::GroupedPlanes) {
            d.grouped_geometry=true;
            d.channels_per_stack=c.controller.dies_per_stack;
            d.planes_per_die=c.controller.planes_per_die;
            d.planes_per_command_group=c.controller.planes_per_command_group;
            d.blocks_per_plane=c.controller.blocks_per_plane;
        }
        d.pages_per_block=c.controller.pages_per_block;
        d.t_read_page_ns=c.controller.sense_latency_us*1000;
        d.t_program_page_ns=c.controller.program_latency_us*1000;
        if(c.controller.enable_read_sram) {
            d.logic_sram_bandwidth_GBps=c.controller.read_sram_bandwidth_GBps_per_stack;
            d.read_sram_access_latency_ns=c.controller.read_sram_access_latency_ns;
        }
        cfg.static_hbf_blocks_per_plane=1;
        static_bytes_=static_cast<std::uint64_t>(d.stacks)*d.channels_per_stack*d.dies_per_channel*d.planes_per_die*d.pages_per_block*d.page_size_bytes;
        raw_bytes_=static_bytes_*d.blocks_per_plane;
        native_config_=std::move(cfg);
    }
    void initialize() {
        if(session_) return;
        native_config_.static_hbf_blocks_per_plane=std::max<std::uint64_t>(1,
            preload_end_/static_bytes_+(preload_end_%static_bytes_!=0));
        session_=std::make_unique<f::SimulationSession>(native_config_);
    }
    bool try_submit(const MemRequest& r,int now) override {
        if(r.size_bytes<=0 || r.global_addr<base_ || r.operation!=MemoryOperation::Read) throw std::invalid_argument("FlashLab static image requires valid reads");
        if(pending_.size()>=static_cast<std::size_t>(capacity_)) { ++stats_.admission_retries; return false; }
        const double at=clock_.first_edge_at_or_after(now)/1000.0;
        if(at<origin_) throw std::logic_error("FlashLab arrival precedes frontier");
        initialize();
        f::SimulationTransaction t;
        t.id=std::to_string(++sequence_); t.target=f::SimulationTarget::HbfStatic;
        t.addr=r.global_addr-base_; t.bytes=r.size_bytes; t.issue_ns=at-origin_;
        t.read_data_path=r.read_path==MemoryReadPath::Prefetch?f::ReadDataPath::Prefetch:
            r.read_path==MemoryReadPath::Resident?f::ReadDataPath::Resident:f::ReadDataPath::Host;
        f::SimulationBatchOptions options; options.frontier=std::vector<std::string>{};
        auto result=session_->run_batch(t.id,{t},options);
        origin_=result.blocking_finish_ns;
        if(result.completions.size()!=1 || result.completions[0].finish_ns<at) throw std::runtime_error("invalid FlashLab completion");
        pending_.push({result.completions[0].finish_ns*1000,r}); ++stats_.accepted_requests;
        return true;
    }
    std::vector<MemResponse> step(int now) override {
        std::vector<MemResponse> out;
        while(!pending_.empty() && pending_.top().ps<=clock_.ticks_at(now)) {
            auto r=pending_.top().r; pending_.pop();
            out.push_back({r.request_id, r.global_addr, r.tier, true,
                           r.size_bytes, r.operation, 0,
                           r.operation == MemoryOperation::Read
                               ? r.shadow_digest
                               : 0});
            ++stats_.completed_requests;
        }
        return out;
    }
    std::size_t pending_requests() const override { return pending_.size(); }
    bool supports_sram_paths() const override { return true; }
    MemoryLocation locate(std::uint64_t address) const override {
        if(address<base_ || address-base_>=raw_bytes_) throw std::invalid_argument("FlashLab SRAM address outside device");
        auto* self=const_cast<FlashLab*>(this);
        self->initialize();
        const auto a=session_->hbf_static_location(address-base_);
        return {static_cast<int>(a.stack),
                static_cast<int>(a.channel*native_config_.hbf.device.dies_per_channel+a.die),
                static_cast<int>(a.plane)};
    }
    MemoryBackendStatistics backend_statistics() const override { return stats_; }
    DeviceCounters device_counters() const override {
        DeviceCounters d;
        if(!session_) {
            for(const auto* name:{"nand_senses","sense_commands","setup_bytes","setup_ps",
                                  "native_prefetch_fill_bytes","native_resident_read_bytes"})
                d.cumulative.emplace(name,0);
            return d;
        }
        const auto* s=session_->hbf_stats();
        d.cumulative={{"nand_senses",native_config_.hbf.device.grouped_geometry?s->group_sensed_pages:s->page_reads},
            {"sense_commands",native_config_.hbf.device.grouped_geometry?s->group_sense_commands:s->page_reads},{"setup_bytes",0},{"setup_ps",0},
            {"native_prefetch_fill_bytes",s->prefetch_fill_bytes}, {"native_resident_read_bytes",s->resident_read_bytes}};
        return d;
    }
    void preload_region(std::uint64_t address,std::uint64_t bytes,bool readonly) override {
        // Record the dense static image before session construction. Reserve enough
        // blocks per plane for the full declared weight extent, including large models.
        if(!readonly || address<base_ || address-base_>raw_bytes_ || bytes>raw_bytes_-(address-base_))
            throw std::invalid_argument("FlashLab preload exceeds device capacity");
        const auto end=address-base_+bytes;
        if(session_ && end>native_config_.static_hbf_blocks_per_plane*static_bytes_)
            throw std::invalid_argument("cannot grow a measured FlashLab static image");
        preload_end_=std::max(preload_end_,end);
    }
};
}
std::unique_ptr<MemoryBackend> make_flashlab_backend(const HBFConfig& c,double f,std::uint64_t b) { return std::make_unique<FlashLab>(c,f,b); }
}
