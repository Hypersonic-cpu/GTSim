#include "memory/base.h"
#include "memory/clock.h"
#include <physical/simulation_session.hpp>
#include <queue>
namespace GTSim {
namespace {
namespace f=hbfsim::physical;
class FlashLab final : public MemoryBackend {
    MemoryClock clock_;
    std::uint64_t base_, sequence_{}, static_bytes_{};
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
        d.blocks_per_plane=c.controller.blocks_per_plane * (c.controller.banks_per_channel / 16); d.pages_per_block=c.controller.pages_per_block;
        d.t_read_page_ns=c.controller.sense_latency_us*1000;
        d.t_program_page_ns=c.controller.program_latency_us*1000;
        cfg.static_hbf_blocks_per_plane=1;
        static_bytes_=static_cast<std::uint64_t>(d.stacks)*d.channels_per_stack*d.dies_per_channel*d.planes_per_die*d.pages_per_block*d.page_size_bytes;
        session_=std::make_unique<f::SimulationSession>(cfg);
    }
    bool try_submit(const MemRequest& r,int now) override {
        if(r.size_bytes<=0 || r.global_addr<base_ || r.operation!=MemoryOperation::Read) throw std::invalid_argument("FlashLab static image requires valid reads");
        if(pending_.size()>=static_cast<std::size_t>(capacity_)) { ++stats_.admission_retries; return false; }
        const double at=clock_.first_edge_at_or_after(now)/1000.0;
        if(at<origin_) throw std::logic_error("FlashLab arrival precedes frontier");
        f::SimulationTransaction t;
        t.id=std::to_string(++sequence_); t.target=f::SimulationTarget::HbfStatic;
        t.addr=r.global_addr-base_; t.bytes=r.size_bytes; t.issue_ns=at-origin_;
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
            out.push_back({r.request_id,r.global_addr,r.tier,true,r.size_bytes,r.operation}); ++stats_.completed_requests;
        }
        return out;
    }
    std::size_t pending_requests() const override { return pending_.size(); }
    MemoryBackendStatistics backend_statistics() const override { return stats_; }
    DeviceCounters device_counters() const override {
        DeviceCounters d;
        const auto* s=session_->hbf_stats();
        d.cumulative={{"nand_senses",s->page_reads},{"setup_bytes",0},{"setup_ps",0}};
        return d;
    }
    void preload_region(std::uint64_t address,std::uint64_t bytes,bool readonly) override {
        // Session construction installs real immutable static blocks. Validate the declared extent.
        if(!readonly || address<base_ || address-base_>static_bytes_ || bytes>static_bytes_-(address-base_))
            throw std::invalid_argument("FlashLab preload exceeds reserved static image extent");
    }
};
}
std::unique_ptr<MemoryBackend> make_flashlab_backend(const HBFConfig& c,double f,std::uint64_t b) { return std::make_unique<FlashLab>(c,f,b); }
}
