#include "memory/base.h"
#include "memory/clock.h"
#include <hbf_eventlab/runtime/simulation_session.hpp>
#include <unordered_map>
#include <set>
namespace GTSim {
namespace {
namespace e = hbf_eventlab;
class EventLab final : public MemoryBackend {
    MemoryClock clock_;
    std::uint64_t base_, offset_{}, next_{1};
    std::vector<std::unique_ptr<e::SimulationSession>> sessions_;
    unsigned stacks_, channels_, banks_, group_size_{1};
    struct Parent { MemRequest request; std::size_t remaining; bool failed{}; };
    std::unordered_map<std::uint64_t, Parent> pending_;
    std::unordered_map<std::uint64_t, std::uint64_t> parents_;
    std::vector<MemResponse> ready_;
    std::set<std::uint64_t> pages_;
    MemoryBackendStatistics stats_;
    std::uint64_t setup_bytes_{};
    int capacity_;
    bool setup_{false}, setup_failed_{false};
    static void check_progress(const e::SessionReport& report) {
        const auto status=report.run.status;
        if(status==e::RunStatus::LimitReached || status==e::RunStatus::ModelError ||
           status==e::RunStatus::ImplementationError || status==e::RunStatus::Stalled)
            throw std::runtime_error("EventLab progress stopped: "+report.run.explanation);
    }
public:
    EventLab(const HBFConfig& c, double frequency, std::uint64_t base)
        : clock_(frequency, 1), base_(base), stacks_(c.controller.stacks), channels_(c.controller.host_channels_per_stack),
          banks_(c.controller.banks_per_channel), capacity_(c.queue_capacity) {
        e::SimulationConfiguration cfg;
        cfg.device.geometry={static_cast<unsigned>(c.controller.host_channels_per_stack),1,
            static_cast<unsigned>(c.controller.banks_per_channel),static_cast<unsigned>(c.controller.pages_per_block),
            static_cast<unsigned>(c.controller.blocks_per_plane),1};
        if(c.controller.geometry==HBFGeometryProfile::GroupedPlanes) {
            if(c.controller.planes_per_command_group<=0 ||
               c.controller.planes_per_die%c.controller.planes_per_command_group)
                throw std::invalid_argument("invalid EventLab coupled plane group");
            group_size_=c.controller.planes_per_command_group;
            channels_=c.controller.dies_per_stack;
            banks_=c.controller.planes_per_die/group_size_;
            cfg.device.spec_profile="grouped_planes";
            cfg.device.geometry={channels_,1,banks_,
                static_cast<unsigned>(c.controller.pages_per_block*group_size_),
                static_cast<unsigned>(c.controller.blocks_per_plane),0,group_size_};
        }
        cfg.address.interleave_bytes=4096ULL*group_size_;
        if(c.controller.enable_read_sram) {
            cfg.execution.sram_access_latency=e::Tick{static_cast<std::uint64_t>(c.controller.read_sram_access_latency_ns*1000)};
            cfg.execution.sram_bandwidth_GBps=c.controller.read_sram_bandwidth_GBps_per_stack;
        }
        cfg.execution.ucie.speed_grade=e::UcieSpeedGrade::Grade2;
        cfg.execution.media_timing.sense=e::Tick{static_cast<std::uint64_t>(c.controller.sense_latency_us*1000000)};
        cfg.execution.media_timing.program=e::Tick{static_cast<std::uint64_t>(c.controller.program_latency_us*1000000)};
        cfg.host.writes=e::WriteCommandMode::Burst4096;
        cfg.host.max_resident_fragments=65536;
        cfg.host.retain_completed_results=false;
        cfg.host.max_resident_fragments_per_channel=c.eventlab_fragments_per_channel;
        // This is a persistent online session; bound resident work, not lifetime IO.
        cfg.host.max_logical_fragments=0;
        for(unsigned stack=0;stack<stacks_;++stack) {
        auto session_=std::make_unique<e::SimulationSession>(e::resolve_configuration(cfg));
        auto limits=session_->run_limits();
        // GTSim owns the lifetime/cycle budget. Native absolute limits would
        // otherwise include setup time and halt an online session silently.
        limits.max_tick.reset();limits.events.total_events=0;
        session_->set_run_limits(limits);
        session_->set_completion_handler([this](const e::GlobalCompletion& c) {
            if(setup_) { setup_failed_ |= c.result.state != e::LogicalResult::Success; return; }
            auto child=parents_.find(c.result.id.value);
            if(child==parents_.end()) throw std::logic_error("unknown EventLab completion");
            auto it=pending_.find(child->second);
            auto& parent=it->second;
            parent.failed |= c.result.state!=e::LogicalResult::Success;
            parents_.erase(child);
            if(--parent.remaining) return;
            const auto r=parent.request;
            ready_.push_back({r.request_id,r.global_addr,r.tier,!parent.failed,r.size_bytes,r.operation,parent.failed?1:0});
            pending_.erase(it); ++stats_.completed_requests;
        });
        sessions_.push_back(std::move(session_));
        }
    }
    bool try_submit(const MemRequest& r,int now) override {
        if(r.size_bytes<=0 || r.global_addr<base_) throw std::invalid_argument("invalid EventLab range");
        if(pending_.size()+ready_.size()>=static_cast<std::size_t>(capacity_)) { ++stats_.admission_retries; return false; }
        const auto at=offset_+clock_.first_edge_at_or_after(now);
        const auto address=r.global_addr-base_;
        if (static_cast<std::uint64_t>(r.size_bytes)>UINT64_MAX-address) throw std::invalid_argument("EventLab range overflow");
        std::vector<std::vector<e::GlobalRequest>> batches(stacks_);
        auto cursor=address/64*64;
        const auto end=(address+r.size_bytes+63)/64*64;
        auto id=next_;
        std::size_t children=0;
        while(cursor<end) {
            const auto stop=std::min(end,(cursor/4096+1)*4096);
            const auto megapage=4096ULL*group_size_;
            const auto stack=(cursor/megapage)%stacks_;
            const auto local=cursor/megapage/stacks_*megapage+cursor%megapage;
            e::GlobalRequest req{};
            req.id=e::GlobalRequestId{id++}; req.op=r.operation==MemoryOperation::Read?e::Op::Read:e::Op::Write;
            req.read_data_path=r.read_path==MemoryReadPath::Prefetch?e::ReadDataPath::Prefetch:
                r.read_path==MemoryReadPath::Resident?e::ReadDataPath::Resident:e::ReadDataPath::Host;
            req.address=e::GlobalByteAddress{local}; req.bytes=stop-cursor; req.content=1; req.not_before=e::Tick{at};
            if(req.op==e::Op::Read) req.expected_content=1;
            batches[stack].push_back(req); cursor=stop; ++children;
        }
        for(unsigned stack=0;stack<stacks_;++stack) {
            check_progress(sessions_[stack]->advance_before(e::Tick{at}));
            const auto cap=sessions_[stack]->input_capacity();
            std::uint64_t fragments=0;
            std::vector<std::uint64_t> per_channel(channels_);
            for(const auto& req:batches[stack]) {
                fragments+=req.bytes/64;
                per_channel[(req.address.value/(4096ULL*group_size_))%channels_]+=req.bytes/64;
            }
            for(unsigned ch=0;ch<channels_;++ch) {
                const auto lane=sessions_[stack]->host().channel_input_capacity(ch);
                if(lane.limit && per_channel[ch]>lane.limit-lane.resident) {
                    ++stats_.admission_retries;return false;
                }
            }
            if(cap.limit && fragments>cap.limit-cap.resident) { ++stats_.admission_retries; return false; }
        }
        const auto parent=next_;
        pending_.emplace(parent,Parent{r,children});
        for(unsigned stack=0;stack<stacks_;++stack) if(!batches[stack].empty()) {
            for(const auto& req:batches[stack]) parents_.emplace(req.id.value,parent);
            sessions_[stack]->submit(batches[stack]);
        }
        next_=id; ++stats_.accepted_requests;
        return true;
    }
    std::vector<MemResponse> step(int now) override {
        for(auto& session_:sessions_) check_progress(session_->advance_before(e::Tick{offset_+clock_.first_edge_at_or_after(now)}));
        auto result=std::move(ready_); ready_.clear(); return result;
    }
    std::size_t pending_requests() const override { return pending_.size()+ready_.size(); }
    bool supports_sram_paths() const override { return true; }
    MemoryLocation locate(std::uint64_t address) const override {
        if(address<base_) throw std::invalid_argument("EventLab SRAM address outside device");
        const auto page=(address-base_)/4096;
        const auto group=page/group_size_;
        return {static_cast<int>(group%stacks_),static_cast<int>((group/stacks_)%channels_),
                static_cast<int>(((group/stacks_/channels_)%banks_)*group_size_+page%group_size_)};
    }
    MemoryBackendStatistics backend_statistics() const override { return stats_; }
    DeviceCounters device_counters() const override {
        DeviceCounters d;
        for(const auto& session_:sessions_) {const auto& s=session_->device().statistics(); d.cumulative["nand_senses"]+=s.sensed_pages; d.cumulative["sense_commands"]+=s.senses;
            d.cumulative["sensed_bytes"]+=s.sensed_pages*4096; d.cumulative["page_buffer_hits"]+=s.cache_hits;
            d.cumulative["native_prefetch_fill_bytes"]+=s.prefetch_fill_bytes;
            d.cumulative["native_resident_read_bytes"]+=s.resident_read_bytes;
            d.cumulative["external_request_bytes"]+=s.external_request_bytes;
            d.cumulative["external_response_bytes"]+=s.external_response_bytes;}
        d.cumulative["setup_bytes"]=setup_bytes_; d.cumulative["setup_ps"]=offset_; return d;
    }
    void preload_region(std::uint64_t address,std::uint64_t bytes,bool) override {
        if(!pending_.empty() || !ready_.empty() || address<base_) throw std::invalid_argument("invalid EventLab preload");
        setup_=true;
        const auto megapage=4096ULL*group_size_;
        const auto first=(address-base_)/megapage*group_size_;
        const auto end=(address-base_+bytes+megapage-1)/megapage*group_size_;
        for(auto page=first;page<end;++page) {
            if(pages_.contains(page)) continue;
            auto& session_=sessions_[(page/group_size_)%stacks_];
            e::GlobalRequest req{};
            req.id=e::GlobalRequestId{next_++}; req.op=e::Op::Write;
            req.address=e::GlobalByteAddress{(page/(group_size_*stacks_)*group_size_+page%group_size_)*4096}; req.bytes=4096; req.content=1;
            req.not_before=session_->now();
            if(!session_->accepts_at_floor()) req.not_before=e::Tick{session_->now().value+1};
            session_->submit(req);
            check_progress(session_->advance_until_yield());
            if(setup_failed_ || !session_->host().activity().drained()) throw std::runtime_error("EventLab preload did not drain successfully");
            pages_.insert(page); setup_bytes_+=4096;
        }
        for(auto& session_:sessions_) offset_=std::max(offset_,session_->now().value+1);
        for(auto& session_:sessions_) check_progress(session_->advance_before(e::Tick{offset_}));
        setup_=false;
    }
};
}
std::unique_ptr<MemoryBackend> make_eventlab_backend(const HBFConfig& c,double f,std::uint64_t b) { return std::make_unique<EventLab>(c,f,b); }
}
