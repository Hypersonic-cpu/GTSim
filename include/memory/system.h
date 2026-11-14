#pragma once
#include "memory/cache.h"
namespace GTSim {
class D2DLink {
public:
    D2DLink(const D2DLinkConfig& config, double core_frequency_mhz)
        : bandwidth_bytes_per_cycle_(bytes_per_cycle_from_GBps(
              config.bandwidth_GBps, core_frequency_mhz)),
          latency_cycles_(config.latency_cycles),
          queue_capacity_(static_cast<std::size_t>(config.queue_capacity)) {}

    bool enqueue_request(const MemRequest& request, int now) {
        if (request_queue_.size() >= queue_capacity_) {
            return false;
        }
        const int transfer_bytes = request.operation == MemoryOperation::Read
                                       ? kMemoryControlBytes
                                       : std::max(1, request.size_bytes);
        const double serialization = serialization_cycles(transfer_bytes);
        const double start_cycle =
            std::max(static_cast<double>(now), next_forward_cycle_);
        const double finish_cycle = start_cycle + serialization;
        request_queue_.push_back(
            {request, arrival_cycle(finish_cycle + latency_cycles_)});
        next_forward_cycle_ = finish_cycle;
        statistics_.forward_bytes += static_cast<std::uint64_t>(transfer_bytes);
        statistics_.forward_busy_cycles += serialization;
        return true;
    }

    bool enqueue_response(const MemResponse& response, int now) {
        if (response_queue_.size() >= queue_capacity_) {
            return false;
        }
        const int transfer_bytes =
            response.size_bytes > 0 ? response.size_bytes : kMemoryControlBytes;
        const double serialization = serialization_cycles(transfer_bytes);
        const double start_cycle =
            std::max(static_cast<double>(now), next_reverse_cycle_);
        const double finish_cycle = start_cycle + serialization;
        response_queue_.push_back(
            {response, arrival_cycle(finish_cycle + latency_cycles_)});
        next_reverse_cycle_ = finish_cycle;
        statistics_.reverse_bytes += static_cast<std::uint64_t>(transfer_bytes);
        statistics_.reverse_busy_cycles += serialization;
        return true;
    }

    void step_requests(int now, std::vector<MemRequest>& ready) {
        ready.clear();
        while (!request_queue_.empty() &&
               request_queue_.front().ready_cycle <= now) {
            ready.push_back(request_queue_.front().value);
            request_queue_.pop_front();
        }
    }

    std::vector<MemRequest> step_requests(int now) {
        std::vector<MemRequest> ready;
        step_requests(now, ready);
        return ready;
    }

    void step_responses(int now, std::vector<MemResponse>& ready) {
        ready.clear();
        while (!response_queue_.empty() &&
               response_queue_.front().ready_cycle <= now) {
            ready.push_back(response_queue_.front().value);
            response_queue_.pop_front();
        }
    }

    std::vector<MemResponse> step_responses(int now) {
        std::vector<MemResponse> ready;
        step_responses(now, ready);
        return ready;
    }

    D2DLinkStatistics statistics(int elapsed_cycles) const {
        D2DLinkStatistics result = statistics_;
        const double duration =
            static_cast<double>(std::max(1, elapsed_cycles));
        result.forward_utilization = std::min(
            1.0, static_cast<double>(result.forward_busy_cycles) / duration);
        result.reverse_utilization = std::min(
            1.0, static_cast<double>(result.reverse_busy_cycles) / duration);
        return result;
    }

private:
    template <typename T> struct Transit {
        T value;
        int ready_cycle;
    };

    double bandwidth_bytes_per_cycle_;
    int latency_cycles_;
    std::size_t queue_capacity_;
    double next_forward_cycle_{};
    double next_reverse_cycle_{};
    std::deque<Transit<MemRequest>> request_queue_;
    std::deque<Transit<MemResponse>> response_queue_;
    D2DLinkStatistics statistics_{};

    double serialization_cycles(int bytes) const {
        return static_cast<double>(bytes) / bandwidth_bytes_per_cycle_;
    }

    static int arrival_cycle(double fractional_cycle) {
        return static_cast<int>(std::ceil(fractional_cycle - 1.0e-12));
    }
};

// Owns one of the four supported topologies. Routing is explicit and fixed;
// D2DLink instances are shared by the routes that physically traverse them.
class MemorySystem {
public:
    MemorySystem(int cache_size_bytes, int line_size_bytes,
                 int hit_latency_cycles, int l2_bandwidth_bytes_per_cycle,
                 int l2_write_bandwidth_bytes_per_cycle, int l2_queue_depth,
                 bool bypass_cache, const MemorySystemConfig& memory_config,
                 double core_frequency_mhz, const TensorRegistry& tensors);

    L2Cache* l2() const {
        return l2_.get();
    }

    std::vector<int> step(int current_cycle) {
        last_step_cycle_ = current_cycle;
        admit_prefetch_queue();
        std::vector<int> completed = l2_->step(current_cycle);
        collect_l2_requests(current_cycle);
        launch_pending_requests(current_cycle);
        advance_forward_links(current_cycle);
        retry_controller_admission(current_cycle);

        if (hbm_) {
            for (const auto& response : hbm_->step(current_cycle)) {
                route_response(response, current_cycle);
            }
        }
        if (hbf_) {
            for (const auto& response : hbf_->step(current_cycle)) {
                route_response(response, current_cycle);
            }
        }

        launch_pending_responses(current_cycle);
        advance_cascaded_return(current_cycle);
        advance_gpu_return_links(current_cycle);

        // A response can evict a dirty L2 line. Start its writeback on the
        // appropriate route in this same cycle, as the integrated model did.
        collect_l2_requests(current_cycle);
        launch_pending_requests(current_cycle);
        if (pending_background_requests()) ++background_cycles_;
        return completed;
    }

    std::uint64_t background_cycles() const {
        return background_cycles_;
    }
    std::uint64_t prefetch_capacity_stalls() const {
        return prefetch_capacity_stalls_;
    }
    std::uint64_t prefetch_canceled() const {
        return prefetch_canceled_;
    }

    struct ReadAlias {
        MemoryTier tier;
        std::uint64_t token, address, bytes;
        std::uint64_t queue_id{};
        std::optional<PrefetchBuffer> buffer;
    };

    ReadAlias prefetch_range(int tensor_id, std::uint64_t offset,
                             std::uint64_t bytes) const {
        const auto& tensor = tensors_.at(tensor_id);
        const auto size = tensor.byte_size();
        const auto line_size =
            static_cast<std::uint64_t>(l2_->get_line_size_bytes());
        if (!tensor.preloaded || !tensor.read_only || !bytes || offset > size ||
            bytes > size - offset || offset % line_size || bytes % line_size)
            throw std::invalid_argument(
                "prefetch range must cover aligned lines of a preloaded "
                "read-only tensor");
        const auto tier = tensor.initial_memory_tier;
        const auto address = tensor.global_base_address + offset;
        return {tier, 0, address, bytes, 0, std::nullopt};
    }

    ReadAlias
    reserve_prefetch(int tensor_id, std::uint64_t offset, std::uint64_t bytes,
                     std::optional<PrefetchBuffer> buffer = std::nullopt) {
        auto alias = prefetch_range(tensor_id, offset, bytes);
        alias.buffer = buffer;
        alias.token =
            device(alias.tier)
                .reserve_prefetch(alias.address, alias.bytes, alias.buffer);
        return alias;
    }

    ReadAlias
    enqueue_prefetch(int tensor_id, std::uint64_t offset, std::uint64_t bytes,
                     std::optional<PrefetchBuffer> buffer = std::nullopt) {
        auto alias = prefetch_range(tensor_id, offset, bytes);
        alias.buffer = buffer;
        alias.queue_id = next_queue_id_++;
        queued_prefetches_.emplace(alias.queue_id, alias);
        prefetch_queue_.push_back(alias.queue_id);
        return alias;
    }

    void start_prefetch(const ReadAlias& alias) {
        device(alias.tier).start_prefetch(alias.token);
    }

    bool prefetch_ready(const ReadAlias& alias) const {
        const auto& current =
            alias.queue_id ? queued_prefetches_.at(alias.queue_id) : alias;
        return current.token &&
               device(current.tier).prefetch_ready(current.token);
    }

    void release_prefetch(const ReadAlias& alias, bool require_idle = true) {
        if (require_idle && !foreground_idle())
            throw std::logic_error("release requires foreground retirement");
        if (alias.queue_id) {
            const auto queued = queued_prefetches_.at(alias.queue_id);
            if (queued.token)
                device(queued.tier).release_prefetch(queued.token);
            else
                ++prefetch_canceled_;
            queued_prefetches_.erase(alias.queue_id);
        } else {
            device(alias.tier).release_prefetch(alias.token);
        }
    }

    void bind_read_aliases(std::vector<ReadAlias> aliases) {
        if (!foreground_idle())
            throw std::logic_error("cannot change active read aliases");
        std::sort(
            aliases.begin(), aliases.end(),
            [](const auto& a, const auto& b) { return a.address < b.address; });
        for (std::size_t i = 1; i < aliases.size(); ++i)
            if (aliases[i - 1].address + aliases[i - 1].bytes >
                aliases[i].address)
                throw std::invalid_argument("overlapping read aliases");
        read_aliases_ = std::move(aliases);
    }

    void
    bind_node_read_aliases(const std::vector<int>& nodes,
                               std::shared_ptr<std::vector<ReadAlias>> aliases) {
        for (const auto node : nodes) {
            if (node < 0) continue;
            if (static_cast<std::size_t>(node) >= node_read_aliases_.size())
                node_read_aliases_.resize(static_cast<std::size_t>(node) + 1);
            node_read_aliases_[static_cast<std::size_t>(node)] = aliases;
        }
    }
    void unbind_node_read_aliases(const std::vector<int>& nodes) {
        for (const auto node : nodes)
            if (node >= 0 &&
                static_cast<std::size_t>(node) < node_read_aliases_.size())
                node_read_aliases_[static_cast<std::size_t>(node)].reset();
    }

    std::size_t pending_background_requests() const {
        return (hbm_ ? hbm_->pending_background_requests() : 0) +
               (hbf_ ? hbf_->pending_background_requests() : 0);
    }

    bool foreground_idle() const {
        return l2_->idle() && request_routes_.empty() &&
               (!hbm_ || hbm_->pending_requests() == 0) &&
               (!hbf_ || hbf_->pending_requests() == 0);
    }

    bool idle() const {
        return foreground_idle() && pending_background_requests() == 0;
    }

    MemoryBackendStatistics hbm_backend_statistics() const {
        return hbm_ ? hbm_->backend_statistics() : MemoryBackendStatistics{};
    }

    HBFBackendStatistics hbf_backend_statistics() const {
        return hbf_ ? hbf_->hbf_statistics() : HBFBackendStatistics{};
    }

    std::map<std::string, std::vector<MemoryBandwidthSample>>
    bandwidth_samples() const {
        std::map<std::string, std::vector<MemoryBandwidthSample>> result;
        for (const auto& [tier, device] :
             std::array{std::pair{MemoryTier::HBM, hbm_.get()},
                        std::pair{MemoryTier::HBF, hbf_.get()}}) {
            if (device) {
                auto samples = device->bandwidth_samples();
                if (!samples.empty())
                    result.emplace(memory_tier_name(tier), std::move(samples));
            }
        }
        return result;
    }

    std::map<MemoryTier, TierCounterSnapshot> tier_counters() const {
        std::map<MemoryTier, TierCounterSnapshot> result;
        const auto snapshot = [&](MemoryTier tier, const MemoryBackend* device,
                                  const TierMemoryStatistics& traffic) {
            result.emplace(
                tier, TierCounterSnapshot{
                          traffic.read_bytes, traffic.write_bytes,
                                                    device ? device->device_counters() : DeviceCounters{},
                                                    traffic.read_completions, traffic.read_latency_cycles,
                                                    traffic.unadmitted_alias_read_bytes});
        };
        snapshot(MemoryTier::HBM, hbm_.get(), hbm_statistics_);
        snapshot(MemoryTier::HBF, hbf_.get(), hbf_statistics_);
        return result;
    }

    MemorySystemStatistics statistics(int current_cycle) const {
        MemorySystemStatistics result;
        result.hbm = hbm_statistics_;
        result.hbf = hbf_statistics_;
        const int elapsed_cycles =
            std::max(1, std::max(current_cycle, last_step_cycle_) + 1);
        if (gpu_to_hbm_) {
            result.gpu_to_hbm = gpu_to_hbm_->statistics(elapsed_cycles);
            result.has_gpu_to_hbm = true;
        }
        if (gpu_to_hbf_) {
            result.gpu_to_hbf = gpu_to_hbf_->statistics(elapsed_cycles);
            result.has_gpu_to_hbf = true;
        }
        if (hbm_to_hbf_) {
            result.hbm_to_hbf = hbm_to_hbf_->statistics(elapsed_cycles);
            result.has_hbm_to_hbf = true;
        }
        return result;
    }

private:
    struct RequestRoute {
        MemoryTier tier;
        int start_cycle;
    };

    int last_step_cycle_{};
    MemorySystemConfig memory_config_;
    const TensorRegistry& tensors_;
    std::unique_ptr<L2Cache> l2_;
    std::unique_ptr<MemoryBackend> hbm_;
    std::unique_ptr<MemoryBackend> hbf_;
    std::unique_ptr<D2DLink> gpu_to_hbm_;
    std::unique_ptr<D2DLink> gpu_to_hbf_;
    std::unique_ptr<D2DLink> hbm_to_hbf_;
    std::deque<MemRequest> pending_to_hbm_;
    std::deque<MemRequest> pending_to_hbf_;
    std::deque<MemRequest> pending_to_base_;
    std::deque<MemRequest> pending_hbm_to_hbf_;
    std::deque<MemRequest> pending_hbm_device_;
    std::deque<MemRequest> pending_hbf_device_;
    std::deque<MemResponse> pending_hbm_response_;
    std::deque<MemResponse> pending_hbf_response_;
    std::deque<MemResponse> pending_hbm_to_hbf_response_;
    std::vector<MemRequest> link_request_scratch_;
    std::vector<MemResponse> link_response_scratch_;
    std::unordered_map<std::uint64_t, RequestRoute> request_routes_;
    std::vector<ReadAlias> read_aliases_;
    std::vector<std::shared_ptr<std::vector<ReadAlias>>> node_read_aliases_;
    std::deque<std::uint64_t> prefetch_queue_;
    std::unordered_map<std::uint64_t, ReadAlias> queued_prefetches_;
    std::uint64_t background_cycles_{}, prefetch_capacity_stalls_{},
        prefetch_canceled_{};
    std::uint64_t next_queue_id_{1}, blocked_head_{}, blocked_epoch_{};

    void admit_prefetch_queue() {
        while (!prefetch_queue_.empty()) {
            const auto id = prefetch_queue_.front();
            const auto found = queued_prefetches_.find(id);
            if (found == queued_prefetches_.end()) {
                // The last consumer finished before this request could issue.
                prefetch_queue_.pop_front();
                continue;
            }
            auto& alias = found->second;
            auto& backend = device(alias.tier);
            const auto epoch = backend.residency_epoch();
            if (blocked_head_ == id && blocked_epoch_ == epoch) return;
            alias.token = backend.reserve_prefetch(alias.address, alias.bytes,
                                                   alias.buffer);
            if (!alias.token) {
                ++prefetch_capacity_stalls_;
                blocked_head_ = id;
                blocked_epoch_ = epoch;
                return;
            }
            backend.start_prefetch(alias.token);
            prefetch_queue_.pop_front();
        }
    }

    MemoryBackend& device(MemoryTier tier) const {
        auto* selected = tier == MemoryTier::HBM ? hbm_.get() : hbf_.get();
        if (!selected)
            throw std::invalid_argument("no active device for requested tier");
        return *selected;
    }
    TierMemoryStatistics hbm_statistics_{};
    TierMemoryStatistics hbf_statistics_{};

    void build_full_hbm(double core_frequency_mhz) {
        hbm_ = make_hbm_backend(memory_config_.hbm, core_frequency_mhz);
        gpu_to_hbm_ = std::make_unique<D2DLink>(memory_config_.gpu_to_hbm,
                                                 core_frequency_mhz);
    }

    void build_full_hbf(double core_frequency_mhz) {
        hbf_ = make_hbf_backend(memory_config_.hbf, core_frequency_mhz,
                                tensors_.address_layout().hbf_base);
        gpu_to_hbf_ = std::make_unique<D2DLink>(memory_config_.gpu_to_hbf,
                                                 core_frequency_mhz);
    }

    void build_co_located(double core_frequency_mhz) {
        hbm_ = make_hbm_backend(memory_config_.hbm, core_frequency_mhz);
        hbf_ = make_hbf_backend(memory_config_.hbf, core_frequency_mhz,
                                tensors_.address_layout().hbf_base);
        gpu_to_hbm_ = std::make_unique<D2DLink>(memory_config_.gpu_to_hbm,
                                                 core_frequency_mhz);
        gpu_to_hbf_ = std::make_unique<D2DLink>(memory_config_.gpu_to_hbf,
                                                 core_frequency_mhz);
    }

    void build_cascaded(double core_frequency_mhz) {
        hbm_ = make_hbm_backend(memory_config_.hbm, core_frequency_mhz);
        hbf_ = make_hbf_backend(memory_config_.hbf, core_frequency_mhz,
                                tensors_.address_layout().hbf_base);
        gpu_to_hbm_ = std::make_unique<D2DLink>(memory_config_.gpu_to_hbm,
                                                 core_frequency_mhz);
        hbm_to_hbf_ = std::make_unique<D2DLink>(memory_config_.hbm_to_hbf,
                                                 core_frequency_mhz);
    }

    MemoryTier effective_tier(MemoryTier requested) const {
        if (memory_config_.topology == MemoryTopology::FullHBM) {
            return MemoryTier::HBM;
        }
        if (memory_config_.topology == MemoryTopology::FullHBF) {
            return MemoryTier::HBF;
        }
        return requested;
    }

    void report_placement_override() const {
        if (memory_config_.topology == MemoryTopology::FullHBM &&
            tensors_.contains_tier(MemoryTier::HBF)) {
            std::clog << "MemorySystem: automatic placement override: "
                         "FullHBM routes HBF-tagged allocations to HBM.\n";
        } else if (memory_config_.topology == MemoryTopology::FullHBF &&
                   tensors_.contains_tier(MemoryTier::HBM)) {
            std::clog << "MemorySystem: automatic placement override: "
                         "FullHBF routes HBM-tagged allocations to HBF.\n";
        }
    }

    TierMemoryStatistics& statistics_for(MemoryTier tier) {
        return tier == MemoryTier::HBM ? hbm_statistics_ : hbf_statistics_;
    }

    void route_request(MemRequest request, int current_cycle) {
        tensors_.validate_access(request.global_addr, request.size_bytes,
                                 request.tier,
                                 request.operation != MemoryOperation::Read);
        const MemoryTier tier = effective_tier(request.tier);
        if (request.operation == MemoryOperation::Read) {
            const auto consumer = request.routing.consumer_node;
            const auto* bound =
                consumer >= 0 && static_cast<std::size_t>(consumer) <
                                     node_read_aliases_.size()
                    ? node_read_aliases_[static_cast<std::size_t>(consumer)]
                          .get()
                    : nullptr;
            const auto& aliases = bound ? *bound : read_aliases_;
            auto found = std::upper_bound(
                aliases.begin(), aliases.end(), request.global_addr,
                [](std::uint64_t address, const ReadAlias& alias) {
                    return address < alias.address;
                });
            if (found != aliases.begin()) {
                const auto& alias = *std::prev(found);
                if (alias.tier == tier &&
                    request.global_addr >= alias.address &&
                    request.global_addr - alias.address < alias.bytes &&
                    static_cast<std::uint64_t>(request.size_bytes) <=
                        alias.bytes - (request.global_addr - alias.address)) {
                    request.read_alias =
                        alias.queue_id
                            ? queued_prefetches_.at(alias.queue_id).token
                            : alias.token;
                    if (!request.read_alias)
                        statistics_for(tier).unadmitted_alias_read_bytes +=
                            request.size_bytes;
                }
            }
        }
        const auto [it, inserted] = request_routes_.emplace(
            request.request_id, RequestRoute{tier, current_cycle});
        if (!inserted) {
            throw std::logic_error(
                "duplicate memory request id in MemorySystem");
        }
        TierMemoryStatistics& stats = statistics_for(tier);
        ++stats.request_count;
        if (request.operation == MemoryOperation::Read) {
            ++stats.read_requests;
            stats.read_bytes +=
                static_cast<std::uint64_t>(std::max(1, request.size_bytes));
        } else {
            ++stats.write_requests;
            stats.write_bytes +=
                static_cast<std::uint64_t>(std::max(1, request.size_bytes));
        }

        switch (memory_config_.topology) {
            case MemoryTopology::FullHBM:
            case MemoryTopology::CoLocated:
                if (tier == MemoryTier::HBM) {
                    pending_to_hbm_.push_back(request);
                } else {
                    pending_to_hbf_.push_back(request);
                }
                break;
            case MemoryTopology::FullHBF:
                pending_to_hbf_.push_back(request);
                break;
            case MemoryTopology::Cascaded:
                pending_to_base_.push_back(request);
                break;
        }
    }

    void collect_l2_requests(int current_cycle) {
        for (const auto& request : l2_->take_backend_requests()) {
            route_request(request, current_cycle);
        }
    }

    static void launch_requests(D2DLink* link, std::deque<MemRequest>& pending,
                                int current_cycle) {
        while (link && !pending.empty() &&
               link->enqueue_request(pending.front(), current_cycle)) {
            pending.pop_front();
        }
    }

    static void launch_responses(D2DLink* link,
                                 std::deque<MemResponse>& pending,
                                 int current_cycle) {
        while (link && !pending.empty() &&
               link->enqueue_response(pending.front(), current_cycle)) {
            pending.pop_front();
        }
    }

    void launch_pending_requests(int current_cycle) {
        switch (memory_config_.topology) {
            case MemoryTopology::FullHBM:
                launch_requests(gpu_to_hbm_.get(), pending_to_hbm_, current_cycle);
                break;
            case MemoryTopology::FullHBF:
                launch_requests(gpu_to_hbf_.get(), pending_to_hbf_, current_cycle);
                break;
            case MemoryTopology::CoLocated:
                launch_requests(gpu_to_hbm_.get(), pending_to_hbm_, current_cycle);
                launch_requests(gpu_to_hbf_.get(), pending_to_hbf_, current_cycle);
                break;
            case MemoryTopology::Cascaded:
                launch_requests(gpu_to_hbm_.get(), pending_to_base_, current_cycle);
                launch_requests(hbm_to_hbf_.get(), pending_hbm_to_hbf_,
                                current_cycle);
                break;
        }
    }

    void route_first_hop_arrivals(const std::vector<MemRequest>& requests) {
        for (const auto& request : requests) {
            const auto route = request_routes_.find(request.request_id);
            if (route == request_routes_.end()) {
                throw std::logic_error(
                    "memory request lost its route metadata");
            }
            if (memory_config_.topology == MemoryTopology::Cascaded &&
                route->second.tier == MemoryTier::HBF) {
                // The HBM base die forwards HBF traffic without submitting it
                // to HBM DRAM or consuming HBM controller bandwidth.
                pending_hbm_to_hbf_.push_back(request);
            } else if (route->second.tier == MemoryTier::HBM) {
                pending_hbm_device_.push_back(request);
            } else {
                pending_hbf_device_.push_back(request);
            }
        }
    }

    void advance_forward_links(int current_cycle) {
        switch (memory_config_.topology) {
            case MemoryTopology::FullHBM:
            gpu_to_hbm_->step_requests(current_cycle, link_request_scratch_);
            route_first_hop_arrivals(link_request_scratch_);
                break;
            case MemoryTopology::FullHBF:
            gpu_to_hbf_->step_requests(current_cycle, link_request_scratch_);
            route_first_hop_arrivals(link_request_scratch_);
                break;
            case MemoryTopology::CoLocated:
            gpu_to_hbm_->step_requests(current_cycle, link_request_scratch_);
            route_first_hop_arrivals(link_request_scratch_);
            gpu_to_hbf_->step_requests(current_cycle, link_request_scratch_);
            route_first_hop_arrivals(link_request_scratch_);
                break;
            case MemoryTopology::Cascaded:
            gpu_to_hbm_->step_requests(current_cycle, link_request_scratch_);
            route_first_hop_arrivals(link_request_scratch_);
                launch_requests(hbm_to_hbf_.get(), pending_hbm_to_hbf_,
                                current_cycle);
            hbm_to_hbf_->step_requests(current_cycle, link_request_scratch_);
            for (const auto& request : link_request_scratch_) {
                    pending_hbf_device_.push_back(request);
                }
                break;
        }
    }

    void retry_controller_admission(int current_cycle) {
        while (hbm_ && !pending_hbm_device_.empty() &&
               hbm_->try_submit(pending_hbm_device_.front(), current_cycle)) {
            pending_hbm_device_.pop_front();
        }
        while (hbf_ && !pending_hbf_device_.empty() &&
               hbf_->try_submit(pending_hbf_device_.front(), current_cycle)) {
            pending_hbf_device_.pop_front();
        }
    }

    void route_response(const MemResponse& response, int /*current_cycle*/) {
        const auto route = request_routes_.find(response.request_id);
        if (route == request_routes_.end()) {
            throw std::logic_error("memory response lost its route metadata");
        }
        switch (memory_config_.topology) {
            case MemoryTopology::FullHBM:
                pending_hbm_response_.push_back(response);
                break;
            case MemoryTopology::FullHBF:
                pending_hbf_response_.push_back(response);
                break;
            case MemoryTopology::CoLocated:
                if (route->second.tier == MemoryTier::HBM) {
                    pending_hbm_response_.push_back(response);
                } else {
                    pending_hbf_response_.push_back(response);
                }
                break;
            case MemoryTopology::Cascaded:
                if (route->second.tier == MemoryTier::HBM) {
                    pending_hbm_response_.push_back(response);
                } else {
                    pending_hbm_to_hbf_response_.push_back(response);
                }
                break;
        }
    }

    void launch_pending_responses(int current_cycle) {
        switch (memory_config_.topology) {
            case MemoryTopology::FullHBM:
                launch_responses(gpu_to_hbm_.get(), pending_hbm_response_,
                                 current_cycle);
                break;
            case MemoryTopology::FullHBF:
                launch_responses(gpu_to_hbf_.get(), pending_hbf_response_,
                                 current_cycle);
                break;
            case MemoryTopology::CoLocated:
                launch_responses(gpu_to_hbm_.get(), pending_hbm_response_,
                                 current_cycle);
                launch_responses(gpu_to_hbf_.get(), pending_hbf_response_,
                                 current_cycle);
                break;
            case MemoryTopology::Cascaded:
            launch_responses(hbm_to_hbf_.get(), pending_hbm_to_hbf_response_,
                             current_cycle);
                launch_responses(gpu_to_hbm_.get(), pending_hbm_response_,
                                 current_cycle);
                break;
        }
    }

    void advance_cascaded_return(int current_cycle) {
        if (!hbm_to_hbf_) {
            return;
        }
        hbm_to_hbf_->step_responses(current_cycle, link_response_scratch_);
        for (const auto& response : link_response_scratch_) {
            pending_hbm_response_.push_back(response);
        }
        launch_responses(gpu_to_hbm_.get(), pending_hbm_response_,
                         current_cycle);
    }

    void advance_gpu_return_links(int current_cycle) {
        const auto retire = [&](const std::vector<MemResponse>& responses) {
        for (const auto& response : responses) {
            const auto route = request_routes_.find(response.request_id);
            if (route == request_routes_.end()) {
                    throw std::logic_error(
                        "completed response has no request route");
            }
                TierMemoryStatistics& stats =
                    statistics_for(route->second.tier);
            ++stats.completed_requests;
            if (response.operation == MemoryOperation::Read) {
                ++stats.read_completions;
                stats.read_latency_cycles += static_cast<std::uint64_t>(
                    std::max(0, current_cycle - route->second.start_cycle));
            }
            stats.total_latency_cycles += static_cast<std::uint64_t>(
                std::max(0, current_cycle - route->second.start_cycle));
            request_routes_.erase(route);
            l2_->handle_response(response, current_cycle);
        }
        };
        switch (memory_config_.topology) {
        case MemoryTopology::FullHBM:
            gpu_to_hbm_->step_responses(current_cycle, link_response_scratch_);
            retire(link_response_scratch_);
            break;
        case MemoryTopology::FullHBF:
            gpu_to_hbf_->step_responses(current_cycle, link_response_scratch_);
            retire(link_response_scratch_);
            break;
        case MemoryTopology::CoLocated:
            gpu_to_hbm_->step_responses(current_cycle, link_response_scratch_);
            retire(link_response_scratch_);
            gpu_to_hbf_->step_responses(current_cycle, link_response_scratch_);
            retire(link_response_scratch_);
            break;
        case MemoryTopology::Cascaded:
            gpu_to_hbm_->step_responses(current_cycle, link_response_scratch_);
            retire(link_response_scratch_);
            break;
        }
    }
};

} // namespace GTSim
