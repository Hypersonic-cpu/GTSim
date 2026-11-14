#include "memory/base.h"
#include "memory/sram.h"
#include <cassert>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <deque>
#include <tuple>
namespace GTSim {
namespace {
class SramBackend final : public MemoryBackend {
public:
    SramBackend(std::unique_ptr<MemoryBackend> backend, const HBFConfig& config,
                std::uint64_t base)
        : backend_(std::move(backend)),
          config_{config.controller.stacks,
                  config.controller.dies_per_stack,
                  config.controller.planes_per_die,
                  config.controller.page_bytes,
                  config.controller.sram_bytes_per_plane,
          config.controller.base_die_sram_bytes_per_stack},
          enabled_(config.controller.enable_read_sram &&
                   config.prefetch_engine.enabled),
          cancel_on_miss_(config.prefetch_engine.cancel_queued_on_demand_miss),
          queue_capacity_(static_cast<std::size_t>(
              std::max(1, config.prefetch_engine.queue_capacity))),
          base_(base),
          interleave_(config.prefetch_hyperpage_order),
          lanes_(config.controller.planes_per_command_group),
          groups_(static_cast<std::uint64_t>(config.controller.stacks) *
                  config.controller.dies_per_stack *
                  config.controller.planes_per_die /
                  std::max(1, config.controller.planes_per_command_group)),
          sram_(config_,
                [this](auto address) { return backend_->locate(address); }) {
        if (interleave_ &&
            (config.controller.geometry != HBFGeometryProfile::GroupedPlanes ||
             !lanes_ || !groups_))
            throw std::invalid_argument(
                "hyperpage prefetch ordering requires grouped geometry");
        if (enabled_ && !backend_->supports_sram_paths())
            throw std::invalid_argument(
                "native backend does not support SRAM transfer paths");
    }
    bool try_submit(const MemRequest& request, int now) override {
        assert(request.request_id < background_id_base);
        if (request.read_path != MemoryReadPath::Host)
            throw std::invalid_argument(
                "explicit SRAM paths require a managed read alias");
        auto routed = request;
        if (request.read_alias) {
            const auto& region = regions_.at(request.read_alias);
            bool ready = true;
            const auto last = (request.global_addr + request.size_bytes - 1) /
                              config_.page_bytes;
            for (auto page = request.global_addr / config_.page_bytes;
                 page <= last; ++page)
                ready &= region.ready_pages.contains(page);
            routed.read_path =
                ready ? MemoryReadPath::Resident : MemoryReadPath::Host;
        }
        routed.read_alias = 0;
        if (!backend_->try_submit(routed, now)) return false;
        if (request.read_alias) {
            ++regions_.at(request.read_alias).consumers;
            (routed.read_path == MemoryReadPath::Resident
                 ? resident_read_bytes_
                 : fallback_read_bytes_) += request.size_bytes;
        }
        const auto inserted = pending_.emplace(request.request_id, request).second;
        assert(inserted);
        (void)inserted;
        return true;
    }
    std::vector<MemResponse> step(int now) override {
        advance_prefetch(now);
        submit_background(now);
        std::vector<MemResponse> result;
        for (auto response : backend_->step(now)) {
            if (auto fill = background_.find(response.request_id);
                fill != background_.end()) {
                auto token = fill->second;
                background_.erase(fill);
                if (!response.completed || response.error_code ||
                    response.shadow_digest !=
                        memory_shadow_digest(
                            response.global_addr,
                            static_cast<std::uint64_t>(
                                std::max(1, response.size_bytes))))
                    throw std::runtime_error("SRAM fill failed");
                auto& region = regions_.at(token);
                --region.outstanding;
                region.ready_pages.insert(response.global_addr /
                                          config_.page_bytes);
                if (region.ready()) {
                    --active_fills_;
                    ++completed_fills_;
            }
                reclaim(token);
                continue;
            }
            const auto request = pending_.at(response.request_id);
            pending_.erase(response.request_id);
            result.push_back(response);
            if (request.read_alias) {
                --regions_.at(request.read_alias).consumers;
                reclaim(request.read_alias);
            }
        }
        return result;
    }
    std::size_t pending_requests() const override {
        return pending_.size();
    }
    std::uint64_t
    reserve_prefetch(std::uint64_t address, std::uint64_t bytes,
                     std::optional<PrefetchBuffer> buffer) override {
        if (!enabled_ || !bytes || address % 64 || bytes % 64 ||
            address > std::numeric_limits<std::uint64_t>::max() - bytes)
            throw std::invalid_argument(
                "prefetch requires enabled SRAM and a nonempty aligned range");
        backend_->locate(address + bytes - 1);
        Region region;
        region.address = address;
        region.bytes = bytes;
        if (interleave_) {
            for (auto cursor = address; cursor < address + bytes;) {
                const auto count = std::min<std::uint64_t>(
                    config_.page_bytes - cursor % config_.page_bytes,
                    address + bytes - cursor);
                region.fills.push_back({cursor, count});
                cursor += count;
            }
            std::stable_sort(
                region.fills.begin(), region.fills.end(),
                [this](const auto& a, const auto& b) {
                    const auto key = [this](std::uint64_t address) {
                        const auto page =
                            (address - base_) / config_.page_bytes;
                        return std::tuple{page / (groups_ * lanes_),
                                          page % lanes_,
                                          (page / lanes_) % groups_};
                };
                    return key(a.address) < key(b.address);
            });
        }
        auto reservation = sram_.reserve(address, bytes, buffer);
        if (!reservation) return 0;
        region.reservation = std::move(*reservation);
        const auto token = next_token_++;
        reserved_bytes_ += region.reservation.pages.size() * config_.page_bytes;
        peak_reserved_bytes_ = std::max(peak_reserved_bytes_, reserved_bytes_);
        regions_.emplace(token, std::move(region));
        return token;
    }

    std::uint64_t enqueue_prefetch(
        std::uint64_t address, std::uint64_t bytes,
        std::optional<PrefetchBuffer> buffer) override {
        if (!enabled_ || !bytes || address % 64 || bytes % 64 ||
            address > std::numeric_limits<std::uint64_t>::max() - bytes)
            throw std::invalid_argument(
                "queued prefetch requires enabled SRAM and an aligned range");
        if (queued_.size() >= queue_capacity_)
            throw std::runtime_error("prefetch descriptor queue is full");
        backend_->locate(address + bytes - 1);
        const auto id = next_queue_id_++;
        queued_.emplace(id, Queued{address, bytes, 0, buffer});
        queue_order_.push_back(id);
        ++queued_total_;
        return id;
    }

    void advance_prefetch(int now) override {
        if (admission_cycle_ == now) return;
        admission_cycle_ = now;
        while (!queue_order_.empty()) {
            const auto id = queue_order_.front();
            auto found = queued_.find(id);
            if (found == queued_.end()) {
                queue_order_.pop_front();
                continue;
            }
            if (blocked_head_ == id && blocked_epoch_ == residency_epoch_)
                return;
            const auto token = reserve_prefetch(found->second.address,
                                                found->second.bytes,
                                                found->second.buffer);
            if (!token) {
                ++capacity_stalls_;
                blocked_head_ = id;
                blocked_epoch_ = residency_epoch_;
                return;
            }
            start_prefetch(token);
            found->second.token = token;
            queue_order_.pop_front();
        }
    }

    std::uint64_t prefetch_alias(std::uint64_t handle) const override {
        if (handle < queue_id_base) return handle;
        const auto found = queued_.find(handle);
        return found == queued_.end() ? 0 : found->second.token;
    }

    void on_demand_miss(std::uint64_t handle, std::uint64_t address,
                        std::uint64_t bytes) override {
        if (!cancel_on_miss_ || handle < queue_id_base) return;
        const auto found = queued_.find(handle);
        if (found == queued_.end() || found->second.token ||
            address + bytes <= found->second.address ||
            found->second.address + found->second.bytes <= address)
            return;
        queued_.erase(found);
        ++canceled_;
    }

    void start_prefetch(std::uint64_t token) override {
        auto& region = regions_.at(token);
        if (region.started || region.released)
            throw std::invalid_argument("invalid prefetch start");
        region.started = true;
        const auto position = std::lower_bound(
            active_regions_.begin(), active_regions_.end(), token);
        active_regions_.insert(position, token);
        ++started_fills_;
        ++active_fills_;
        peak_active_fills_ = std::max(peak_active_fills_, active_fills_);
    }

    bool prefetch_ready(std::uint64_t token) const override {
        if (token >= queue_id_base) {
            const auto found = queued_.find(token);
            return found != queued_.end() && found->second.token &&
                   prefetch_ready(found->second.token);
        }
        const auto& region = regions_.at(token);
        return region.ready() && !region.released;
    }

    void release_prefetch(std::uint64_t token) override {
        if (token >= queue_id_base) {
            const auto found = queued_.find(token);
            if (found == queued_.end()) return;
            if (!found->second.token) {
                queued_.erase(found);
                return;
            }
            token = found->second.token;
            queued_.erase(found);
        }
        auto& region = regions_.at(token);
        if (region.released)
            throw std::invalid_argument("duplicate prefetch release");
        region.released = true;
        reclaim(token);
    }

    std::uint64_t residency_epoch() const override {
        return residency_epoch_;
    }

    std::uint64_t prefetch_capacity_stalls() const override {
        return capacity_stalls_;
    }
    std::uint64_t prefetch_canceled() const override { return canceled_; }

    std::size_t pending_background_requests() const override {
        return active_fills_;
    }
    DeviceCounters device_counters() const override {
        auto counters = backend_->device_counters();
        if (enabled_) {
            counters.cumulative["prefetch_bytes"] = prefetch_bytes_;
            counters.cumulative["prefetch_queued"] = queued_total_;
            counters.cumulative["prefetch_started"] = started_fills_;
            counters.cumulative["prefetch_completed"] = completed_fills_;
            counters.cumulative["prefetch_canceled"] = canceled_;
            counters.cumulative["resident_read_bytes"] = resident_read_bytes_;
            counters.cumulative["fallback_read_bytes"] = fallback_read_bytes_;
            counters.gauges["sram_reserved_bytes"] = reserved_bytes_;
            counters.gauges["sram_peak_reserved_bytes"] = peak_reserved_bytes_;
            counters.gauges["prefetch_peak_outstanding"] = peak_active_fills_;
            counters.gauges.merge(sram_.gauges());
        }
        return counters;
    }
    bool supports_sram_paths() const override {
        return backend_->supports_sram_paths();
    }
    MemoryLocation locate(std::uint64_t a) const override {
        return backend_->locate(a);
    }
    bool supports_prefetch() const override {
        return backend_->supports_sram_paths();
    }
    MemoryBackendStatistics backend_statistics() const override {
        return backend_->backend_statistics();
    }
    HBFBackendStatistics hbf_statistics() const override {
        return backend_->hbf_statistics();
    }
    std::vector<MemoryBandwidthSample> bandwidth_samples() const override {
        return backend_->bandwidth_samples();
    }
    void preload_region(std::uint64_t a, std::uint64_t n, bool r) override {
        backend_->preload_region(a, n, r);
    }

private:
    std::unique_ptr<MemoryBackend> backend_;
    ReadSramAllocator::Config config_;
    bool enabled_;
    bool cancel_on_miss_;
    std::size_t queue_capacity_;
    std::uint64_t base_;
    bool interleave_;
    std::uint64_t lanes_, groups_;
    static constexpr std::uint64_t queue_id_base = std::uint64_t{1} << 62;
    static constexpr std::uint64_t background_id_base = std::uint64_t{1} << 63;
    struct Fill {
        std::uint64_t address, bytes;
    };
    struct Region {
        std::uint64_t address{}, bytes{}, submitted{}, outstanding{},
            consumers{};
        bool started{}, released{};
        ReadSramAllocator::Reservation reservation;
        std::unordered_set<std::uint64_t> ready_pages;
        std::vector<Fill> fills;
        std::size_t next_fill{};
        bool ready() const {
            return started && submitted == bytes && !outstanding;
        }
    };
    struct Queued {
        std::uint64_t address{}, bytes{}, token{};
        std::optional<PrefetchBuffer> buffer;
    };

    void submit_background(int time) {
        while (!active_regions_.empty()) {
            const auto token = active_regions_.front();
            const auto found = regions_.find(token);
            if (found == regions_.end()) {
                active_regions_.pop_front();
                continue;
            }
            auto& region = found->second;
            while (region.submitted < region.bytes) {
                const auto address =
                    interleave_ ? region.fills.at(region.next_fill).address
                                : region.address + region.submitted;
                const auto bytes = std::min<std::uint64_t>(
                    config_.page_bytes - address % config_.page_bytes,
                    interleave_ ? region.fills.at(region.next_fill).bytes
                                : region.bytes - region.submitted);
                assert(next_background_id_ >= background_id_base);
                assert(next_background_id_ <
                       std::numeric_limits<std::uint64_t>::max());
                const auto id = next_background_id_++;
                const MemRequest fill{id,
                                      address,
                                      static_cast<int>(bytes),
                                      MemoryOperation::Read,
                                      MemoryTier::HBF,
                                      {},
                                      0,
                                      MemoryReadPath::Prefetch,
                                      memory_shadow_digest(address, bytes)};
                if (!backend_->try_submit(fill, time))
                    return;
                background_.emplace(id, token);
                region.submitted += bytes;
                ++region.next_fill;
                ++region.outstanding;
                prefetch_bytes_ += bytes;
            }
            active_regions_.pop_front();
        }
    }

    void reclaim(std::uint64_t token) {
        const auto& region = regions_.at(token);
        if (!region.released || region.consumers ||
            (region.started && !region.ready()))
            return;
        sram_.release(region.reservation);
        reserved_bytes_ -= region.reservation.pages.size() * config_.page_bytes;
        regions_.erase(token);
        for (auto active = active_regions_.begin();
             active != active_regions_.end(); ++active) {
            if (*active == token) {
                active_regions_.erase(active);
                break;
            }
        }
        ++residency_epoch_;
    }

    ReadSramAllocator sram_;
    std::map<std::uint64_t, Region> regions_;
    std::deque<std::uint64_t> active_regions_;
    std::unordered_map<std::uint64_t, std::uint64_t> background_;
    std::unordered_map<std::uint64_t, Queued> queued_;
    std::deque<std::uint64_t> queue_order_;
    std::uint64_t next_token_{1}, next_queue_id_{queue_id_base},
        next_background_id_{background_id_base};
    std::uint64_t residency_epoch_{};
    std::uint64_t reserved_bytes_{}, peak_reserved_bytes_{};
    std::uint64_t started_fills_{}, completed_fills_{}, active_fills_{},
        peak_active_fills_{};
    std::uint64_t prefetch_bytes_{}, resident_read_bytes_{},
        fallback_read_bytes_{}, queued_total_{};
    std::unordered_map<std::uint64_t, MemRequest> pending_;
    int admission_cycle_{-1};
    std::uint64_t blocked_head_{}, blocked_epoch_{}, capacity_stalls_{},
        canceled_{};
};
} // namespace
std::unique_ptr<MemoryBackend>
make_sram_backend(std::unique_ptr<MemoryBackend> backend,
                  const HBFConfig& config, std::uint64_t base) {
    return std::make_unique<SramBackend>(std::move(backend), config, base);
}
} // namespace GTSim
