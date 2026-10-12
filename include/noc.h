#ifndef NOC_H
#define NOC_H

#include <vector>
#include <deque>
#include <cmath>
#include <cstdlib>
#include <algorithm>

namespace GTSim {

enum class NoCPhase {
    FORWARD,  // Request traveling from source to destination SM
    RETURN    // Read data traveling back from destination to source SM
};

enum class NoCTopology {
    CROSSBAR,  // Full crossbar: single-hop, uniform latency
    MESH       // 2D mesh: Manhattan-distance routing
};

struct NoCRequest {
    int node_id;
    int src_sm_id;      // Originating SM
    int dst_sm_id;      // Target SM (for FORWARD) or source SM (for RETURN)
    int bytes;
    bool is_write;
    int bank_conflict;  // Bank conflict factor for SRAM access
    NoCPhase phase;
    int complete_cycle; // Cycle when this request completes NoC traversal
    std::vector<int> route;  // Link IDs along XY path (used by MeshNoC)
    int enqueue_cycle;        // Cycle when request entered NoC
    int nominal_complete_cycle;  // completion without NoC arbitration delay
};

// Abstract base class for Network-on-Chip implementations.
// All topologies share the same interface so GPU routing logic is topology-agnostic.
class NoC {
public:
    int bandwidth_bytes_per_cycle;
    int queue_depth;
    int sms_per_cluster;  // 0 = all SMs in one cluster (no restriction)

    std::deque<NoCRequest> in_flight;
    long long stat_enqueued_requests;
    long long stat_completed_requests;
    long long stat_enqueued_bytes;
    long long stat_completed_bytes;
    long long stat_total_residence_cycles;
    long long stat_total_nominal_cycles;
    long long stat_total_internal_delay_cycles;
    int stat_peak_inflight;
    int stat_first_enqueue_cycle;
    int stat_last_completion_cycle;
    long long stat_active_cycles;  // Cycles where at least one request is in-flight

    NoC(int bandwidth, int depth, int sms_per_cluster_ = 0)
        : bandwidth_bytes_per_cycle(bandwidth), queue_depth(depth),
          sms_per_cluster(sms_per_cluster_),
          stat_enqueued_requests(0),
          stat_completed_requests(0),
          stat_enqueued_bytes(0),
          stat_completed_bytes(0),
          stat_total_residence_cycles(0),
          stat_total_nominal_cycles(0),
          stat_total_internal_delay_cycles(0),
          stat_peak_inflight(0),
          stat_first_enqueue_cycle(-1),
          stat_last_completion_cycle(-1),
          stat_active_cycles(0) {}

    // Cluster helpers: cluster_id = sm_id / sms_per_cluster
    int cluster_of(int sm_id) const {
        return (sms_per_cluster > 0) ? sm_id / sms_per_cluster : 0;
    }
    bool same_cluster(int sm_a, int sm_b) const {
        return cluster_of(sm_a) == cluster_of(sm_b);
    }

    virtual ~NoC() = default;

    // Enqueue a forward request (subpartition -> target SM's SRAM)
    virtual bool enqueue(int node_id, int src_sm_id, int dst_sm_id,
                         int bytes, bool is_write, int bank_conflict,
                         int current_cycle) = 0;

    // Enqueue a return request (read data traveling back to source SM)
    virtual bool enqueue_return(int node_id, int src_sm_id, int dst_sm_id,
                                int bytes, int bank_conflict,
                                int current_cycle) = 0;

    // Step the NoC: returns all requests that completed this cycle
    // Default implementation: drain in_flight queue respecting bandwidth cap
    virtual std::vector<NoCRequest> step(int current_cycle) {
        if (!in_flight.empty()) stat_active_cycles++;

        std::vector<NoCRequest> completed;
        int bytes_this_cycle = 0;

        auto it = in_flight.begin();
        while (it != in_flight.end()) {
            if (it->complete_cycle <= current_cycle) {
                // Cap arbitration charge at one cycle worth of bandwidth so
                // requests larger than per-cycle BW do not deadlock forever.
                int arbitration_bytes =
                    std::min(it->bytes, std::max(1, bandwidth_bytes_per_cycle));
                if (bytes_this_cycle + arbitration_bytes <= bandwidth_bytes_per_cycle) {
                    bytes_this_cycle += arbitration_bytes;
                    record_completion(*it, current_cycle);
                    completed.push_back(*it);
                    it = in_flight.erase(it);
                } else {
                    // Bandwidth exceeded: delay by 1 cycle
                    it->complete_cycle = current_cycle + 1;
                    ++it;
                }
            } else {
                ++it;
            }
        }
        return completed;
    }

    virtual bool is_full() const {
        return static_cast<int>(in_flight.size()) >= queue_depth;
    }

protected:
    void record_enqueue(const NoCRequest& req) {
        stat_enqueued_requests++;
        stat_enqueued_bytes += req.bytes;
        stat_peak_inflight = std::max(stat_peak_inflight,
                                      static_cast<int>(in_flight.size()));
        if (stat_first_enqueue_cycle < 0) {
            stat_first_enqueue_cycle = req.enqueue_cycle;
        }
    }

    void record_completion(const NoCRequest& req, int current_cycle) {
        stat_completed_requests++;
        stat_completed_bytes += req.bytes;
        int residence = std::max(0, current_cycle - req.enqueue_cycle);
        int nominal = std::max(0, req.nominal_complete_cycle - req.enqueue_cycle);
        stat_total_residence_cycles += residence;
        stat_total_nominal_cycles += nominal;
        stat_total_internal_delay_cycles += std::max(0, residence - nominal);
        stat_last_completion_cycle =
            std::max(stat_last_completion_cycle, current_cycle);
    }
};

// ============================================================================
// Crossbar NoC: single-hop interconnect between all SMs.
// All SM pairs have equal latency (full crossbar).
// ============================================================================
class CrossbarNoC : public NoC {
public:
    int latency_cycles;

    CrossbarNoC(int latency, int bandwidth, int depth, int sms_per_cluster_ = 0)
        : NoC(bandwidth, depth, sms_per_cluster_), latency_cycles(latency) {}

    bool enqueue(int node_id, int src_sm_id, int dst_sm_id,
                 int bytes, bool is_write, int bank_conflict,
                 int current_cycle) override {
        if (!same_cluster(src_sm_id, dst_sm_id)) return false;
        if (is_full()) return false;
        int complete = current_cycle + latency_cycles;
        in_flight.push_back({node_id, src_sm_id, dst_sm_id, bytes, is_write,
                             bank_conflict, NoCPhase::FORWARD, complete, {},
                             current_cycle, complete});
        record_enqueue(in_flight.back());
        return true;
    }

    bool enqueue_return(int node_id, int src_sm_id, int dst_sm_id,
                        int bytes, int bank_conflict,
                        int current_cycle) override {
        if (!same_cluster(src_sm_id, dst_sm_id)) return false;
        if (is_full()) return false;
        int complete = current_cycle + latency_cycles;
        in_flight.push_back({node_id, src_sm_id, dst_sm_id, bytes, false,
                             bank_conflict, NoCPhase::RETURN, complete, {},
                             current_cycle, complete});
        record_enqueue(in_flight.back());
        return true;
    }
    // Crossbar step with a single global NoC budget per cycle.
    // This keeps aggregate crossbar throughput bounded by
    // bandwidth_bytes_per_cycle, regardless of destination fanout.
    std::vector<NoCRequest> step(int current_cycle) override {
        if (!in_flight.empty()) stat_active_cycles++;

        std::vector<NoCRequest> completed;
        int bytes_this_cycle = 0;

        auto it = in_flight.begin();
        while (it != in_flight.end()) {
            if (it->complete_cycle <= current_cycle) {
                int arbitration_bytes =
                    std::min(it->bytes, std::max(1, bandwidth_bytes_per_cycle));
                if (bytes_this_cycle + arbitration_bytes <= bandwidth_bytes_per_cycle) {
                    bytes_this_cycle += arbitration_bytes;
                    record_completion(*it, current_cycle);
                    completed.push_back(*it);
                    it = in_flight.erase(it);
                } else {
                    it->complete_cycle = current_cycle + 1;
                    ++it;
                }
            } else {
                ++it;
            }
        }
        return completed;
    }
};

// ============================================================================
// Mesh NoC: 2D grid interconnect.
// SMs are laid out in a rows x cols grid. SM i is at position
//   row = i / cols,  col = i % cols
// Latency = manhattan_distance * per_hop_latency_cycles.
// Minimum latency is per_hop_latency_cycles (even for same-row/col neighbors).
// ============================================================================
class MeshNoC : public NoC {
public:
    int base_latency_cycles;    // Fixed injection + ejection overhead (DSMEM protocol)
    int per_hop_latency_cycles; // Per-hop router pipeline latency
    int mesh_rows;  // Intra-cluster mesh rows
    int mesh_cols;  // Intra-cluster mesh columns

    // Logical SM → physical mesh node mapping.
    // sm_to_physical[logical_sm] = physical_node
    // physical_node → row = node / mesh_cols, col = node % mesh_cols
    // Empty vector = identity mapping (backward compatible).
    std::vector<int> sm_to_physical;

    // Per-link bandwidth tracking
    int link_bandwidth_bytes_per_cycle;  // Bandwidth limit per physical link
    int num_links;                        // Total number of links in the mesh
    std::vector<int> link_bytes_this_cycle;  // Per-link bandwidth usage
    int last_reset_cycle;                 // For lazy reset of per-cycle counters

    MeshNoC(int base_latency, int per_hop_latency, int bandwidth, int depth,
            int rows, int cols, int sms_per_cluster_ = 0,
            int link_bandwidth = 0,
            const std::vector<int>& mapping = {})
        : NoC(bandwidth, depth, sms_per_cluster_),
          base_latency_cycles(base_latency),
          per_hop_latency_cycles(per_hop_latency),
          mesh_rows(rows), mesh_cols(cols),
          sm_to_physical(mapping),
          link_bandwidth_bytes_per_cycle(link_bandwidth > 0 ? link_bandwidth : bandwidth),
          last_reset_cycle(-1) {
        num_links = num_horizontal_links() + num_vertical_links();
        link_bytes_this_cycle.assign(num_links, 0);
    }

    // Map logical SM ID to physical mesh node ID
    int to_physical(int sm_id) const {
        if (sm_to_physical.empty()) return sm_id;  // identity
        if (sm_id >= 0 && sm_id < static_cast<int>(sm_to_physical.size()))
            return sm_to_physical[sm_id];
        return sm_id;  // fallback
    }

    // Link topology helpers
    int num_horizontal_links() const { return mesh_rows * (mesh_cols - 1); }
    int num_vertical_links() const { return (mesh_rows - 1) * mesh_cols; }
    int h_link_id(int row, int col) const { return row * (mesh_cols - 1) + col; }
    int v_link_id(int row, int col) const { return num_horizontal_links() + row * mesh_cols + col; }

    // Compute XY dimension-order route as a list of link IDs.
    // Applies logical→physical mapping before computing row/col.
    std::vector<int> compute_route(int src_sm_id, int dst_sm_id) const {
        int psrc = to_physical(src_sm_id);
        int pdst = to_physical(dst_sm_id);
        int r0 = psrc / mesh_cols, c0 = psrc % mesh_cols;
        int r1 = pdst / mesh_cols, c1 = pdst % mesh_cols;

        std::vector<int> links;
        // X first (horizontal)
        int c = c0;
        while (c != c1) {
            int next_c = c + (c1 > c0 ? 1 : -1);
            int min_c = std::min(c, next_c);
            links.push_back(h_link_id(r0, min_c));
            c = next_c;
        }
        // Then Y (vertical)
        int r = r0;
        while (r != r1) {
            int next_r = r + (r1 > r0 ? 1 : -1);
            int min_r = std::min(r, next_r);
            links.push_back(v_link_id(min_r, c1));
            r = next_r;
        }
        return links;
    }

    // Compute Manhattan distance using mapped physical mesh positions
    int hop_count(int sm_a, int sm_b) const {
        int pa = to_physical(sm_a);
        int pb = to_physical(sm_b);
        int row_a = pa / mesh_cols, col_a = pa % mesh_cols;
        int row_b = pb / mesh_cols, col_b = pb % mesh_cols;
        int dist = std::abs(row_a - row_b) + std::abs(col_a - col_b);
        return dist > 0 ? dist : 1;
    }

    int compute_latency(int src_sm_id, int dst_sm_id) const {
        return base_latency_cycles + hop_count(src_sm_id, dst_sm_id) * per_hop_latency_cycles;
    }

    bool enqueue(int node_id, int src_sm_id, int dst_sm_id,
                 int bytes, bool is_write, int bank_conflict,
                 int current_cycle) override {
        if (!same_cluster(src_sm_id, dst_sm_id)) return false;
        if (is_full()) return false;
        int latency = compute_latency(src_sm_id, dst_sm_id);
        auto route = compute_route(src_sm_id, dst_sm_id);
        int complete = current_cycle + latency;
        in_flight.push_back({node_id, src_sm_id, dst_sm_id, bytes, is_write,
                             bank_conflict, NoCPhase::FORWARD, complete,
                             std::move(route), current_cycle, complete});
        record_enqueue(in_flight.back());
        return true;
    }

    bool enqueue_return(int node_id, int src_sm_id, int dst_sm_id,
                        int bytes, int bank_conflict,
                        int current_cycle) override {
        if (!same_cluster(src_sm_id, dst_sm_id)) return false;
        if (is_full()) return false;
        int latency = compute_latency(dst_sm_id, src_sm_id);
        // Return path: dst -> src
        auto route = compute_route(dst_sm_id, src_sm_id);
        int complete = current_cycle + latency;
        in_flight.push_back({node_id, src_sm_id, dst_sm_id, bytes, false,
                             bank_conflict, NoCPhase::RETURN, complete,
                             std::move(route), current_cycle, complete});
        record_enqueue(in_flight.back());
        return true;
    }

    // Per-link bandwidth enforcement
    std::vector<NoCRequest> step(int current_cycle) override {
        if (!in_flight.empty()) stat_active_cycles++;

        // Lazy reset: clear per-link counters when cycle changes
        if (current_cycle != last_reset_cycle) {
            std::fill(link_bytes_this_cycle.begin(), link_bytes_this_cycle.end(), 0);
            last_reset_cycle = current_cycle;
        }

        std::vector<NoCRequest> completed;
        auto it = in_flight.begin();
        while (it != in_flight.end()) {
            if (it->complete_cycle <= current_cycle) {
                // Check per-link bandwidth for all links on this request's route
                bool can_complete = true;
                int arbitration_bytes =
                    std::min(it->bytes, std::max(1, link_bandwidth_bytes_per_cycle));
                for (int link_id : it->route) {
                    if (link_bytes_this_cycle[link_id] + arbitration_bytes >
                        link_bandwidth_bytes_per_cycle) {
                        can_complete = false;
                        break;
                    }
                }
                if (can_complete) {
                    // Consume bandwidth on all route links
                    for (int link_id : it->route) {
                        link_bytes_this_cycle[link_id] += arbitration_bytes;
                    }
                    record_completion(*it, current_cycle);
                    completed.push_back(*it);
                    it = in_flight.erase(it);
                } else {
                    // Bandwidth exceeded on at least one link: delay by 1 cycle
                    it->complete_cycle = current_cycle + 1;
                    ++it;
                }
            } else {
                ++it;
            }
        }
        return completed;
    }
};

} // namespace GTSim

#endif // NOC_H
