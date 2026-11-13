#ifndef TMEM_H
#define TMEM_H

#include "dag_node.h"
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <array>

namespace GTSim {

class TMEMUnit {
public:
    TMEMUnit(int cols_per_cta = 256, int issue_interval = 1)
        : max_cols_per_cta(cols_per_cta),
          issue_interval_cycles(issue_interval) {}

    void init_warp_state(int warp_count) {
        next_issue_cycle.assign(std::max(0, warp_count), 0);
    }

    // Launch-local TMEM allocation and paired-CTA rendezvous state must not
    // retain node/CTA IDs after a kernel is detached.  The actual TMEM unit
    // remains part of the persistent GPU.
    void reset() {
        next_issue_cycle.clear();
        cta_allocated_cols.clear();
        cta_issued_mma_seq.clear();
        cta_pair_issue_seq.clear();
        node_pair_seq.clear();
        pair_pending.clear();
        pair_issue_pending.clear();
        node_issue_grant_cycle.clear();
        release_tokens.clear();
    }

    bool can_issue(const DAGNode* node, int current_cycle) {
        if (!node || !is_tmem_op(node)) {
            return true;
        }
        if (node->warp_id < 0 ||
            node->warp_id >= static_cast<int>(next_issue_cycle.size())) {
            return false;
        }
        if (current_cycle < next_issue_cycle[node->warp_id]) {
            return false;
        }

        int tb_id = cta_key(node);
        int allocated = 0;
        auto it = cta_allocated_cols.find(tb_id);
        if (it != cta_allocated_cols.end()) {
            allocated = it->second;
        }

        if (node->op_type == OpType::TCGEN05_ALLOC) {
            int req = std::max(0, node->tmem_cols);
            return allocated + req <= max_cols_per_cta;
        }

        // tcgen05 ops other than alloc need prior allocation.
        if (allocated <= 0) {
            return false;
        }
        if (node->tmem_cols > 0 && node->tmem_cols > allocated) {
            return false;
        }
        if (is_paired_2cta_mma(node)) {
            return paired_mma_issue_ready(node, current_cycle);
        }
        return true;
    }

    void on_issue(const DAGNode* node, int current_cycle) {
        if (!node || !is_tmem_op(node)) {
            return;
        }
        if (node->warp_id >= 0 &&
            node->warp_id < static_cast<int>(next_issue_cycle.size())) {
            next_issue_cycle[node->warp_id] = current_cycle + std::max(1, issue_interval_cycles);
        }

        if (node->op_type == OpType::TCGEN05_ALLOC) {
            int req = std::max(0, node->tmem_cols);
            int& allocated = cta_allocated_cols[cta_key(node)];
            allocated = std::min(max_cols_per_cta, allocated + req);
            return;
        }

        if (node->op_type == OpType::TCGEN05_MMA) {
            int cta = cta_key(node);
            cta_issued_mma_seq[cta]++;
            node_issue_grant_cycle.erase(node->id);
            if (is_paired_2cta_mma(node)) {
                int seq = ++cta_pair_issue_seq[cta];
                node_pair_seq[node->id] = seq;
            }
            return;
        }

    }

    void on_complete(const DAGNode* node) {
        if (!node || node->op_type != OpType::TCGEN05_MMA) {
            return;
        }
    }

    // Pair rendezvous for 2-CTA UMMA completion.
    // Returns true if this node can retire now; false means it must be deferred.
    bool on_mma_completion_arrival(const DAGNode* node) {
        if (!node || node->op_type != OpType::TCGEN05_MMA) {
            return true;
        }
        if (!is_paired_2cta_mma(node)) {
            return true;
        }

        auto it_seq = node_pair_seq.find(node->id);
        if (it_seq == node_pair_seq.end()) {
            // Fallback to non-blocking completion if issue sequence is unavailable.
            return true;
        }
        int seq = it_seq->second;
        PairKey key{node->cta_group_id, seq};
        PairSlot& slot = pair_pending[key];

        int rank = node->cta_group_rank;
        if (rank < 0 || rank > 1) {
            return true;
        }
        slot.node_by_rank[rank] = node->id;

        if (slot.node_by_rank[0] >= 0 && slot.node_by_rank[1] >= 0) {
            release_tokens[slot.node_by_rank[0]]++;
            release_tokens[slot.node_by_rank[1]]++;
            pair_pending.erase(key);
        }
        return consume_release_token(node->id);
    }

    bool consume_release_token(int node_id) {
        auto it = release_tokens.find(node_id);
        if (it == release_tokens.end() || it->second <= 0) {
            return false;
        }
        it->second--;
        if (it->second == 0) {
            release_tokens.erase(it);
        }
        return true;
    }

    static bool is_tmem_op(const DAGNode* node) {
        if (!node) return false;
        return node->op_type == OpType::TCGEN05_MMA ||
               node->op_type == OpType::TCGEN05_ALLOC ||
               node->op_type == OpType::CP_SRAM2TMEM ||
               node->op_type == OpType::TCGEN05_LD ||
               node->op_type == OpType::TCGEN05_ST;
    }

private:
    struct PairKey {
        int cta_group_id;
        int mma_seq;
        bool operator==(const PairKey& other) const {
            return cta_group_id == other.cta_group_id &&
                   mma_seq == other.mma_seq;
        }
    };

    struct PairKeyHash {
        size_t operator()(const PairKey& k) const {
            return (static_cast<size_t>(k.cta_group_id) << 32) ^
                   static_cast<size_t>(k.mma_seq);
        }
    };

    struct PairSlot {
        std::array<int, 2> node_by_rank{{-1, -1}};
    };

    struct PairIssueState {
        int cycle = -1;
        std::array<int, 2> node_by_rank{{-1, -1}};
    };

    static int cta_key(const DAGNode* node) {
        if (!node) return -1;
        if (node->thread_block_id >= 0) {
            return node->thread_block_id;
        }
        return node->warp_id;
    }

    static bool is_paired_2cta_mma(const DAGNode* node) {
        if (!node) return false;
        if (node->op_type != OpType::TCGEN05_MMA) return false;
        return node->cta_group_id >= 0 && node->cta_group_size == 2;
    }

    bool paired_mma_issue_ready(const DAGNode* node, int current_cycle) {
        auto grant_it = node_issue_grant_cycle.find(node->id);
        if (grant_it != node_issue_grant_cycle.end()) {
            return grant_it->second == current_cycle;
        }

        int rank = node->cta_group_rank;
        if (rank < 0 || rank > 1) {
            return false;
        }
        int cta = cta_key(node);
        int seq = cta_issued_mma_seq[cta] + 1;
        PairKey key{node->cta_group_id, seq};
        PairIssueState& state = pair_issue_pending[key];

        if (state.cycle != current_cycle) {
            state.cycle = current_cycle;
            state.node_by_rank = {{-1, -1}};
        }
        state.node_by_rank[rank] = node->id;

        if (state.node_by_rank[0] >= 0 && state.node_by_rank[1] >= 0) {
            // Handshake and grant both peers to issue in the next cycle.
            int grant_cycle = current_cycle + 1;
            node_issue_grant_cycle[state.node_by_rank[0]] = grant_cycle;
            node_issue_grant_cycle[state.node_by_rank[1]] = grant_cycle;
            pair_issue_pending.erase(key);
        }
        return false;
    }

    int max_cols_per_cta;
    int issue_interval_cycles;
    std::vector<int> next_issue_cycle;
    std::unordered_map<int, int> cta_allocated_cols;
    std::unordered_map<int, int> cta_issued_mma_seq;
    std::unordered_map<int, int> cta_pair_issue_seq;
    std::unordered_map<int, int> node_pair_seq;

    inline static std::unordered_map<PairKey, PairSlot, PairKeyHash> pair_pending{};
    inline static std::unordered_map<PairKey, PairIssueState, PairKeyHash> pair_issue_pending{};
    inline static std::unordered_map<int, int> node_issue_grant_cycle{};
    inline static std::unordered_map<int, int> release_tokens{};
};

} // namespace GTSim

#endif // TMEM_H
