#include "arch_conf.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace GTSim;

DAG* build_fa3_h100(int seq_q = 128,
                     int seq_kv = 128,
                     int num_heads_q = 1,
                     int num_heads_kv = 1,
                     int batch = 1) {
    DAG* dag = new DAG();
    int node_id = 0;

    const int num_sms = 132;
    const int q_tile = 128;
    const int kv_tile = 128;
    const int head_dim = 128;
    const int num_k = (seq_kv + kv_tile - 1) / kv_tile;

    const int warps_per_tb = 12;
    const int consumer_groups = 2;
    const int warps_per_group = 4;

    auto add_node = [&](const std::string& name, const std::string& pipeline_type,
                        const std::string& op, int warp_id, int r_off, int c_off,
                        int r, int c, int register_change,
                        const std::vector<int>& depends_on = {}) -> DAGNode* {
        DAGNode* node = new DAGNode(node_id++, name, pipeline_type, op, warp_id,
                                    depends_on, register_change, Tile(r_off, c_off, r, c));
        dag->add_node(node);
        return node;
    };

    int tb_id = 0;
    int blocks_q = (seq_q + q_tile - 1) / q_tile;

    int group_size = 1;
    if (num_heads_kv > 0) {
        group_size = num_heads_q / num_heads_kv;
    }

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < num_heads_q; ++h) {
            for (int bx = 0; bx < blocks_q; ++bx) {
                int base_r = bx * q_tile;
                int q_rows = std::min(q_tile, std::max(0, seq_q - base_r));
                if (q_rows <= 0) {
                    continue;
                }

                int warp_base = tb_id * warps_per_tb;
                int loader_warp = warp_base + 8;
                int kv_head = (group_size > 0) ? (h / group_size) : h;
                int matrix_base_q = (b * num_heads_q + h) * 4;
                int matrix_base_kv = (b * num_heads_kv + kv_head) * 4;
                int sm_id = (num_sms > 0) ? (tb_id % num_sms) : 0;

                int q_rows_0 = std::min(64, q_rows);
                int q_rows_1 = std::min(64, std::max(0, q_rows - 64));

                auto q0 = add_node("TB" + std::to_string(tb_id) + "_Q_Load0_b" +
                                       std::to_string(b) + "_h" + std::to_string(h),
                                   "LD", "cp.dram2sram_tma", loader_warp,
                                   base_r, 0, q_rows_0, 128, 0);
                q0->sm_id = sm_id;
                q0->thread_block_id = tb_id;
                q0->matrix_id = matrix_base_q + 0;
                q0->layout = Layout::RowMajor;
                q0->matrix_leading_dim = head_dim;

                auto q1 = add_node("TB" + std::to_string(tb_id) + "_Q_Load1_b" +
                                       std::to_string(b) + "_h" + std::to_string(h),
                                   "LD", "cp.dram2sram_tma", loader_warp,
                                   base_r + 64, 0, q_rows_1, 128, 0);
                q1->sm_id = sm_id;
                q1->thread_block_id = tb_id;
                q1->matrix_id = matrix_base_q + 0;
                q1->layout = Layout::RowMajor;
                q1->matrix_leading_dim = head_dim;

                std::vector<std::vector<int>> wgmma_ss_per_k(num_k);
                std::vector<std::vector<int>> wgmma_rs_per_k(num_k);
                std::vector<std::vector<int>> last_wgmma_ss_per_warp(consumer_groups,
                                                                      std::vector<int>(warps_per_group, -1));
                std::vector<std::vector<int>> last_wgmma_rs_per_warp(consumer_groups,
                                                                      std::vector<int>(warps_per_group, -1));
                std::vector<std::vector<int>> last_logsum_per_warp(consumer_groups,
                                                                    std::vector<int>(warps_per_group, -1));
                std::vector<std::vector<std::vector<int>>> acc_o_scale_ids(
                    num_k, std::vector<std::vector<int>>(consumer_groups, std::vector<int>(warps_per_group, -1)));
                std::vector<std::vector<std::vector<int>>> wgmma_rs_tail_ids(
                    num_k, std::vector<std::vector<int>>(consumer_groups, std::vector<int>(warps_per_group, -1)));
                std::vector<std::vector<std::vector<int>>> score_logsum_ids(
                    num_k, std::vector<std::vector<int>>(consumer_groups, std::vector<int>(warps_per_group, -1)));
                std::vector<std::vector<std::vector<int>>> score_cast_ids(
                    num_k, std::vector<std::vector<int>>(consumer_groups, std::vector<int>(warps_per_group, -1)));

                for (int k = 0; k < num_k; ++k) {
                    int kv_cols = std::min(kv_tile, std::max(0, seq_kv - k * kv_tile));
                    if (kv_cols <= 0) {
                        break;
                    }

                    std::vector<int> k_deps;
                    if (k >= 2) {
                        // K uses double-buffering: once SS(k-2) consumes K(k-2), K(k) can be loaded.
                        k_deps.insert(k_deps.end(), wgmma_ss_per_k[k - 2].begin(),
                                      wgmma_ss_per_k[k - 2].end());
                    }

                    auto k0 = add_node("TB" + std::to_string(tb_id) + "_K_Load0_k" + std::to_string(k),
                                       "LD", "cp.dram2sram_tma", loader_warp,
                                       0, k * kv_tile, 64, kv_cols, 0, k_deps);
                    k0->sm_id = sm_id;
                    k0->thread_block_id = tb_id;
                    k0->matrix_id = matrix_base_kv + 1;
                    k0->layout = Layout::RowMajor;
                    k0->matrix_leading_dim = seq_kv;

                    auto k1 = add_node("TB" + std::to_string(tb_id) + "_K_Load1_k" + std::to_string(k),
                                       "LD", "cp.dram2sram_tma", loader_warp,
                                       64, k * kv_tile, 64, kv_cols, 0, k_deps);
                    k1->sm_id = sm_id;
                    k1->thread_block_id = tb_id;
                    k1->matrix_id = matrix_base_kv + 1;
                    k1->layout = Layout::RowMajor;
                    k1->matrix_leading_dim = seq_kv;

                    std::vector<int> v_deps;
                    if (k >= 2) {
                        // V uses double-buffering: once RS(k-2) consumes V(k-2), V(k) can be loaded.
                        v_deps.insert(v_deps.end(), wgmma_rs_per_k[k - 2].begin(),
                                      wgmma_rs_per_k[k - 2].end());
                    }

                    auto v0 = add_node("TB" + std::to_string(tb_id) + "_V_Load0_k" + std::to_string(k),
                                       "LD", "cp.dram2sram_tma", loader_warp,
                                       0, k * kv_tile, 64, kv_cols, 0, v_deps);
                    v0->sm_id = sm_id;
                    v0->thread_block_id = tb_id;
                    v0->matrix_id = matrix_base_kv + 2;
                    v0->layout = Layout::RowMajor;
                    v0->matrix_leading_dim = seq_kv;

                    auto v1 = add_node("TB" + std::to_string(tb_id) + "_V_Load1_k" + std::to_string(k),
                                       "LD", "cp.dram2sram_tma", loader_warp,
                                       64, k * kv_tile, 64, kv_cols, 0, v_deps);
                    v1->sm_id = sm_id;
                    v1->thread_block_id = tb_id;
                    v1->matrix_id = matrix_base_kv + 2;
                    v1->layout = Layout::RowMajor;
                    v1->matrix_leading_dim = seq_kv;

                    for (int g = 0; g < consumer_groups; ++g) {
                        int group_row_base = g * 64;
                        for (int warp_lane = 0; warp_lane < warps_per_group; ++warp_lane) {
                            int warp_id = warp_base + g * warps_per_group + warp_lane;

                            for (int ki = 0; ki < 16; ++ki) {
                            std::vector<int> deps = {q0->id, q1->id, k0->id, k1->id};
                            if (last_wgmma_ss_per_warp[g][warp_lane] != -1) {
                                deps.push_back(last_wgmma_ss_per_warp[g][warp_lane]);
                            }
                            auto wgmma_ss = add_node("TB" + std::to_string(tb_id) + "_WGMMA_SS_g" +
                                                         std::to_string(g) + "_w" +
                                                         std::to_string(warp_id) + "_k" + std::to_string(k) +
                                                         "_s" + std::to_string(ki),
                                                     "Tensor", "wgmma.m64n64k16", warp_id,
                                                     group_row_base, 0, 16, 64, 0, deps);
                                if (ki == 0 && k > 0) {
                                    // Cross-round ordering: round-k first op must issue after round-(k-1) score tail.
                                    if (score_logsum_ids[k - 1][g][warp_lane] != -1) {
                                        dag->add_issue_dependency(score_logsum_ids[k - 1][g][warp_lane],
                                                                  wgmma_ss->id);
                                    }
                                    if (score_cast_ids[k - 1][g][warp_lane] != -1) {
                                        dag->add_issue_dependency(score_cast_ids[k - 1][g][warp_lane],
                                                                  wgmma_ss->id);
                                    }
                                }
                                wgmma_ss->sm_id = sm_id;
                                wgmma_ss->thread_block_id = tb_id;
                                wgmma_ss->matrix_id = 3;
                                wgmma_ss->data_type = DataType::FP32;
                                wgmma_ss_per_k[k].push_back(wgmma_ss->id);
                                last_wgmma_ss_per_warp[g][warp_lane] = wgmma_ss->id;
                            }

                            if (k > 0 && acc_o_scale_ids[k - 1][g][warp_lane] != -1) {
                                // In main loop enforce stage order: SS(k) issues before acc_o_scale(k-1).
                                dag->add_issue_dependency(last_wgmma_ss_per_warp[g][warp_lane],
                                                          acc_o_scale_ids[k - 1][g][warp_lane]);
                            }

                            int last_ss = last_wgmma_ss_per_warp[g][warp_lane];

                            auto local_max_r0 = add_node("TB" + std::to_string(tb_id) + "_Softmax_LocalMax_r0_g" +
                                                             std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                             "_k" + std::to_string(k),
                                                         "SIMD", "local_max", warp_id,
                                                         group_row_base, 0, 8, 128, 0, {last_ss});
                            local_max_r0->sm_id = sm_id;
                            local_max_r0->thread_block_id = tb_id;
                            if (k > 0 && wgmma_rs_tail_ids[k - 1][g][warp_lane] != -1) {
                                // In main loop enforce stage order: RS(k-1) issues before score(k).
                                dag->add_issue_dependency(wgmma_rs_tail_ids[k - 1][g][warp_lane],
                                                          local_max_r0->id);
                            }

                            auto shfl_max_r0 = add_node("TB" + std::to_string(tb_id) + "_Softmax_ShflMax_r0_g" +
                                                            std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                            "_k" + std::to_string(k),
                                                        "SHFL", "shfl_max", warp_id,
                                                        group_row_base, 0, 8, 4, 0, {local_max_r0->id});
                            shfl_max_r0->sm_id = sm_id;
                            shfl_max_r0->thread_block_id = tb_id;

                            auto reduce_max_r0 = add_node("TB" + std::to_string(tb_id) + "_Softmax_ReduceMax_r0_g" +
                                                              std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                              "_k" + std::to_string(k),
                                                          "SIMD", "reduce_max", warp_id,
                                                          group_row_base, 0, 8, 4, 0, {shfl_max_r0->id});
                            reduce_max_r0->sm_id = sm_id;
                            reduce_max_r0->thread_block_id = tb_id;

                            auto local_max_r1 = add_node("TB" + std::to_string(tb_id) + "_Softmax_LocalMax_r1_g" +
                                                             std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                             "_k" + std::to_string(k),
                                                         "SIMD", "local_max", warp_id,
                                                         group_row_base, 0, 8, 128, 0, {last_ss});
                            local_max_r1->sm_id = sm_id;
                            local_max_r1->thread_block_id = tb_id;
                            if (k > 0 && wgmma_rs_tail_ids[k - 1][g][warp_lane] != -1) {
                                // Same RS(k-1) -> score(k) ordering on the second score branch.
                                dag->add_issue_dependency(wgmma_rs_tail_ids[k - 1][g][warp_lane],
                                                          local_max_r1->id);
                            }

                            auto shfl_max_r1 = add_node("TB" + std::to_string(tb_id) + "_Softmax_ShflMax_r1_g" +
                                                            std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                            "_k" + std::to_string(k),
                                                        "SHFL", "shfl_max", warp_id,
                                                        group_row_base, 0, 8, 4, 0, {local_max_r1->id});
                            shfl_max_r1->sm_id = sm_id;
                            shfl_max_r1->thread_block_id = tb_id;

                            auto reduce_max_r1 = add_node("TB" + std::to_string(tb_id) + "_Softmax_ReduceMax_r1_g" +
                                                              std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                              "_k" + std::to_string(k),
                                                          "SIMD", "reduce_max", warp_id,
                                                          group_row_base, 0, 8, 4, 0, {shfl_max_r1->id});
                            reduce_max_r1->sm_id = sm_id;
                            reduce_max_r1->thread_block_id = tb_id;

                            auto merge_max = add_node("TB" + std::to_string(tb_id) + "_Softmax_MergeMax_g" +
                                                          std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                          "_k" + std::to_string(k),
                                                      "SIMD", "merge_max", warp_id,
                                                      group_row_base, 0, 8, 4, 0, {reduce_max_r0->id, reduce_max_r1->id});
                            merge_max->sm_id = sm_id;
                            merge_max->thread_block_id = tb_id;

                            auto scores_scale = add_node("TB" + std::to_string(tb_id) + "_Softmax_Scale_g" +
                                                             std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                             "_k" + std::to_string(k),
                                                         "SFU", "scores_scale", warp_id,
                                                         group_row_base, 0, 8, 16, 0, {merge_max->id});
                            scores_scale->sm_id = sm_id;
                            scores_scale->thread_block_id = tb_id;

                            auto exp2_r0 = add_node("TB" + std::to_string(tb_id) + "_Softmax_Exp2_r0_g" +
                                                        std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                        "_k" + std::to_string(k),
                                                    "SFU", "exp2", warp_id,
                                                    group_row_base, 0, 16, 64, 0, {scores_scale->id});
                            exp2_r0->sm_id = sm_id;
                            exp2_r0->thread_block_id = tb_id;

                            auto exp2_r1 = add_node("TB" + std::to_string(tb_id) + "_Softmax_Exp2_r1_g" +
                                                        std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                        "_k" + std::to_string(k),
                                                    "SFU", "exp2", warp_id,
                                                    group_row_base, 0, 16, 64, 0, {scores_scale->id});
                            exp2_r1->sm_id = sm_id;
                            exp2_r1->thread_block_id = tb_id;

                            auto local_sum_r0 = add_node("TB" + std::to_string(tb_id) + "_Softmax_LocalSum_r0_g" +
                                                             std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                             "_k" + std::to_string(k),
                                                         "SIMD", "local_sum", warp_id,
                                                         group_row_base, 0, 8, 128, 0, {exp2_r0->id});
                            local_sum_r0->sm_id = sm_id;
                            local_sum_r0->thread_block_id = tb_id;

                            auto shfl_sum_r0 = add_node("TB" + std::to_string(tb_id) + "_Softmax_ShflSum_r0_g" +
                                                            std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                            "_k" + std::to_string(k),
                                                        "SHFL", "shfl_sum", warp_id,
                                                        group_row_base, 0, 8, 4, 0, {local_sum_r0->id});
                            shfl_sum_r0->sm_id = sm_id;
                            shfl_sum_r0->thread_block_id = tb_id;

                            auto reduce_sum_r0 = add_node("TB" + std::to_string(tb_id) + "_Softmax_ReduceSum_r0_g" +
                                                              std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                              "_k" + std::to_string(k),
                                                          "SIMD", "reduce_sum", warp_id,
                                                          group_row_base, 0, 8, 4, 0, {shfl_sum_r0->id});
                            reduce_sum_r0->sm_id = sm_id;
                            reduce_sum_r0->thread_block_id = tb_id;

                            auto local_sum_r1 = add_node("TB" + std::to_string(tb_id) + "_Softmax_LocalSum_r1_g" +
                                                             std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                             "_k" + std::to_string(k),
                                                         "SIMD", "local_sum", warp_id,
                                                         group_row_base, 0, 8, 128, 0, {exp2_r1->id});
                            local_sum_r1->sm_id = sm_id;
                            local_sum_r1->thread_block_id = tb_id;

                            auto shfl_sum_r1 = add_node("TB" + std::to_string(tb_id) + "_Softmax_ShflSum_r1_g" +
                                                            std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                            "_k" + std::to_string(k),
                                                        "SHFL", "shfl_sum", warp_id,
                                                        group_row_base, 0, 8, 4, 0, {local_sum_r1->id});
                            shfl_sum_r1->sm_id = sm_id;
                            shfl_sum_r1->thread_block_id = tb_id;

                            auto reduce_sum_r1 = add_node("TB" + std::to_string(tb_id) + "_Softmax_ReduceSum_r1_g" +
                                                              std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                              "_k" + std::to_string(k),
                                                          "SIMD", "reduce_sum", warp_id,
                                                          group_row_base, 0, 8, 4, 0, {shfl_sum_r1->id});
                            reduce_sum_r1->sm_id = sm_id;
                            reduce_sum_r1->thread_block_id = tb_id;

                            auto merge_sum = add_node("TB" + std::to_string(tb_id) + "_Softmax_MergeSum_g" +
                                                          std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                          "_k" + std::to_string(k),
                                                      "SIMD", "merge_sum", warp_id,
                                                      group_row_base, 0, 8, 4, 0, {reduce_sum_r0->id, reduce_sum_r1->id});
                            merge_sum->sm_id = sm_id;
                            merge_sum->thread_block_id = tb_id;

                            auto logsum = add_node("TB" + std::to_string(tb_id) + "_Softmax_Logsum_g" +
                                                       std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                       "_k" + std::to_string(k),
                                                   "SIMD", "logsum", warp_id,
                                                   group_row_base, 0, 8, 8, 0, {merge_sum->id, scores_scale->id});
                            logsum->sm_id = sm_id;
                            logsum->thread_block_id = tb_id;
                            last_logsum_per_warp[g][warp_lane] = logsum->id;
                            score_logsum_ids[k][g][warp_lane] = logsum->id;

                            auto acc_o_scale = add_node("TB" + std::to_string(tb_id) + "_Softmax_AccOScale_g" +
                                                           std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                           "_k" + std::to_string(k),
                                                       "SIMD", "acc_o_scale", warp_id,
                                                       group_row_base, 0, 32, 64, 0, {scores_scale->id});
                            acc_o_scale->sm_id = sm_id;
                            acc_o_scale->thread_block_id = tb_id;
                            acc_o_scale_ids[k][g][warp_lane] = acc_o_scale->id;

                            auto acc_s_cast = add_node("TB" + std::to_string(tb_id) + "_Softmax_Cast_g" +
                                                           std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                           "_k" + std::to_string(k),
                                                       "SIMD", "acc_s_cast", warp_id,
                                                       group_row_base, 0, 8, 64, 0, {exp2_r0->id, exp2_r1->id});
                            acc_s_cast->sm_id = sm_id;
                            acc_s_cast->thread_block_id = tb_id;
                            score_cast_ids[k][g][warp_lane] = acc_s_cast->id;

                            for (int ki = 0; ki < 16; ++ki) {
                                std::vector<int> deps = {v0->id, v1->id, acc_s_cast->id, acc_o_scale->id, logsum->id};
                                if (last_wgmma_rs_per_warp[g][warp_lane] != -1) {
                                    deps.push_back(last_wgmma_rs_per_warp[g][warp_lane]);
                                }
                                auto wgmma_rs = add_node("TB" + std::to_string(tb_id) + "_WGMMA_RS_g" +
                                                             std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                             "_k" + std::to_string(k) + "_s" + std::to_string(ki),
                                                         "Tensor", "wgmma.m64n64k16", warp_id,
                                                         group_row_base, 0, 16, 64, 0, deps);
                                wgmma_rs->sm_id = sm_id;
                                wgmma_rs->thread_block_id = tb_id;
                                wgmma_rs->matrix_id = 4;
                                wgmma_rs->data_type = DataType::FP32;
                                wgmma_rs_per_k[k].push_back(wgmma_rs->id);
                                last_wgmma_rs_per_warp[g][warp_lane] = wgmma_rs->id;
                            }
                            wgmma_rs_tail_ids[k][g][warp_lane] = last_wgmma_rs_per_warp[g][warp_lane];
                        }
                    }
                }

                std::vector<std::vector<int>> acc_o_div_ids(consumer_groups);
                std::vector<int> last_round_wgmma_rs_ids;
                if (!wgmma_rs_per_k.empty()) {
                    last_round_wgmma_rs_ids = wgmma_rs_per_k.back();
                }
                for (int g = 0; g < consumer_groups; ++g) {
                    acc_o_div_ids[g].reserve(warps_per_group);
                    for (int warp_lane = 0; warp_lane < warps_per_group; ++warp_lane) {
                        int warp_id = warp_base + g * warps_per_group + warp_lane;
                        int last_acc_o = last_wgmma_rs_per_warp[g][warp_lane];
                        int last_logsum = last_logsum_per_warp[g][warp_lane];
                        if (last_acc_o == -1 || last_logsum == -1) {
                            continue;
                        }
                        std::vector<int> div_deps = {last_acc_o, last_logsum};
                        // Global barrier: all divs start only after every RS in the final round is done.
                        div_deps.insert(div_deps.end(),
                                        last_round_wgmma_rs_ids.begin(),
                                        last_round_wgmma_rs_ids.end());
                        auto div_node = add_node("TB" + std::to_string(tb_id) + "_AccO_Div_g" +
                                                 std::to_string(g) + "_w" + std::to_string(warp_id),
                                                 "SFU", "acc_o_div", warp_id,
                                                 g * 64, 0, 32, 64, 0, div_deps);
                        div_node->sm_id = sm_id;
                        div_node->thread_block_id = tb_id;
                        acc_o_div_ids[g].push_back(div_node->id);
                    }
                }

                std::vector<int> acc_o_div_ids_all;
                for (int g = 0; g < consumer_groups; ++g) {
                    acc_o_div_ids_all.insert(acc_o_div_ids_all.end(),
                                             acc_o_div_ids[g].begin(),
                                             acc_o_div_ids[g].end());
                }

                std::vector<std::vector<int>> store_ids_by_group(consumer_groups);
                const int rows_per_warp = 16;
                const int col_blocks = head_dim / 16;
                for (int g = 0; g < consumer_groups; ++g) {
                    store_ids_by_group[g].reserve(32);
                    for (int warp_lane = 0; warp_lane < warps_per_group; ++warp_lane) {
                        int warp_id = warp_base + g * warps_per_group + warp_lane;
                        std::vector<int> deps = acc_o_div_ids_all;
                        if (deps.empty() && !wgmma_rs_per_k.empty()) {
                            deps = wgmma_rs_per_k.back();
                        }
                        for (int c_blk = 0; c_blk < col_blocks; ++c_blk) {
                            int tile_r = base_r + g * 64 + warp_lane * rows_per_warp;
                            int tile_c = c_blk * 16;
                            auto store = add_node("TB" + std::to_string(tb_id) + "_StSram_g" +
                                                  std::to_string(g) + "_w" + std::to_string(warp_id) +
                                                  "_c" + std::to_string(c_blk),
                                                  "ST", "st.reg2sram", warp_id,
                                                  tile_r, tile_c, 16, 16, 0, deps);
                            store->sm_id = sm_id;
                            store->thread_block_id = tb_id;
                            store->matrix_id = 5;
                            store->data_type = DataType::FP16;
                            store->layout = Layout::RowMajor;
                            store->matrix_leading_dim = head_dim;
                            store_ids_by_group[g].push_back(store->id);
                        }
                    }
                }

                std::vector<int> store_ids_all;
                for (int g = 0; g < consumer_groups; ++g) {
                    store_ids_all.insert(store_ids_all.end(),
                                         store_ids_by_group[g].begin(),
                                         store_ids_by_group[g].end());
                }

                int out_rows_0 = std::min(64, q_rows);
                int out_rows_1 = std::min(64, std::max(0, q_rows - 64));
                if (out_rows_0 > 0) {
                    auto out0 = add_node("TB" + std::to_string(tb_id) + "_Out_Store_g0", "ST",
                                         "cp.sram2dram_tma", loader_warp,
                                         base_r, 0, out_rows_0, 128, 0, store_ids_all);
                    out0->sm_id = sm_id;
                    out0->thread_block_id = tb_id;
                    out0->matrix_id = matrix_base_q + 3;
                    out0->layout = Layout::RowMajor;
                    out0->matrix_leading_dim = head_dim;
                }

                if (out_rows_1 > 0) {
                    auto out1 = add_node("TB" + std::to_string(tb_id) + "_Out_Store_g1", "ST",
                                         "cp.sram2dram_tma", loader_warp,
                                         base_r + 64, 0, out_rows_1, 128, 0, store_ids_all);
                    out1->sm_id = sm_id;
                    out1->thread_block_id = tb_id;
                    out1->matrix_id = matrix_base_q + 3;
                    out1->layout = Layout::RowMajor;
                    out1->matrix_leading_dim = head_dim;
                }

                ++tb_id;
            }
        }
    }

    return dag;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: gtsim_fa3 <architecture.yaml> [batch heads_q heads_kv sequence_q sequence_kv]\n";
        return 1;
    }

    SimulatorConfig config;
    try {
        config = make_h100_sxm_variants(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << "Configuration error: " << error.what() << '\n';
        return 1;
    }

    int seq_q = 128;
    int seq_kv = 128;
    int num_heads_q = 1;
    int num_heads_kv = 1;
    int batch = 1;
    if (argc > 2) batch = std::atoi(argv[2]);
    if (argc > 3) num_heads_q = std::atoi(argv[3]);
    if (argc > 4) num_heads_kv = std::atoi(argv[4]);
    if (argc > 5) seq_q = std::atoi(argv[5]);
    if (argc > 6) seq_kv = std::atoi(argv[6]);

    if (num_heads_kv <= 0 || num_heads_q <= 0 || (num_heads_q % num_heads_kv) != 0) {
        std::cerr << "Invalid head configuration: head_q must be > 0, head_kv must be > 0, and head_q % head_kv == 0.\n";
        return 1;
    }

    std::cout << "=== H100 FA3 ===\n\n";
    DAG* dag = build_fa3_h100(seq_q, seq_kv, num_heads_q, num_heads_kv, batch);
    std::cout << "Generated DAG with " << dag->nodes.size() << " nodes\n\n";

    config.silence_mode = true;
    config.workload_type = WorkloadType::FA3;

    Simulator simulator(dag, config);
    simulator.run(100000000, true);

    delete dag;
    return 0;
}
