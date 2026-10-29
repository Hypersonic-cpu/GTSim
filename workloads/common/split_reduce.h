#ifndef GTSIM_SPLIT_REDUCE_H
#define GTSIM_SPLIT_REDUCE_H
#include "dag_node.h"
#include <algorithm>
namespace GTSim {
// A separate reduction CTA reads HBM scratch after all partition stores.
// Direct register loads keep the merge bounded to one 16x16 fragment per warp.
inline void append_split_reduce(DAG* dag, const std::vector<int>& partials,
                                const std::vector<int>& lse_tensors,
                                const std::vector<int>& producers, int output,
                                int row, int col, int rows, int cols, int leading,
                                int tb, int sm, DataType output_dtype = DataType::FP16) {
    dag->cta_resources[tb] = {0, 72 * 1024, 12};
    auto add = [&](const std::string& pipe, const std::string& op, int warp,
                   Tile tile, const std::vector<int>& deps, int matrix,
                   DataType dtype = DataType::FP32) {
        int id = static_cast<int>(dag->nodes.size());
        auto* n = new DAGNode(id, "Merge" + std::to_string(tb) + "_" + std::to_string(id),
                              pipe, op, tb * 12 + warp, deps, 0, tile, dtype);
        n->sm_id = sm;
        n->thread_block_id = tb;
        n->matrix_id = matrix;
        n->matrix_leading_dim = leading;
        dag->add_node(n);
        return id;
    };
    std::vector<int> tail(8, -1);
    for (int r = 0; r < rows; r += 16) {
        int warp = (r / 16) % 8;
        const int fragment_rows = std::min(16, rows - r);
        std::vector<int> start = producers;

        std::vector<int> normalization;
        if (!lse_tensors.empty()) {
            std::vector<int> loads;
            for (int tensor : lse_tensors) {
                int id = add("LD", "ld.dram2reg", warp, Tile(row+r, 0, fragment_rows, 1), start, tensor);
                dag->nodes[id]->matrix_leading_dim = 1;
                if (tail[warp] >= 0) dag->add_issue_dependency(tail[warp], id);
                loads.push_back(id);
            }
            int max = add("SIMD", "split_lse_max", warp, Tile(0,0,fragment_rows,static_cast<int>(loads.size())), loads, output);
            int exp = add("SFU", "split_lse_exp", warp, Tile(0,0,fragment_rows,static_cast<int>(loads.size())), {max}, output);
            int sum = add("SIMD", "split_weight_sum", warp, Tile(0,0,fragment_rows,static_cast<int>(loads.size())), {exp}, output);
            normalization = {exp, sum};
        }
        for (int c = 0; c < cols; c += 16) {
            std::vector<int> deps = start;
            deps.insert(deps.end(), normalization.begin(), normalization.end());

            int accum = -1;
            for (int tensor : partials) {
                auto ld_deps = deps;
                if (accum >= 0) ld_deps.push_back(accum);
                int ld = add("LD", "ld.dram2reg", warp, Tile(row+r,col+c,fragment_rows,16), ld_deps, tensor);
                // A store's queued payload no longer needs this register
                // fragment; its memory acknowledgement still retires the CTA.
                if (accum < 0 && tail[warp] >= 0) dag->add_issue_dependency(tail[warp], ld);
                int value = ld;
                if (!lse_tensors.empty())
                    value = add("SIMD", "split_weighted_output", warp, Tile(0,0,fragment_rows,16), {ld, normalization[0]}, output);
                if (accum < 0) accum = value;
                else accum = add("SIMD", "split_output_sum", warp, Tile(0,0,fragment_rows,16), {accum,value}, output);
            }
            if (!lse_tensors.empty())
                accum = add("SFU", "split_normalize", warp, Tile(0,0,fragment_rows,16), {accum,normalization[1]}, output);
            int cast = add("SIMD", "split_cast_fp16", warp, Tile(0,0,fragment_rows,16), {accum}, output);
            tail[warp] = add("ST", "st.reg2dram", warp, Tile(row+r,col+c,fragment_rows,16), {cast}, output, output_dtype);
        }
    }
}
}
#endif
