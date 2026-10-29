#ifndef GTSIM_WORKLOAD_OPTIONS_H
#define GTSIM_WORKLOAD_OPTIONS_H
#include "simulator.h"
#include <string>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace GTSim {
struct KernelOptions {
    int stages{2};
    int head_dim{128};
    DataType io_dtype{DataType::FP16};
    int bm{128};
    int bn{256};
    int bk{64};
    int q_tile{128};
    int kv_tile{128};
    int split{1};
    int group_size{8};
    std::string order{"original"};

    bool parse(const std::string& flag, const std::string& value) {
        if (flag == "--dtype") {
            if (value != "fp16" && value != "bf16") throw std::invalid_argument("dtype must be fp16 or bf16");
            io_dtype = value == "bf16" ? DataType::BF16 : DataType::FP16; return true;
        }
        if (flag == "--cta-order") { order = value; return true; }
        int* target = nullptr;
        if (flag == "--head-dim") target = &head_dim;
        else if (flag == "--stages") target = &stages;
        else if (flag == "--bm") target = &bm;
        else if (flag == "--bn") target = &bn;
        else if (flag == "--bk") target = &bk;
        else if (flag == "--q-tile") target = &q_tile;
        else if (flag == "--kv-tile") target = &kv_tile;
        else if (flag == "--split") target = &split;
        else if (flag == "--group-size") target = &group_size;
        if (!target) return false;
        std::size_t used = 0;
        *target = std::stoi(value, &used);
        if (used != value.size()) throw std::invalid_argument("invalid integer for " + flag);
        return true;
    }
    bool gemm_baseline() const {
        return stages == 2 && bm == 128 && bn == 256 && bk == 64 && split == 1 && order == "original";
    }
    bool fa3_baseline() const {
        return stages == 2 && q_tile == 128 && kv_tile == 128 && split == 1 && order == "original";
    }
    CTAResources resources(bool attention) const {
        CTAResources r;
        r.warps = 12;
        if (attention) {
            // Q stays live throughout all QK operations. Output reuses input
            // storage after the last PV; probabilities and O live in registers.
            r.shared_bytes = 2 * q_tile * head_dim + stages * 4 * kv_tile * head_dim;
            r.register_bytes = 4 * q_tile * (head_dim + kv_tile) + 12 * 32 * 32 * 4;
        } else {
            // Persistent GEMM can overlap next-tile inputs with C staging.
            r.shared_bytes = stages * 2 * bk * (bm + bn) + bm * bn * (split > 1 ? 4 : 2);
            r.register_bytes = 4 * bm * bn + 12 * 32 * 32 * 4;
        }
        return r;
    }
    void validate(bool attention, int rows, int columns, int reduction,
                  const SimulatorConfig& hardware) const {
        auto fail = [](const std::string& msg) { throw std::invalid_argument("infeasible kernel: " + msg); };
        if (stages < 2 || stages > 4) fail("stages must be 2, 3, or 4");
        if (split != 1 && split != 2 && split != 4 && split != 8) fail("split must be 1, 2, 4, or 8");
        if (group_size <= 0 || rows <= 0 || columns <= 0 || reduction <= 0) fail("dimensions/group size must be positive");
        if (attention) {
            if (head_dim != 128 && head_dim != 256) fail("FA3 head dimension must be 128 or 256");
            if ((q_tile != 64 && q_tile != 128) ||
                (kv_tile != 64 && kv_tile != 128 && kv_tile != 256)) fail("unsupported FA3 tile");
            if ((rows != 1 && rows % q_tile) || reduction % split || (reduction / split) % 64) fail("FA3 dimensions must divide query tiles and split KV lengths must align to 64 elements");
            if (order != "original" && order != "query-major" && order != "kv-head-grouped") fail("unsupported FA3 CTA order");
        } else {
            // if ((bm != 64 && bm != 128) || (bn != 128 && bn != 256) ||
            //     (bk != 32 && bk != 64 && bk != 128)) fail("unsupported GEMM tile");
            // if (rows % bm || columns % bn || reduction % (split * bk)) fail("GEMM dimensions must divide tiles and split K tiles");
            if (order != "original" && order != "m-major" && order != "n-major" && order != "grouped") fail("unsupported GEMM CTA order");
        }
        auto r = resources(attention);
        if (r.shared_bytes > hardware.shared_memory_per_cta_bytes ||
            r.shared_bytes > hardware.shared_memory_per_sm_bytes) fail("shared-memory footprint " + std::to_string(r.shared_bytes) + " exceeds capacity");
        if (r.register_bytes > hardware.register_file_per_sm_bytes ||
            r.register_bytes > r.warps * 32 * 255 * 4) fail("register footprint exceeds capacity");
        if (r.warps > hardware.max_warps_per_sm) fail("warp footprint exceeds capacity");
        std::cout << "Kernel options: stages=" << stages << " tile="
                  << (attention ? q_tile : bm) << "x" << (attention ? kv_tile : bn)
                  << "x" << (attention ? head_dim : bk) << " split=" << split << " order=" << order
                  << " shared_bytes=" << r.shared_bytes << " register_bytes=" << r.register_bytes
                  << " resident_CTA_limit=" << std::min({hardware.max_concurrent_blocks_per_sm,
                      hardware.shared_memory_per_sm_bytes / r.shared_bytes,
                      hardware.register_file_per_sm_bytes / r.register_bytes,
                      hardware.max_warps_per_sm / r.warps}) << '\n';
    }
};

// Stable merge of independently normalized attention partitions:
// w_s=exp(LSE_s-max LSE), O=sum(w_s O_s)/sum(w_s).
inline std::vector<double> merge_attention(const std::vector<std::vector<double>>& outputs,
                                         const std::vector<double>& lse) {
    double m = *std::max_element(lse.begin(), lse.end());
    std::vector<double> out(outputs.front().size(), 0.0);
    double denom = 0;
    for (std::size_t s = 0; s < lse.size(); ++s) {
        double w = std::exp(lse[s] - m);
        denom += w;
        for (std::size_t d = 0; d < out.size(); ++d) out[d] += w * outputs[s][d];
    }
    for (double& x : out) x /= denom;
    return out;
}
}
#endif
