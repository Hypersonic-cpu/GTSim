#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

namespace {
float value(int seed) {
    return static_cast<float>((seed * 17) % 29 - 14) / 8.0F;
}

void gemm(const std::vector<float>& a, const std::vector<float>& b,
          std::vector<float>& c, int m, int n, int k, int b_cols,
          int b_col_offset = 0) {
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < n; ++col) {
            float sum = 0.0F;
            for (int inner = 0; inner < k; ++inner)
                sum += a[row * k + inner] *
                       b[inner * b_cols + b_col_offset + col];
            c[row * n + col] = sum;
        }
}

float silu(float x) { return x / (1.0F + std::exp(-x)); }
} // namespace

int main() {
    constexpr int m = 3, k = 5, n = 11, slab = 4;
    std::vector<float> a(m * k), b(k * n);
    for (int i = 0; i < static_cast<int>(a.size()); ++i) a[i] = value(i + 1);
    for (int i = 0; i < static_cast<int>(b.size()); ++i) b[i] = value(i + 7);
    std::vector<float> full(m * n), partitioned(m * n);
    gemm(a, b, full, m, n, k, n);
    for (int col = 0; col < n; col += slab) {
        const int width = std::min(slab, n - col);
        std::vector<float> part(m * width);
        gemm(a, b, part, m, width, k, n, col);
        for (int row = 0; row < m; ++row)
            for (int local = 0; local < width; ++local)
                partitioned[row * n + col + local] = part[row * width + local];
    }
    for (std::size_t i = 0; i < full.size(); ++i)
        if (std::fabs(full[i] - partitioned[i]) > 1.0e-6F) {
            std::cerr << "partition mismatch at " << i << '\n';
            return 1;
        }

    // Two selected experts exercise the row-offset contract used by the
    // fused GateUp/Down schedule. Each expert owns a separate K-row view.
    constexpr int experts = 2, expert_k = 5, expert_n = 7;
    std::vector<float> expert_a(m * expert_k), experts_b(experts * expert_k * expert_n);
    for (int i = 0; i < static_cast<int>(expert_a.size()); ++i)
        expert_a[i] = value(i + 101);
    for (int i = 0; i < static_cast<int>(experts_b.size()); ++i)
        experts_b[i] = value(i + 211);
    for (int expert = 0; expert < experts; ++expert) {
        std::vector<float> reference(m * expert_n), sliced(m * expert_n);
        const auto* weights = experts_b.data() + expert * expert_k * expert_n;
        gemm(expert_a, std::vector<float>(weights, weights + expert_k * expert_n),
             reference, m, expert_n, expert_k, expert_n);
        for (int col = 0; col < expert_n; col += slab) {
            const int width = std::min(slab, expert_n - col);
            std::vector<float> part(m * width);
            gemm(expert_a,
                 std::vector<float>(weights, weights + expert_k * expert_n),
                 part, m, width, expert_k, expert_n, col);
            for (int row = 0; row < m; ++row)
                for (int local = 0; local < width; ++local)
                    sliced[row * expert_n + col + local] =
                        part[row * width + local];
        }
        for (std::size_t i = 0; i < reference.size(); ++i)
            if (std::fabs(reference[i] - sliced[i]) > 1.0e-6F) return 1;
    }

    constexpr int intermediate = 4, output = 3;
    std::vector<float> gate_up_weights(expert_k * 2 * intermediate);
    std::vector<float> down_weights(intermediate * output);
    for (int i = 0; i < static_cast<int>(gate_up_weights.size()); ++i)
        gate_up_weights[i] = value(i + 307);
    for (int i = 0; i < static_cast<int>(down_weights.size()); ++i)
        down_weights[i] = value(i + 401);
    std::vector<float> gate_up(m * 2 * intermediate);
    gemm(expert_a, gate_up_weights, gate_up, m, 2 * intermediate, expert_k,
         2 * intermediate);
    std::vector<float> gate_up_slabbed(gate_up.size());
    for (int col = 0; col < 2 * intermediate; col += slab) {
        const int width = std::min(slab, 2 * intermediate - col);
        std::vector<float> part(m * width);
        gemm(expert_a, gate_up_weights, part, m, width, expert_k,
             2 * intermediate, col);
        for (int row = 0; row < m; ++row)
            for (int local = 0; local < width; ++local)
                gate_up_slabbed[row * 2 * intermediate + col + local] =
                    part[row * width + local];
    }
    std::vector<float> activation(m * intermediate);
    std::vector<float> activation_slabbed(m * intermediate);
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < intermediate; ++col) {
            const auto gate = gate_up[row * 2 * intermediate + col];
            const auto up = gate_up[row * 2 * intermediate + intermediate + col];
            const auto gate_slabbed =
                gate_up_slabbed[row * 2 * intermediate + col];
            const auto up_slabbed =
                gate_up_slabbed[row * 2 * intermediate + intermediate + col];
            activation[row * intermediate + col] = silu(gate) * up;
            activation_slabbed[row * intermediate + col] =
                silu(gate_slabbed) * up_slabbed;
        }
    std::vector<float> down(m * output), down_slabbed(m * output);
    gemm(activation, down_weights, down, m, output, intermediate, output);
    for (int col = 0; col < output; col += slab) {
        const int width = std::min(slab, output - col);
        std::vector<float> part(m * width);
        gemm(activation_slabbed, down_weights, part, m, width, intermediate,
             output, col);
        for (int row = 0; row < m; ++row)
            for (int local = 0; local < width; ++local)
                down_slabbed[row * output + col + local] =
                    part[row * width + local];
    }
    for (std::size_t i = 0; i < down.size(); ++i)
        if (std::fabs(down[i] - down_slabbed[i]) > 1.0e-6F) return 1;

    std::cout << "FlashAccel N-slab partition oracle passed\n";
}
