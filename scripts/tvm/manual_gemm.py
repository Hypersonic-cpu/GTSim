#!/usr/bin/env python3
"""Independently authored, expanded H100 GEMM IR specification for Phase 2.

This does not import, invoke, or dump the C++ reference builder. The fixed
schedule and every expansion/mapping/edge rule are written out here. The
checked-in .gtsir fixture is its smallest two-stage instance.
"""

import argparse

from tile_ir import Operation, TileIR


def manual_gemm(m, n, k, sms=132):
    if min(m, n, k, sms) <= 0 or m % 128 or n % 256 or k % 64:
        raise ValueError("manual baseline requires M%128=N%256=K%64=0 and positive SM count")
    ir = TileIR()
    ir.tensor(0, "A", (m, k), (k, 1))
    ir.tensor(1, "B", (k, n), (n, 1))
    ir.tensor(2, "C", (m, n), (n, 1))
    columns = n // 256
    order = [(row, column) for start in range(0, columns, 8)
             for row in range(m // 128) for column in range(start, min(columns, start + 8))]
    grid = min(sms, len(order))
    for cta in range(grid):
        # Two 64*(128+256) FP16 input stages, FP16 output staging,
        # FP32 accumulators, and 32 registers/thread for all 384 threads.
        ir.cta(cta, cta, 2 * 2 * 64 * (128 + 256) + 128 * 256 * 2,
               4 * 128 * 256 + 12 * 32 * 32 * 4)
        previous_compute = []
        previous_penultimate = []
        previous_loads = []
        previous_stores = []
        for tile in range(cta, len(order), grid):
            row, column = order[tile]
            row *= 128
            column *= 256
            prefix = f"TB{cta}_Tile{tile}"
            last = [-1] * 8
            loads, compute = [], []

            def emit(suffix, op, warp, tensor, region, deps=(), issue=(), dtype="fp16"):
                return ir.add(Operation(prefix + suffix, op, cta, warp, tensor, region,
                                        dtype, list(deps), list(issue)))

            for step in range(k // 64):
                # Buffer k%2 cannot be overwritten before all consumers finish.
                reuse = (previous_compute if (k // 64) % 2 else previous_penultimate) if step == 0 else []
                if step >= 2:
                    reuse = compute[step - 2]
                issue = previous_loads if step == 0 else loads[-1]
                stage_loads = [emit(f"_TMA_LoadA_k{step}", "cp.dram2sram_tma", 8, 0,
                                    (row, step * 64, 128, 64), reuse, issue)]
                for chunk in range(4):
                    stage_loads.append(emit(f"_TMA_LoadB_k{step}_c{chunk}", "cp.dram2sram_tma", 8, 1,
                                            (step * 64, column + chunk * 64, 64, 64), reuse, issue))
                loads.append(stage_loads)
                stage_compute = []
                for warp in range(8):
                    for half in range(2):
                        for chunk in range(2):
                            nc = (warp // 4) * 2 + chunk
                            for inner_k in range(4):
                                deps = stage_loads + ([last[warp]] if last[warp] >= 0 else [])
                                if step == 0:
                                    deps = deps + previous_stores
                                last[warp] = emit(f"_WGMMA_w{cta * 12 + warp}_k{step}_mh{half}_n{nc}_ks{inner_k}",
                                    "wgmma.m64n64k16", warp, 2,
                                    (row + half * 64 + (warp % 4) * 16, column + nc * 64, 16, 64), deps,
                                    dtype="fp32")
                                stage_compute.append(last[warp])
                compute.append(stage_compute)
            stores = []
            for warp in range(8):
                for half in range(2):
                    for nt in range(8):
                        stores.append(emit(f"_StSram_w{cta * 12 + warp}_mh{half}_n{nt}", "st.reg2sram", warp, 2,
                            (row + half * 64 + (warp % 4) * 16, column + (warp // 4) * 128 + nt * 16, 16, 16),
                            [last[warp]] + compute[-1]))
            output = []
            for chunk in range(4):
                output.append(emit(f"_TMA_StoreC_c{chunk}", "cp.sram2dram_tma", 8, 2,
                    (row, column + chunk * 64, 128, 64), stores, output[-1:]))
            previous_stores = output
            previous_compute = compute[-1]
            previous_penultimate = compute[-2] if len(compute) >= 2 else []
            previous_loads = loads[-1]
    return ir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--sms", type=int, default=132)
    args = parser.parse_args()
    manual_gemm(args.M, args.N, args.K, args.sms).write(args.output)


if __name__ == "__main__":
    main()
