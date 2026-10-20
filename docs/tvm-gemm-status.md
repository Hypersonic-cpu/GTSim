# TVM GEMM integration: execution record

Current branch extension: see [HBM/HBF placement evaluation](tvm-hbf-placement.md).
The execution record below describes the earlier FullHBM work.

Implementation baseline: branch `tvm`, GTSim `ba50ffc`, pinned TVM
`4d0ac168c78c1e20bf05a6ec5926760bb1c52b05`. No rebase, vendor changes, simulator
timing/scheduling/memory changes, calibration changes, or reference updates.
The implementation is an uncommitted extension; the generated validation JSON
records source hashes as well as HEAD. See [the contract](tile-ir.md) for setup
and scope.

## Phase 1: interface proved

```sh
python3 scripts/tvm/build_tvm.py --jobs 6
cmake --build build --parallel 4
.tvm-venv/bin/python scripts/tvm/smoke.py
```

The pinned compiler built independently on macOS without LLVM or CUDA runtime
support. The actual loaded library hashes match its build manifest. A TVM TIR
FP16 row copy passed through `.gtsir`, `TensorRegistry` / `DAGNode` materialization
and the existing Simulator: **2/2 nodes finished, 1,370 cycles, zero HBF
requests**. Artifacts are in `validation/results/tvm/smoke/`.

The symbolic Hopper probe constructed real TVM operations
`tirx.ptx.cp_async_bulk_tensor_g2s_cluster`, `tirx.ptx.wgmma_f16_ss` and
`tirx.cuda.mbarrier_wait`, plus WGMMA fence/commit/wait. Executed
`StmtSimplify`, `FlattenBuffer`, and `LowerIntrin` snapshots retain thread
bindings, explicit PTX and barrier waits at this pin. The copy snapshots show
that `FlattenBuffer` replaces 2-D indexed regions with flattened views. The
chosen extraction boundary precedes that pass. This is an executable interface
proof and representation inspection, **not automatic GEMM translation**.

## Phase 2: exact parity; accuracy gate failed

```sh
python3 scripts/tvm/validate_gemm_ir.py --build-dir build --jobs 4
```

Both runners used `configs/v0.1.0/h100-sxm.yaml` and
`configs/memconf/full-hbm.yaml`, with BM=128, BN=256, BK=64, two stages,
12 warps, split=1 and original CTA order. The manual IR was authored
independently of the reference builder. All three frontend tensors are HBM.

| M | N | K | Reference cycles | IR cycles | Measured cycles | Absolute error |
|---|---|---|---|---|---|---|
| 512 | 512 | 512 | 12249 | 12249 | 13143 | 6.802% |
| 512 | 512 | 1024 | 21473 | 21473 | 24072 | **10.797%** |
| 512 | 1024 | 512 | 12919 | 12919 | 13085 | 1.269% |
| 512 | 1024 | 1024 | 22143 | 22143 | 23999 | 7.734% |
| 1024 | 512 | 512 | 12919 | 12919 | 13077 | 1.208% |
| 1024 | 512 | 1024 | 22143 | 22143 | 24046 | 7.914% |
| 1024 | 1024 | 512 | 14258 | 14258 | 13296 | 7.235% |
| 1024 | 1024 | 1024 | 23482 | 23482 | 24123 | 2.657% |

Every case passed exact graph comparison: operations and counts, all tensor
access regions, tensor shapes/strides/synthetic addresses, CTA resources and
mapping, and ordered issue/completion dependency lists. The one expected
placement metadata difference is explicit frontend HBM versus the reference's
HBF-tagged A/B routed by FullHBM. Cycle parity demonstrates that distinction
does not affect these runs.

For the failing case, both graphs contain **18,080 nodes**, **306,112 completion
edges**, and **3,024 issue edges**:

| Operation | Count |
|---|---|
| TMA input load | 640 |
| WGMMA fragment | 16384 |
| Register to shared store | 1024 |
| TMA output store | 32 |

The untouched reference reproduces the same error against measurement. This is
evidence that the discrepancy is outside the new IR translation; it does not
identify which baseline modeling assumption explains the measurement gap.
Changing IR metadata to force a fit would break the required builder parity.
The gate command therefore exits **2**, with `passed: false` in
`validation/results/tvm/manual/results.json`. That directory also contains
the eight expanded IR files, hashes, exact commands, graph summaries and both
simulation logs; `report.md` is the concise generated table.

**Original stop condition:** the strict per-case error threshold is `<10%`.
Work initially stopped here. The user then explicitly required ignoring this
stop condition and continuing the plan. Phases 3 and 4 below use that waiver;
the error remains recorded and no simulator or target data was changed.

## Phase 3: automatic TVM lowering proved for the supported scope

```sh
.tvm-venv/bin/python scripts/tvm/gemm.py 512 512 512 --run
.tvm-venv/bin/python scripts/tvm/validate_tvm_gemm.py --allow-measurement-error --jobs 4
```

All eight cases produce **byte-identical `.gtsir`** to the independent manual
specification and exact graph/cycle parity with the original builder. The cycle
and error table above therefore applies to all three paths. The report in
`validation/results/tvm/automatic/` records `translation_pass: true`,
`accuracy_pass: false`, and `measurement_accuracy_waived: true` independently.

The native TVM generator explicitly selects TMA, WGMMA, mbarriers, CTA output
joins and register redistribution. Generated TIR passes TVM SSA/well-formedness
verification and real simplification, flattening and intrinsic-lowering passes.
The frontend consumes a deserialized scheduled TVM module, validates the entire
known schedule, evaluates its access expressions and exports the common Tile
IR. It rejects altered/unknown instruction, binding, access and barrier patterns.
Production generation/lowering does not call or import the manual graph builder.

**Current scope:** automatic translation supports persistent CTAs across multiple
output tiles, with the TVM TIR's bulk-store completion wait and CTA barrier
preserved between tiles. The original eight validation cases use one wave.
Manual persistent-wave graph expansion remains available and tested. This is a bounded schedule frontend, not arbitrary
TVM kernel lowering. No CUDA numerical execution or hardware timing validation
was performed for generated kernels.

## Phase 4: constrained tile exploration executed

```sh
.tvm-venv/bin/python scripts/experiments/run_tvm_gemm_sweep.py --params scripts/experiments/tvm_gemm_tiles.csv --jobs 4 --repeats 2
```

Every candidate goes through native TVM generation and the common IR runner.
Hardware/resource/topology/divisibility/intrinsic checks precede simulation.
The launcher verifies current Phase 3 provenance before exploring and records
both the retained accuracy failure and the explicit waiver.

For M=N=K=512, all accepted schedules reproduced the same cycle count twice:

| BM | BN | BK | Simulated cycles | Fixed-baseline speedup |
|---|---|---|---|---|
| 64 | 128 | 32 | 10884 | 1.1254x |
| 64 | 128 | 64 | 7656 | 1.5999x |
| 64 | 128 | 128 | **6500** | **1.8845x** |
| 64 | 256 | 32 | 12337 | 0.9929x |
| 64 | 256 | 64 | 9105 | 1.3453x |
| 64 | 256 | 128 | 8037 | 1.5241x |
| 128 | 128 | 32 | 13432 | 0.9119x |
| 128 | 128 | 64 | 10072 | 1.2161x |
| 128 | 128 | 128 | 9446 | 1.2967x |
| 128 | 256 | 32 | 15553 | 0.7876x |
| 128 | 256 | 64 | 12249 | 1.0000x |
| 128 | 256 | 128 | rejected | requires 262144 shared bytes |

There are **11 distinct valid native TIR kernels and DAG specifications**. Best
tested simulated tile: **64x128x128**, **6500 cycles**, **1.8845x** the fixed
baseline. This is not a physical H100 result or an optimality claim. Full records,
commands, source/IR hashes, raw logs, input snapshots and the separate fixed
baseline are in `validation/results/tvm/tile-sweep/`.

## Regression checks

```sh
ctest --test-dir build --output-on-failure
python3 scripts/check_workload_profile.py
python3 scripts/run_reference.py --build-dir build --verify
```

All **13 CTests pass** with the optional TVM interpreter enabled, including invalid-schema/HBF/bounds/dependency/collective
checks, executable IR smoke, standalone GEMM fixture parity, and persistent-wave
parity with one, two, and three K stages and a clamped final column group. The
profile check passes. Both deterministic references remain unchanged:
GEMM **12,249 cycles**, FA3 **7,212 cycles**. Native TVM tests additionally compare
legal tile variants against the original builder and reject missing operations,
barriers or register redistribution, invalid dimensions/layouts, HBF topology
and hardware overflow. Multi-wave tests independently audit the TIR semantics,
including odd/even barrier phases and incomplete final waves. These green regression results are
separate from the failed Phase 2 measurement accuracy gate.


## Persistent multi-wave extension and 2048x2048x512 sweep

The automatic frontend now expands the authoritative TVM TIR's persistent wave
loop and final-wave guard. Its per-tile `cp.async.bulk.wait_group(0)` without
`.read`, followed by a CTA barrier, requires all prior output-store completions
before any thread starts its next tile. Completion edges into the next first
loads enforce this boundary; issue-only edges would be incorrect. The native
protocol follows [NVIDIA's PTX bulk wait-group semantics](https://docs.nvidia.com/cuda/archive/13.1.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group).
The C++ builder's different cross-wave overlap is not reproduced. The simulator
core, hardware configuration, pinned TVM source, manual builder and reference
cycles are unchanged.

Validation in `validation/results/tvm/automatic-multiwave/` retains exact graph
and cycle parity for all eight historical single-wave cases. Seven additional
multi-wave cases pass independent native-TIR trace checks for TMA regions,
barrier phases, input/output buffer reuse, four-warp collectives, issue order
and completion-event reachability; all seven run twice with identical cycles.
They cover K iteration counts 1/2/3/4, one/two compute warpgroups, two/four waves,
multiple output rows, clamped column groups and incomplete final waves. A
negative test weakening cross-wave completion dependencies to issue-only edges
fails the audit. All 13 CTests and the workload-profile check pass; the public
GEMM validation still reports MAPE 5.70198% (the existing 10.79678% case remains
waived). These are graph/synchronization and simulated-timing checks, not
numerical CUDA execution.

The existing `scripts/experiments/tvm_gemm_tiles_2048.csv` was executed without
changing its candidate rows. The baseline was regenerated and simulated twice
through the updated TVM pipeline: **26717 cycles**. Every feasible candidate ran
twice, and both samples matched exactly:

| BM x BN x BK | Waves | Simulated cycles | Baseline speedup |
|---|---|---|---|
| 64 x 128 x 32 | 4 | 46058 | 0.5801x |
| 64 x 128 x 64 | 4 | 37434 | 0.7137x |
| 64 x 128 x 128 | 4 | 37059 | 0.7209x |
| 64 x 256 x 32 | 2 | 39870 | 0.6701x |
| 64 x 256 x 64 | 2 | 34785 | 0.7681x |
| 64 x 256 x 128 | 2 | 26978 | 0.9903x |
| 128 x 128 x 32 | 2 | 41014 | 0.6514x |
| 128 x 128 x 64 | 2 | 33231 | 0.8040x |
| 128 x 128 x 128 | 2 | 32570 | 0.8203x |
| 128 x 256 x 32 | 1 | 31384 | 0.8513x |
| 128 x 256 x 64 | 1 | 26717 | 1.0000x |
| 128 x 256 x 128 | - | rejected | shared-memory footprint 262144 exceeds capacity |

**Best tested tile: 128x256x64, 26717 cycles, 1.0000x baseline.** All 11 feasible
configurations complete, including all nine that previously failed the one-wave
guard. Only 128x256x128 is rejected: 262144 shared bytes exceeds the unchanged
232448-byte per-CTA limit. No valid candidate is rejected for multiple waves.

Results, source/IR hashes, baseline, input CSV snapshot and both per-case logs
are in `validation/results/tvm/tile-sweep-2048x2048x512-multiwave/`. Previous
results remain in their original directories. To reproduce:

```sh
.tvm-venv/bin/python scripts/tvm/validate_tvm_gemm.py \
  --allow-measurement-error --jobs 4 \
  --out-dir validation/results/tvm/automatic-multiwave
.tvm-venv/bin/python scripts/experiments/run_tvm_gemm_sweep.py \
  --params scripts/experiments/tvm_gemm_tiles_2048.csv \
  --validation validation/results/tvm/automatic-multiwave/results.json \
  --out-dir validation/results/tvm/tile-sweep-2048x2048x512-multiwave \
  --jobs 4 --repeats 2
```
