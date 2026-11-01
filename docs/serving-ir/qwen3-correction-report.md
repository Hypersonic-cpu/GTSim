# Qwen3 decode correction evidence

The reviewed baseline is `f554d79f65b46718fda479140c036923496ee318`.
The canonical compiler migration is recorded in `2c3b887`; subsequent validation,
precision tests and CI adjustments are in the working tree. Work stays on
`serve-sim`. All six corrected programs and the final artifact/hash/stream audit passed.

## Architecture and workload changes

There is one top-level IR pipeline: `Algorithm` → `FlowKernel` → `ArchKernel` →
`ExecKernel`. The former `ir/decode.py`, `passes/decode_lowering.py`, and
`passes/decode_expand.py` are removed. The driver compiles every invocation using
`passes/instantiate.py`, `passes/hopper/lower.py`, and
`passes/lower_execution.py`; it releases each eager executable DAG before
compiling the next kernel. IR-03/04 retain structured loops, regions, workgroups,
SSA operands and explicit dependencies.

GEMM uses the existing mathematical contract and reference tiled Flow
construction. Small-M and grouped schedules extend the profile and group-row
mapping, using shared Hopper matrix patterns, copies and instruction emission.
The original reference profile remains intact. The grouped implementation is
precisely a **grouped small-M MMA baseline**, not a calibrated optimized grouped
GEMM. Reports expose active experts, assignment counts, physical M=16 tile rows,
masked rows, utilization, and amplification.

Decode attention uses the canonical FA3 algorithm family, schedule vocabulary,
streaming state/dependency abstractions, and shared Hopper Native math and
instruction expansion. Its explicit Tq=1 SIMT schedule lives in
`kernels/hopper/attention_decode.py`; stable split-result merging lives in
`attention_combine.py`. The original full-tile reference remains a separate,
unchanged profile within that framework. A decode query never creates 127 extra
logical tokens. Each query head maps to its own request and one of four KV heads.
Valid lengths include the appended token.

The layer builder declares tensors, bindings, routing and kernel invocations;
it contains no instruction generation. It preserves `attention_residual` as the
input of post-attention RMSNorm and of the final MoE residual add. The next layer
consumes the preceding layer's final output. Layer-specific weights and KV
allocations remain distinct. Q/K RMSNorm and position-dependent rotate-half RoPE
are explicit; V is extracted without normalization or rotation. SwiGLU reads
distinct Gate and Up regions. Router Softmax, Top-8 and renormalization remain
charged even when routing comes from a supplied or synthetic trace. Grouped
histogram/prefix and gather are separate timed launches. Expert row/slot/weight
identities persist through both GEMMs and the weighted sum.

## Evidence and remaining gates

| Gate                                                     | Current evidence                                                                                               |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Canonical serialized Flow/Native authority               | Dependency mutation, independent roundtrip and rejection tests in `tests/ir_frontend`                          |
| Residual tensor identities and cross-layer flow          | `test_qwen3_decode.py`; wrong same-shaped final residual is rejected by the semantic test                      |
| QK reduction, scale, online Softmax, PV, Split-LSE merge | Independent CPU interpretation of emitted Native math and direct attention reference                           |
| Q/K RMSNorm, RoPE positions/theta, SwiGLU regions        | Native numerical oracle and region assertions                                                                  |
| Routing, grouping, weighted reduction and KV append      | Numerical/trace oracles in `test_canonical_moe_memory.py`                                                      |
| BF16 storage and FP32 math                               | Optional rounded Native oracle, BF16 tie checks, rounded residual-add comparison and FP32 reduction assertions |
| Frontend suite                                           | 36 tests passed                                                                                                |
| Native build                                             | Fresh Release/Ninja build; subsequent build is up to date                                                      |
| CTest                                                    | 23/23 selected tests passed, including synchronization and sequential runtime checks                           |
| All measured reference timings                           | 8 GEMM + 6 FA3 CSVs byte-identical to frozen baseline                                                          |
| All measured reference graphs                            | All 8 GEMM + 6 FA3 graphs exactly match independent C++ oracle                                                 |
| B=1, one layer                                           | PASS: 446,472 cycles, 18 kernels                                                                               |
| B=8, one layer                                           | PASS: 2,015,147 cycles, 19 kernels                                                                             |
| B=16, one layer                                          | PASS: 3,078,542 cycles, 19 kernels                                                                             |
| B=1, two layers                                          | PASS: 892,944 cycles, 36 kernels; layer boundary at 446,472; different layer routes                            |
| B=8, two layers                                          | PASS: 3,933,819 cycles, 38 kernels                                                                             |
| B=16, two layers                                         | PASS: 6,232,262 cycles, 38 kernels                                                                             |
| Final six-case artifact/hash and stream audit            | PASS for all six cases; every manifest hash verified                                                           |

Reference GEMM MAPE/maxAPE remain 5.701982533931243% / 10.796776337653705%;
FA3 remains 4.953759564438511% / 6.822141804047714%. These accuracy numbers
apply to the calibrated reference kernels only.

CI explicitly initializes required `glaze` and `hbfine` submodules using HTTPS;
optional Ramulator2 and legacy TVM tests are outside its default build path.
The known textual `workload_profile_boundary` false positive is excluded as
requested. Local C++ compilation and CTest execution are verified; a remote
GitHub Actions run has not been claimed.

## Reproduction

Run from the repository root, using CMake 3.20+, a C++23 compiler, Ninja and Python.

```sh
git -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init --depth 1 -- 3rdparty/glaze 3rdparty/hbfine
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGTSIM_ENABLE_RAMULATOR2=OFF -DGTSIM_ENABLE_LEGACY_TVM_TESTS=OFF
cmake --build build --parallel 8
ctest --test-dir build -E '^workload_profile_boundary$' --output-on-failure
PYTHONPATH=frontend/src python3 -m unittest discover -s tests/ir_frontend -v
python3 tests/reference_parity/test_native_parity.py \
  --oracle build/reference_graph_dump --full-validation-matrix
python3 tests/reference_parity/test_native_parity.py \
  --oracle build/reference_graph_dump
python3 scripts/run_reference.py --build-dir build --verify
python3 scripts/run_validation.py --build-dir build \
  --out-dir validation/results/qwen3-correction/final-reference --verbose
PYTHONPATH=frontend/src python3 scripts/run_qwen3_correction_matrix.py \
  --runner build/gtsim_program --out-dir validation/results/qwen3-correction/final
PYTHONPATH=frontend/src python3 scripts/run_qwen3_correction_matrix.py \
  --runner build/gtsim_program --out-dir validation/results/qwen3-correction/final \
  --audit-only
```

The matrix script runs all one-layer cases before two-layer cases. It supplies
heterogeneous 10K–32K contexts for B=8/16. Individual commands, routing trace
format, split overrides and launch-overhead options are documented in
[qwen3-30b-a3b-decode.md](qwen3-30b-a3b-decode.md).

Each case under `validation/results/qwen3-correction/final/b{B}-l{L}` retains
`program.progir`, `program.gtsir`, each invocation's `.algoir/.flowir/.nvir/.gtsir`,
`simulation.log`, `simulator-timing.json`, `kernel-timings.json`,
`stream-timeline.json`, `metrics.json`, `workload.json`, `graph-summary.json`,
`lowering-manifest.json`, `manifest.json`, and a human-readable `results.md`.
Artifacts are intentionally ignored by Git. Kernel intervals use the persistent
global clock; device completion and memory-drain completion are separate fields.
Default configured launch overhead is zero.

## Fidelity limitations

New kernel latencies are uncalibrated H100 estimates. Neither graph correctness
nor the reference accuracy numbers establish decode latency accuracy. No
physical H100 measurement of these new schedules is available in this checkout.
The upstream [Hopper FlashAttention combination implementation](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_combine_kernel.h)
provides a structural comparison for stable LSE merging; its optimized schedule
is not reproduced here.

Numerical tests execute an independent CPU oracle, not GTSim tensor inference.
Most algebra tests use ideal casts; precision checks cover BF16 storage and
selected FP32 operations, not end-to-end model inference accuracy. KV storage
is contiguous and separated by request/head, not Paged KV. Routing is supplied
or seeded synthetic state. There is no prefill, extra decode step, sampling,
LM head, shared expert, prefetch, kernel cache, DSE or multi-stream execution.

## Final execution and memory summary

| Batch | Layers | Stream cycles | Layer durations  | HBM read bytes | HBM write bytes |
| ----: | -----: | ------------: | ---------------- | -------------: | --------------: |
|     1 |      1 |        446472 | 446472           |      134458752 |          191488 |
|     1 |      2 |        892944 | 446472, 446472   |      268917504 |          382976 |
|     8 |      1 |       2015147 | 2015147          |      923202304 |         1266048 |
|     8 |      2 |       3933819 | 2015147, 1918672 |     1761469952 |         2532096 |
|    16 |      1 |       3078542 | 3078542          |     1473946112 |         2530944 |
|    16 |      2 |       6232262 | 3078542, 3153720 |     2995078144 |         5061888 |

All cases report zero HBF bytes under the selected full-HBM configuration.
Every kernel completed with a positive execution interval and without overlap.
The artifact audit checks SHA-256 for every manifest artifact as well as
exact request contexts, current-token inclusion, token/expert counts, ordered
completion boundaries and persistent runtime flags. See `matrix-audit.json`
and each case’s `results.md` for the complete ordered per-kernel timeline.
