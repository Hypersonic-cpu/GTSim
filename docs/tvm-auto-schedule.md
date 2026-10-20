# Experimental TE → MetaSchedule → GTSim optimization

The default workload is **512×512×512 FP16 GEMM**, with FP32 accumulation and
FP16 output, on the existing H100 FullHBM configuration. This experimental
frontend is separate from the v0.1.0 release scope. Simulator timing, hardware
profiles, reference cycles, and pinned TVM sources are unchanged.

The merged [heterogeneous-memory extension](tvm-memory-auto-schedule.md) also
supports device-scoped B placement and charged HBF-to-HBM prefetch kernels.

## Setup and execution

For a fresh workspace, first follow the complete
[TVM and GTSim setup](tile-ir.md#fresh-workspace-setup-and-phase-1-smoke).
It creates `.tvm-venv/`, builds the pinned compiler in `build-tvm/`, and builds
GTSim in `build/`. The environment survives deletion of `build/`.

Run from the repository root, choosing an unused output directory:

```sh
.tvm-venv/bin/python scripts/tvm/tune_gemm.py \
  --trials 32 --seed 0 --jobs 4 \
  --out-dir validation/results/tvm/auto-search-512

# Replay the best native scheduling trace, check computed values, regenerate
# identical Tile IR, and reproduce its simulated cycle count.
.tvm-venv/bin/python scripts/tvm/replay_gemm.py \
  --search-dir validation/results/tvm/auto-search-512

# Check every completed candidate, with a separate report for each one.
.tvm-venv/bin/python scripts/tvm/replay_gemm.py \
  --search-dir validation/results/tvm/auto-search-512 --all

ctest --test-dir build --output-on-failure
```

`--m`, `--n`, and `--k` default to 512. `--workload` accepts a Python module
exporting `workload(m, n, k)` returning A, B, C TE tensors. The example
[`te_gemm.py`](../workloads/gemm/te_gemm.py) specifies only placeholders, the
matrix-product reduction, accumulation dtype, and output cast. It contains no
tile sizes, warps, pipelines, barriers, or hardware instructions. No CSV is
used by the automatic search.

## Compiler and feedback path

1. `te.create_prim_func` constructs S-TIR from the computation.
2. `auto_rules.py` applies native split, reorder, cache-read, compute-at,
   storage-scope, thread-binding, reduction-decomposition, and tensorization
   transformations. Sampled factors remain native schedule random variables.
3. TVM's `InjectSoftwarePipeline` constructs prologue, steady state, tail, and
   shared ring buffers. `hopper_compiler.py` consumes the resulting access
   regions, loop order, tensorized operations, and bindings. It lowers cache
   transfers to TMA, distributes physical WGMMA registers, selects matching
   swizzled layouts, and inserts the instruction protocol and CTA joins.
4. `native_dag.py` consumes serialized native IR. It interprets actual bounded
   loops, guards, descriptors, thread bindings, barrier phases/transaction
   bytes, asynchronous commit/wait groups, and shared-buffer versions. These
   determine issue and completion dependencies. It checks physical operand and
   epilogue mappings against tensorization's preserved regions, without
   comparing the kernel to a fixed AST or reading the search parameters.
5. Native MetaSchedule `Builder`, `Runner`, JSON database, and evolutionary
   search evaluate the generated `.gtsir` through `gtsim_gemm_ir`. The runner
   returns simulated cycles divided by the configured core frequency as
   `run_secs`; compilation and wall-clock simulator time are excluded.
6. A custom MetaSchedule cost model fits log simulated cost from completed
   observations. Native evolution and trace-decision mutation use those
   predictions to select subsequent batches; an exploration fraction remains.
   `cost-model.json` and each history row record when feedback was available.

The pin (`4d0ac168c78c1e20bf05a6ec5926760bb1c52b05`) provides the native
scheduling, tensorization, pipeline injector, PTX intrinsics, and search APIs.
Its generic GEMM scheduling rules do not select the required Hopper TMA/WGMMA
kernel. The additional compiler rule and `TensorIntrin` implement that missing
path. `h100_schedule.py` remains the old manually scheduled baseline and is not
imported by the new scheduling rule or Hopper compiler. Its strict recognizer
is isolated in `reference_gemm_lower.py` for historical validation.

The compiler explicitly invalidates completed mbarriers before initializing
them for another persistent tile, as required by the
[PTX barrier lifetime rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-init).
TMA boxes and WGMMA descriptors use matching swizzled physical layouts; see
the [PTX WGMMA layout specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout).

## Supported space and validation boundaries

The space is derived from dimension factors, instruction granularity, TMA
layout requirements, shared memory, and register capacity. It explores BM,
BN, BK, ring-buffer depth, one or two compute warpgroups, and ordinary versus
persistent CTA grids. The current wire ABI has twelve warps per CTA. WGMMA
uses m64n64k16; physical TMA staging supports BK 16/32/64 and BN up to 256.
The number of stages is bounded by both K iterations and actual allocations,
including barrier storage. Unsupported layouts or operations fail explicitly.
These are compiler/backend support bounds, not a claim to explore every H100
implementation. The initial implementation stages both inputs and the output
through shared memory; it does not search arbitrary memory scopes or partial
async-group waits. The memory extension supports FullHBM and hybrid profiles;
FullHBF cannot satisfy the required HBM placement of A and C.

Every successfully lowered candidate must cover every C element once, execute
each required reduction fragment once, stay within access/resource bounds,
wait for input/output DMA, and avoid reusing shared storage before its
consumers complete. Removing synchronization, malformed descriptors, unknown
operations, divergent barriers, or invalid resource metadata are errors.
Structurally different loop-unrolled IR is accepted when its operations have
the same semantics.

`test_tvm_auto_schedule.py` covers one/two compute warpgroups, BK 16/32/64,
pipeline depth 1/2/3/8, odd/even reductions, buffer reuse, and a 135-tile
persistent grid with a three-tile final wave on 132 SMs. It also removes each
required synchronization operation and checks rejection. Multi-wave cases with
five and six K iterations exercise repeated reuse of a two-stage ring across
odd/even barrier phases. Per-case numerical
errors and cycles are written to
`validation/results/tvm/auto-schedule-tests/correctness.json`.

The replay checker independently compares interpreted lowered instruction
values with integer GEMM for seeded small integer FP16 inputs, followed by
the specified FP16 cast. These inputs have exact FP32 reductions. It checks
every output and reports maximum absolute error and mismatch count per
candidate. GTSim evaluates timing and does not execute numerical tensor data;
these checks are software instruction interpretation, not CUDA execution or
on-device accuracy validation. Reassociation can change FP32 rounding for
general noninteger inputs. A fixed search budget reports the best measured
candidate, without a global-optimality claim.

## Result files

| Artifact | Contents |
|---|---|
| `report.md`, `search-history.json` | All measured candidates, cycles, speedups, failures, prior feedback count |
| `results.json` | Best result, repeated baseline, seed, versions, compiler manifest, source/config hashes |
| `space.json`, `postproc-pruning.json` | Derived factor space and resource-postprocessor rejection counts |
| `cost-model.json`, `database_*.json` | Observations, feedback rounds, native MetaSchedule records |
| `workload.py`, `computation.py/json` | User computation and initial S-TIR |
| `candidate-NNNN/trace.py/json` | Native scheduling decisions and replayable trace |
| `candidate-NNNN/scheduled.py/json` | Tensorized scheduled S-TIR |
| `candidate-NNNN/pipeline.py` | Native software-pipeline pass output |
| `candidate-NNNN/native.py/json` | Actual hardware operations consumed by the DAG lowerer |
| `candidate-NNNN/lowered-tvm.py` | Compiler check through FlattenBuffer and LowerIntrin |
| `candidate-NNNN/kernel.gtsir`, `lowering.json`, `simulation.log` | DAG, semantic/resource evidence, simulator command/output |
| `manual-baseline/` | Existing manually scheduled GEMM, simulated twice |
| `replay-all/verification.json` | Numerical errors and repeated cycles for every completed candidate |

The best candidate is also simulated twice during search. Replay verifies
source/configuration identity and refuses changed inputs. Native trace JSON is
stored using `Trace.as_json()` and replayed with `Trace.apply_json_to_schedule`;
serializing the Trace object itself is unsuitable for this pin's callback
metadata. Resource-postprocessor counts describe proposals pruned before
measurement; native trace-transformation failures are recorded in TVM's logs.

## Recorded 512×512×512 search

With seed 0, 32 trials, and four simulation workers:

| Implementation | BM × BN × BK | Simulated cycles | Baseline speedup |
|---|---|---|---|
| Existing manual baseline | 128 × 256 × 64 | 12,249 | 1.0000× |
| Best measured native schedule | 64 × 128 × 16 | 6,439 | 1.9023× |

The first best schedule uses 32 stages, two compute warpgroups, and a
32-CTA persistent grid, with 214,016 shared bytes and 81,920 register bytes per
CTA. An eight-stage candidate also reaches 6,439 cycles. All 32 measured
candidates completed, producing 32 distinct native modules and 25 distinct
Tile IR graphs. Eight cost-model feedback updates were applied. The resource
postprocessor rejected 268 register-limit, 262 shared-memory, and 306
tile/pipeline-legality proposals; counts include repeated proposals.

All 32 candidates passed trace replay, identical Tile IR/cycle reproduction,
and the seeded numerical check with zero error and zero mismatches. A second
full search reproduced all choices, native/Tile IR hashes, cycles, and
cost-model observations. The complete records are in
`validation/results/tvm/auto-search-512-seed0/`, including `report.md`,
`replay-all/verification.json`, and `reproducibility.json`; the independent
search is in `validation/results/tvm/auto-search-512-seed0-repeat/`.
