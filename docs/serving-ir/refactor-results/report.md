# Structured IR refactor results

Baseline: `9031d1aa8acff0e14705f9fc7eeb20c3b5579fcb` (`serve-sim`). Implementation: `c425f65eb6e74707e28484174e1e7c972048fa5e` on `refactor/structured-serving-ir`.

All requested reference behavior is preserved: **23 active CTests pass; all 8 GEMM + 6 FA3 cases have exact independent C++ graph/cycle parity.** No correctness-fix waiver or timing-parameter change was used. The corrected GEMM two-SM K=64/128/192/256 audit remains passing.

| Workload | Cases |        MAPE |       maxAPE | Change from frozen baseline |
| -------- | ----: | ----------: | -----------: | --------------------------: |
| gemm     |     8 | 5.70198253% | 10.79677634% |         0 percentage points |
| fa3      |     6 | 4.95375956% |  6.82214180% |         0 percentage points |

## Compact early IR, unchanged final DAG

IR-03 is a typed tile/pipeline loop body. IR-04 selects Hopper instructions, native warp/fragment loops, layouts and async protocol without enumerating CTAs. Only IR-05 performs full expansion. The final DAG remains eager; no lazy execution was introduced.

| Largest measured FA3 case (1,16,16,1536,1536) |      Before |   After |
| --------------------------------------------- | ----------: | ------: |
| Flow bytes                                    |  55,843,238 |  63,074 |
| Native bytes                                  | 477,057,129 | 100,960 |
| Flow body operations                          |      66,048 |      29 |
| Native instructions / patterns                |     982,272 |      29 |
| Final execution nodes                         |     982,272 | 982,272 |

All 14 per-case file sizes and counts are in `ir-sizes.csv`. The final `.gtsir` remains large and its node count is unchanged. Hash/provenance fields and intermediate schema changes mean byte equality of JSON files is not the graph-parity criterion.

## Semantic lowering and supported transformations

Flow operation roles, loop/tile/iteration identities, SSA producers/results, loop-carried initializers, slot generations, phased dependencies, regions and executors are typed fields. Display-name changes are tested to leave execution operations unchanged. Generic target patterns select TMA chunks, WGMMA collectives, SIMD/SFU/SHFL operations and store/synchronization steps. IR-05 resolves those actual native dependency records; it does not split or match Flow display IDs.

The implementation supports exactly the two fixed reference schedules. Noncanonical dependency/ready-vs-issue, slot/depth, output-region, mapping, native-layout or commit/wait transformations are **explicitly rejected** with operation/field diagnostics. This is an intentional support boundary, not acceptance of arbitrary schedule transformations. Mutation tests verify rejection rather than silently replacing modified semantics with a reference DAG.

Typed Algorithm contracts now carry operand/index domains, reductions, initialization, result axes/dtypes, QK scaling, FP16 P conversion, streaming max/sum recurrences and GQA mapping. Readable formulas are derived inspection output. The version-2 Algorithm/Flow/native schemas retain all requested file extensions; regenerate older compiler intermediates. Execution JSON and legacy text readers remain unchanged.

## Persistent runtime lifecycle

`KernelExecutionSession` factors initialization, admission/scheduler/scoreboard binding and retirement out of Simulator. Constructor and rebind share the initialization seam. Pending pipeline, TMA, SRAM, retry and remote operations prevent retirement. GPU/L2/MemorySystem and program allocations remain persistent. Tests cover GEMM→FA3, FA3→GEMM, growing/shrinking 1→8→1 CTA/warp ranges, changed resource footprints, stable bindings, warm-cache traffic, launch states, event clocks and explicit overhead. Concrete traces are in `runtime-evidence.json`.

**Limitation:** v0 still conservatively drains the complete memory system before the next launch. Cache contents and controller state persist, but cross-kernel asynchronous prefetch/overlap is not modeled.

## Verification and reproducibility

Formatting-only commits precede semantic commits. Final checks used Ruff 0.14.4, clang-format 23.1.3 and Prettier 3.6.2; developer dependencies and the formatting CI workflow pin their versions. Syntax/control-flow lint rules are E9/F63/F7/F82. CI configuration was added; remote CI was not executed in this local task.

`workload_profile_boundary` remains the user-authorized false-positive exclusion. Previous TVM code is unchanged; optional TVM tests remain OFF. No new algorithm, numerical execution, MoE, prefetch, DSE or concurrency functionality was added.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build -E '^workload_profile_boundary$' --output-on-failure
PYTHONPATH=frontend/src python3 -m gtsim_frontend.cli validate --out-dir validation/results/ir-refactor/final
ruff format --check frontend/src/gtsim_frontend tests/ir_frontend tests/reference_parity
ruff check --select E9,F63,F7,F82 frontend/src/gtsim_frontend tests/ir_frontend tests/reference_parity
```

Raw before/final logs, compiled stages and manifests are under ignored `validation/results/ir-refactor/`. Small frozen baseline, final accuracy, size and runtime records are tracked here. The original supplied measured CSVs and cycle reference manifest were not changed.
