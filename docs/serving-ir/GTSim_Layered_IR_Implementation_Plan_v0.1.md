# GTSim Layered IR Framework — Implementation Plan
**Status:** Implementation contract / agent-executable draft
**Base:** `Hypersonic-cpu/GTSim`, branch `serve-sim`, inspected tree `16a7f6aa8d29d32b9e5d0cacd6d1bc0b9b67b69b` (2026-10-31)
**Spec:** `llm_serving_ir_design_v0_2.md` (IR-01…05)
**Comparative reference:** `SJTU-ReArch-Group/GTSim`, public `main`, inspected tree `45a996c1585e60e98194057d94f7e664d41150ba`
**Independent IR reference:** uploaded `uniform-sim-ir-main.zip` (`uniform-flow-ir/1.0`, Python reference implementation)
## 0. Deliverable and non-goals
**Deliverable:** two executable *single-kernel, single-stream* top-level programs, `gemm` and `fa3`, each traversing `IR-01 → [IR-02 + explicit reference schedule] → IR-03 (+ mapping) → IR-04 (Hopper) → IR-05 → GTSim`, with configurable **official-baseline-legal** input shapes; their generated graphs, cycles, and validation statistics must be independently compared to baseline oracles. The public C++ reference builders remain available separately.
**In scope:** IR data models, JSON codecs/printers/verifiers, lowering and provenance, fixed reference GEMM/FA3 schedules, Hopper backend, generic execution graph materialization including FA3 SIMD/SFU nodes, **one-stream sequential multi-launch semantics in GTSim**, per-program GPU/context persistence, two single-launch example programs, tests and measured-data validation.
**Out of scope:** full Qwen decode step, whole transformer, MoE grouped kernels, TVM integration, automated IR-02→03 DSE, concurrent kernel execution and multi-stream scheduling, speculative/SW/HW prefetch, general CUDA/PTX compilation, Blackwell/Ampere lowering, numerical model execution. HBF topology stays in the existing memory model but is not a parity-test variable. **Do not claim `seq_q=1` decode parity to official FA3:** the official public FA3 baseline is a full Q-tile algorithm; fork additions for `seq_q=1` are explicitly disabled for the reference path.
**Design invariants:** no hidden direct reference-DAG import in the production lowerers; no timing constants or per-workload delay overrides; all lowerings deterministic and separately inspectable; unsupported shapes/ops fail closed; one simulator owns all timing.
## 1. Verified source facts — implement against these, not assumptions
| Finding | Confirmed source / consequence |
|---|---|
| Official `GTSim v0.1.0` publishes only H100 persistent **two-stage GEMM** and **two-stage FA3** C++ builders, not its whole paper frontend. | `SJTU-ReArch-Group/GTSim` README and `docs/release-scope.md`; use public builders as algorithm oracles, not unshipped paper code. |
| Official GEMM uses `BM=128, BN=256, BK=64`, **12 warps**, persistent CTA waves, two SMEM stages and original grouped-column CTA order. | Official `workloads/gemm/gemm_h100.cpp`. |
| Official FA3 uses `Q_TILE=128, KV_TILE=128, head_dim=128`, two-stage K/V buffering, two matrix operations (QK/ PV), online softmax with separate `SIMD/SFU/SHFL` nodes, and GQA index mapping. | Official `workloads/fa3/fa3_h100.cpp`. No `split-KV`, alternative CTA order, q=1 padded decode or bf16/head_dim=256 mode in the official reference. |
| Fork GEMM has **438** lines vs official **298**, FA3 **722** vs **533**; new schedule, placement, split, single-token, dtype and other features exist. | Both `workloads/*/*_h100.cpp` files; default options are not proof of reference algorithm parity. |
| **Synchronization difference is concrete.** Official GEMM accumulates prior K-tile TMA IDs in `tma_deps` (completion edge); fork baseline adds `add_issue_dependency(previous_tma, new_load)` and separately guards two-stage buffer reuse with WGMMA completion dependencies. | Official GEMM around its K-loop, fork GEMM around `previous_tma_issue`. This difference **must be reviewed as a true issue-vs-completion and buffer-lifetime question**, not silently treated as an innocuous refactor. The source comparison alone does **not** establish which behavior is correct. |
| Fork reference cycle file changed: GEMM `12956 → 12249`, FA3 `7168 → 7212`. Reasons in fork: GEMM TMA stage issue ordering; FA3 added D2D request/response path. | Official vs fork `configs/v0.1.0/references.json`. Do **not** demand both numbers from the same graph + hardware configuration. |
| Original measured CSVs are byte-identical between fork and official (8 GEMM / 6 FA3). Fork `validation/v0.1.0.md` remains byte-identical to official and thus is **not** current-fork recalibration evidence. | Both `validation/data/h100_gemm.csv`, `h100_fa3.csv`, `validation/v0.1.0.md`. Reuse datasets; recompute metrics rather than copying stale document summary. |
| Official published v0.1.0 validation: GEMM **2.65% MAPE**, Pearson **0.999758**; FA3 **9.79% MAPE**, Pearson **0.999679**. | Official `validation/v0.1.0.md`. This is for **public v0.1.0** and its backend, not all configurations in the MICRO paper. |
| Fork's *previously recorded* GEMM validation data: for the eight identical measured points, simulated cycles `[12249,21473,12919,22143,12919,22143,14258,23482]`; recomputed **5.70198% MAPE**, **10.79678% max APE**. | Fork `docs/tvm-gemm-status.md`, independently recomputed from CSV; historical report, **not a fresh execution of current commit**. Current-fork FA3 6-case MAPE remains to be measured. |
| Existing `gtsim.tile_ir 1` recognizes TMA/WGMMA/loads/stores, but not FA3-specific `local_max`, `exp2`, `logsum`, `acc_o_scale` etc. `gtsim.tile_ir 2` is mainly serving Ampere MMA/SIMD and rejects Hopper WGMMA/TMA. | Fork `src/tile_dag_builder.cpp` opcode tables. **A new compatible execution-graph serialization/materializer is needed for full FA3; do not force it through v1/v2 unchecked.** |
| `gtsim_core`, `gtsim_tile_ir`, `gtsim_gemm`, `gtsim_fa3`, `gtsim_gemm_ir`, `gtsim_serve_ir` already exist. TVM tests only register if `GTSIM_TVM_PYTHON` is supplied. | Fork `CMakeLists.txt`. Default core build is already TVM-independent. |
| `Uniform` provides `FlowKernel`, `TensorRegion`, `TileValue`, `Executor`, `Buffer`, `SlotHandle`, `Event`, `PipelineRegion`, `ScheduleRegion`, `FlowJSONCodec` and `FlowVerifier`, plus `MappingProgram` bindings. Its v1.0 builder **eagerly materializes** pipeline iterations. | Uploaded `src/uniform_sim_ir/flow/*`, `mapping/*`; IR-03-compatible source of API patterns. It does not automatically provide exact GTSim H100 FA3 graph semantics. |
| Uploaded Uniform zip has `pyproject.toml` and no obvious top-level `LICENSE` in inspected archive. | Before copying code into fork, verify provenance/redistribution permissions; otherwise use a documented local external dependency or independently implement only necessary interfaces. Do not silently vendor third-party code. |
**Validated before implementation:** source structures, reference numbers, CSV identity, fork GEMM recorded statistics. **Not executed in this review:** current-fork CMake/CTest or full 8+6 simulation; the implementing agent must run them and report actual logs.
## 2. Repository layout (minimal invasive)
Keep existing core directories, binaries and reference entrypoints; add an isolated Python frontend plus a **small C++ runtime layer**. The frontend builds and lowers IR; the C++ runtime processes kernel/stream semantics. Do **not** relocate `src/`, `include/`, `workloads/gemm/`, `workloads/fa3/`, or `scripts/tvm/`.
```text
GTSim/
├── CMakeLists.txt                         # append new targets; old binaries unchanged
├── include/
│   ├── runtime/                            # NEW: execution behavior, NOT IR-01 compiler
│   │   ├── execution_program.h             # parsed IR-05 command/launch/tensor contract
│   │   ├── program_executor.h              # stream command queue, lifetime, clock
│   │   ├── gpu_context.h                   # shared GPU + MemorySystem + allocations
│   │   └── kernel_dispatcher.h             # per-launch scheduler/scoreboard session
│   └── execution_ir/                       # NEW: IR-05 kernel graph interchange
│       └── kernel_graph.h
├── src/
│   ├── runtime/                            # NEW: light single-stream runtime
│   │   ├── execution_program.cpp           # runtime manifest validation/IDs
│   │   ├── program_executor.cpp            # launch/complete/event/synchronize
│   │   ├── gpu_context.cpp                 # persistent GPU + global cycle
│   │   └── kernel_dispatcher.cpp           # bind/release DAG scheduler, NOT GPU
│   └── execution_ir/                       # NEW: IR-05 adapters only
│       ├── graph_codec.cpp                 # JSON kernel/program parser
│       └── graph_materialize.cpp           # per-kernel DAGNode creation, remapping
├── workloads/
│   ├── gemm/gemm_h100.cpp                 # legacy reference + separately audited sync
│   ├── fa3/fa3_h100.cpp                   # legacy reference, existing options kept
│   └── serve/
│       ├── serve_ir.cpp                    # existing executable unchanged
│       └── program_runner.cpp             # NEW: gtsim_program CLI
├── frontend/                               # NEW; has no dependency on C++ runtime internals
│   ├── pyproject.toml
│   └── src/gtsim_frontend/
│       ├── __init__.py
│       ├── ir/
│       │   ├── common.py                  # IDs, shapes, dtypes, events, regions
│       │   ├── program.py                 # IR-01: Program/Stream/KernelInvocation
│       │   ├── algorithm.py               # IR-02
│       │   ├── flow.py                    # IR-03 Uniform-style FlowKernel
│       │   ├── mapping.py                 # IR-03 ResourceMapping adjunct
│       │   ├── arch.py                    # IR-04
│       │   └── execution.py               # IR-05 serializable program/kernel model
│       ├── dialects/
│       │   ├── gemm.py
│       │   ├── fa3.py
│       │   └── hopper.py
│       ├── kernels/hopper/reference/       # fixed IR-03 schedules, not DAG fixtures
│       │   ├── gemm.py
│       │   └── fa3.py
│       ├── passes/
│       │   ├── instantiate.py             # IR-01 + IR-02 → IR-03 (fixed schedule)
│       │   ├── legalize.py                # IR-03 + mapping + hardware verify
│       │   ├── lower_hopper_gemm.py       # IR-03 → IR-04
│       │   ├── lower_hopper_fa3.py        # IR-03 → IR-04
│       │   ├── lower_execution.py         # IR-04 → IR-05 kernel graph
│       │   └── compose.py                 # IR-01 + kernel graphs → IR-05 program
│       ├── serialization/
│       │   ├── json_codec.py              # deterministic versioned round-trip
│       │   └── printer.py                 # Uniform-like inspection output
│       ├── verify/                         # stage verifiers; backend capability checks
│       ├── adapters/
│       │   ├── uniform_flow.py            # optional, isolated Uniform v1.0 adapter
│       │   └── legacy_gtsim.py            # metadata/parity adapter, not DAG builder
│       ├── programs/
│       │   ├── gemm_single.py             # parameterized single-Stream/single-GEMM
│       │   └── fa3_single.py              # parameterized single-Stream/single-FA3
│       ├── driver.py                       # compile, emit, execute, report
│       └── cli.py                          # gtsim-ir compile|run|inspect|verify
├── tests/
│   ├── ir_frontend/                        # NEW: schema, lowering, verifier tests
│   ├── execution_program_tests.cpp        # NEW: 1+2 launches, stream/runtime semantics
│   └── reference_parity/                  # NEW: independent graph/cycle oracles
├── validation/
│   ├── data/                              # original measured CSVs unchanged
│   └── results/ir-framework/              # generated and gitignored
├── docs/
│   ├── ir-framework-spec.md               # versioned IR specification
│   ├── ir-framework-implementation.md     # this implementation contract
│   └── ir-framework-compat.md             # format/ops/profile support
├── scripts/tvm/                           # LEGACY, retained but not imported
├── tests/test_tvm*.py                     # LEGACY, optional as before
└── 3rdparty/tvm/                          # LEGACY, not a default dependency
```
**Ownership:** `frontend/ir/` stores typed immutable representations; `frontend/passes/` performs compilation and composition; `src/execution_ir/` parses/materializes the already-lowered IR-05; `src/runtime/` implements stream commands, kernel lifecycle and GPU state. No `src/program_ir/`: the IR-01 Program parser and compiler belong in `frontend/`, whereas the C++ runtime consumes only `exec.program` in IR-05. Keep `GPUContext`/`KernelExecutionSession` naming flexible if a thinner split within existing `Simulator` is cleaner, but preserve their separate lifetimes. Do not create a second timing engine.
**Uniform reuse:** adopt Flow/Mapping semantics and versioned JSON conventions; prefer a pinned, license-compatible external `uniform_sim_ir.flow` adapter if feasible, otherwise implement a narrow compatible subset. Verify distribution rights before copying code. Offline build must not depend on Uniform or TVM downloads.
## 3. IR contracts, interfaces, lowering ownership
Use deterministic JSON as the authoritative interchange, saved under `.progir` / `.algoir` / `.flowir` / `.nvir` / `.gtsir` respectively. These extensions name the **IR layer**, not a claim that the payload is MLIR syntax. Human-readable MLIR-like print format is for inspection only. Each IR includes `{format, schema_version, symbol, source_hash, ...}`. No parser for the inspection syntax is required. All IDs are deterministic/stable within their declared scope; unresolved symbolic dimensions are rejected before IR-05.
| Stage | Inputs | Output | Pass and invariant | Reject conditions |
|---|---|---|---|---|
| IR-01 | `Program(streams, tensors, launches, events)` | `program.progir` | Owns global tensor identities and kernel invocation graph | Unknown callee, unbound tensors, invalid event, cycle, multiple streams in v0 |
| IR-02 | `AlgorithmTemplate(gemm / fa3)` | `kernel.algoir` | Owns algorithm stages, contracts, implementation family; no BM/BN/BK yet | Invalid dimensions, inconsistent GQA, unsupported algorithm family |
| Instantiate | IR-01 invocation + IR-02 + **explicit fixed reference schedule** | `kernel.flowir` (embedded mapping) | Instantiate known tiled/pipelined implementation; **no optimization/search** | Non-reference options or unsupported geometry, malformed stages |
| IR-03 | `FlowKernel` + `MappingProgram` | canonical scheduled Tile/Buffer/Events | Producer/consumer roles, data regions, SMEM slot generations, online-softmax stage graph, issue vs ready | Slot reuse, lifetime, missing producer, wrong region/coverage, unsupported mapping |
| IR-03→04 | Flow + mapping + target H100 profile | `kernel.nvir` | **GEMM**: TMA + four-warp WGMMA + epilogue; **FA3**: Q/K/V TMA + QK/online softmax/PV + SFU/SIMD/SHFL + output; resource allocation and synchronization | Missing target capability, overcapacity, wrong collective, inconsistent event phase |
| IR-04→05 | Arch native kernel | `kernel.gtsir` | Unroll bounded CTA/warp/iteration events; turn barriers / completion into explicit deps; preserve *exact op kind* and mapping | Unknown op, deadlock, dangling dependency, cycles, incomplete writes |
| Compose | IR-01 + IR-05 kernel(s) | `program.gtsir` | Single ordered stream, **one or more sequential kernel launches**; stable tensor ID ↔ local ID bindings; launch/complete/event/sync commands | Multiple streams in v0, dangling events, alias mismatch |
| Execute | Execution Program + hardware YAML | per-kernel/program timelines and cycles | C++ `ProgramExecutor` processes ordered commands; per-launch scheduler uses **one persistent GPUContext/MemorySystem/global clock** | Unsupported streams/op/capacity, invalid graph, timeout |
**Recommended API shape (illustrative Python, not source syntax):**
```python
p = Program("gemm_once").stream("compute")
callee = templates.gemm(M, N, K, dtype="fp16")
launch = p.launch("compute", callee, inputs={"A": "A", "B": "B"}, outputs={"C": "C"})
flow, mapping = reference_schedule.instantiate(callee, launch, target="sm90")
arch = hopper.lower(flow, mapping)
kernel = execution.lower(arch)
exec_program = compose(p, {launch.id: kernel})
result = run(exec_program, arch_yaml=..., memory_yaml=...)
```
**IR-02 contents (GEMM):** `C=cast_fp16(sum_k(cast_fp32(A)*cast_fp32(B)))`, reduction coverage, stage family `[acquire_A_B, matrix_accumulate, output_stage]`, contract for canonical output tiles, deterministic initial zero of accumulator. *Do not place `after_complete=[...]` or WGMMA in IR-02.*
**IR-02 contents (FA3):** scaled QK, streaming max/sum online softmax (`m`, `l`, output accumulator), PV accumulation, final output normalization, exact head/GQA mapping and causal/mask policy as implemented by official builder. Record whether online-softmax numerical semantics are modeled or only timing; do not silently assume exact numerical CUDA behavior. Family `[Q_load, KV_load, QK, online_softmax, PV, normalize, store]`.
**IR-03 contents:** explicit `TensorRegion`, `TileValue`, `flow.executor`, `flow.buffer slots=2`, `flow.acquire`, `flow.load_async`, `flow.matmul`, `flow.compute`, `flow.store_async`, `flow.release`, structured `flow.pipeline`, Events (`issue`, `ready`, `complete`); `MappingProgram` is serialized as an embedded `mapping` section of `kernel.flowir` (`map.execute`, `map.store`, `map.tensor`, routes); a separate mapping printer may emit debug text, not another canonical IR file. Buffer allocation capacity comes from hardware config. Reference `flow.pipeline` can be expanded eagerly (Uniform v1.0 behavior), but must preserve slot generation and readiness; never use a single untyped `compute` node to hide whole FA3 softmax or WGMMA sequence.
**IR-04 contents:** per-target `ArchInstruction {op, executor, warp_group, tensor_region, layout, input_events, output_events, resource_requirements}`; materialize exact native operations of official algorithm including Softmax stages. This **is not SASS**; no ptxas dependency. Define a `HopperReferenceDialect` with explicit supported op schema; fail for `tcgen05` and unsupported extras.
**IR-05 contents:** `ExecProgram {streams, ordered_commands, launches, tensors, events}`, `ExecKernel {ctas, ops, tensors}`. Commands are `LaunchKernel`, `RecordEvent`, `WaitEvent`, `StreamSynchronize` with distinct kernel launch/completion IDs. A kernel record has `pipeline_type`, `op_name`, Warp/CTA identity, region, dtype, `after_issue`, `after_complete`, and FA3 instruction metadata. Bind kernel-local tensor IDs to a **persistent program-global TensorRegistry**; never reallocate the same logical tensor at each launch. Do not serialize simulated latencies or physical addresses; GTSim remains the address/timing authority.
**Execution format decision:** retain the legacy `.gtsir` v1/v2 text readers *unchanged*. The new export filenames are `kernel.gtsir` (complete kernel DAG including FA3 `SIMD/SFU/SHFL`) and `program.gtsir` (stream/launch manifest referencing `kernel.gtsir`), with **versioned JSON payloads distinguished by required `format` discriminators**, e.g. `gtsim.exec.kernel/1` versus `gtsim.exec.program/1`. Only the new `gtsim_program` runner dispatches the new JSON schema; existing v1/v2 parser and binaries continue to accept their original textual `.gtsir`. The new materializer constructs the same `DAGNode` types as legacy builders and dispatches them through a **new `ProgramExecutor` reusing GTSim's existing GPU/SM/memory timing engine**. It must not instantiate `Simulator(DAG*, config)` separately for each kernel: that constructor recreates GPU, cache, memory and cycle state. Refactor the ownership/lifecycle while preserving a single-DAG compatibility API for legacy binaries. No fake `SIMD`-to-WGMMA substitution. Optional `--emit-legacy-gtsir` for GEMM-only graph parity is useful but not mandatory to run FA3.
### 3.1 Single-stream runtime contract (required, not a frontend-only mock)
**Current fork limitation (confirmed):** `Simulator(DAG*, config)` owns `new GPU(...)`, configures `warp_count` from the DAG, initializes per-node scoreboards/schedulers and begins at cycle 0; `Simulator::run()` drains outstanding memory-system requests on exit. There is no separate multi-kernel Stream/Kernel lifecycle. Thus `Simulator(gemm).run(); Simulator(fa3).run()` is **not** a valid Program implementation, even for an ordered stream.
**Runtime ownership and semantics:**
- `GPUContext` owns one GPU/MemorySystem, program-global TensorRegistry/address allocations and a **monotonic** global cycle. Do not invalidate L2 or recreate memory controllers at a kernel boundary. Keep the program's hardware profile immutable.
- `KernelExecutionSession`/`KernelDispatcher` binds a freshly materialized kernel DAG to CTA/warp schedulers, per-launch dependency scoreboard, completion count, and per-launch IDs. Retire these *only after* kernel completion; do not reset shared GPU resources.
- `ProgramExecutor` owns exactly one ordered `Stream` in v0. Process `LaunchKernel` (submit, run until output-complete), `RecordEvent` (mark completion), `WaitEvent` (block until recorded), `StreamSynchronize` (wait all prior commands). Reject **more than one Stream**; do not silently serialize them.
- **Completion boundary:** all DAG nodes that contribute to visible outputs have completed and any required outstanding writes/visibility effects are satisfied. Distinguish last-node completion from arbitrary request issue and from a full cache flush. Refactor current `run()` memory drain so it does not reset/flush memory between kernels; preserve legacy single-DAG output/cycles under its compatibility API.
- **Clock/overhead:** global Cycle remains continuous across launches. Introduce an explicit optional `launch_overhead_cycles` (default `0` for original reference parity); report device-kernel cycles and program-total cycles separately. Avoid implicit startup/reconfiguration overhead.
- **Tensor consistency:** Program TensorIDs/allocations are stable; per-kernel tensor IDs resolve through immutable invocation bindings, with dtype, region, tier and alias verification. Invalid alias or released allocation must fail before simulation.
- **v0 timing model:** sequential device kernels only. Schema may include Event IDs for forward compatibility, but no cross-stream or simultaneous kernel admission yet. The two requested entrypoints each launch one kernel; a **third test-only two-kernel program** must demonstrate actual sequential semantics and preserved GPU/L2 state.
**Suggested C++ seam:** `ProgramExecutor::run(const ExecutionProgram&)` → `KernelDispatcher::execute(const ExecKernel&, GPUContext&)` → existing GPU `step()` loop and memory backends. Refactor the legacy `Simulator` with composition/helper extraction where possible; do not duplicate memory/cache timing code. Report `KernelTiming {launch_cycle, start_cycle, complete_cycle}` and `ProgramTiming {total_cycles, per_kernel}`. Per-kernel completion must be validated at the actual output visibility boundary.
## 4. Reference algorithm fidelity — exact requirements
### GEMM reference family (`ref_h100_gemm_v0_1_0`)
- Defaults **BM128 × BN256 × BK64, stages=2, split=1, 12 warps**, 4+4 compute warps and producer from warp 8, persistent CTA waves, official original CTA ordering; A/B TMA into two SMEM stages, four-warp collective WGMMA m64n64k16 in FP32, staging/store C, no undocumented fusion or shortcuts.
- **Kernel shapes:** official reference divisible by BM/BN/BK. For v0 reject M=1, masked tails, split-K, nonoriginal CTA orders and alternate dtypes unless separately tested as future variants. `M=N=K=512` and all official validation shapes accepted.
- Tensor allocation and op metadata must match reference including TMA A/B tile regions, 64-wide B chunks, WGMMA fragment regions and counts, CTA resources, source-tier/leading dimension, CTA/warp assignment, all stages. Use stable per-op labels carrying `(cta,wave,tile,k,warp,stage,op)` to compare independent builders.
- **Synchrony defect review:** before using legacy fork builder as new oracle, write a focused dependency specification for TMA stage issue-order, buffer slot reuse, final consumer, output-store completion and wave boundaries. Examine both official `tma_deps` and fork `previous_tma_issue`; do not assume all prior TMA transfers must complete just for issue ordering, nor that issue-order is sufficient for buffer reuse. Emit targeted tests for reuse phase parity K=64/128/192/256, `SM_count=2` multiwave, output-tile carryover, cross-wave store→next-tile loads and full CTA completion. Fix **only demonstrable synchronization errors** in the shared reference implementation. Add explicit `sync_semantics_version` to graph provenance.
- Any intentional sync fix invalidates byte-for-byte official dependency parity but **not algorithm parity**. Diff only the approved issue/complete edges; counts, arithmetic, memory regions, resource metadata must remain equal. Record old/new cycles and changed edges; never silently modify timing parameters to restore a reference number.
### FA3 reference family (`ref_h100_fa3_v0_1_0`)
- Baseline **Q_TILE128, KV_TILE128, D128, stages=2, split=1, fp16**, standard original CTA order, 12-warps and original two-stage Q/K/V/TMA/WGMMA execution. Include the **full original** SIMD/SFU/SHFL sequence for online softmax and output normalization. Preserve GQA mapping as official builder supports it.
- Do **not** enable fork-only `seq_q=1` padded decode, D256, bf16, split-KV/reduction, configurable nondefault q/kv tiles, alternate CTA order or newly added FA3 kernel features in the parity profile. For input parameters, support official validation shapes (`batch`, `heads_q`, `heads_kv`, `seq_q`, `seq_kv`) and reject unsupported geometries with explicit reasons.
- Compare exact node multiplicities of TMA Q/K/V, QK WGMMA, softmax (max/shuffle/reduce/exp/sum/scale), PV WGMMA, normalize/output TMA, barrier/issue/completion dependencies, and tensor IDs/heads. If fork D2D-path changes timing under same Config, attribute the difference to backend, not kernel algorithm.
- Keep all existing fork FA3 optional features and tests intact; **new framework** opts into `reference_v0_1_0` explicitly. Do not delete feature code or change its defaults globally just to pass parity.
### Reference isolation / no circular oracle
- Production compiler may inspect reference source during development but may **not call** `build_gemm_h100`/`build_fa3_h100`, `manual_gemm`, `reference_gemm_lower`, or load precomputed `.gtsir` as its algorithm implementation.
- Parity-test binaries may link existing reference builders using existing `GTSIM_WORKLOAD_LIBRARY` mechanism; include an *independent* official-reference snapshot/result where possible, with documented upstream tree SHA. Never make reference and candidate share a single DAG-creation routine or a copied graph fixture as the sole proof.
## 5. Implementation sequence and acceptance gates
| Phase | Change set (committable and testable) | Required tests / exit criteria |
|---|---|---|
| **P0 Baseline freeze** | Pin inspected upstream/fork refs; run build + original `run_reference.py --verify`, `run_validation.py`; archive baseline logs and graph fingerprints with hardware YAML, memory backend and Git SHA. Identify precise GEMM synchronization edge issue before patching. | Baseline recorded; all existing CTests pass or preexisting failures enumerated. No code changes until evidence captured. |
| **P1 IR skeleton** | Add `frontend/` typed immutable IR-01, IR-02, IR-03, Mapping, IR-04 and IR-05 classes; format-versioned codecs/printers, validators; two Program examples with param bindings. | Round-trip JSON equality; stable hash; missing tensor/invalid stream/unknown template/incompatible dimensions rejected; not dependent on TVM/CUDA Python import. |
| **P2 Program + reference templates** | Implement single-launch GEMM/FA3 IR-01 builders, explicit algorithm families IR-02, **fixed** Hopper reference Flow IR03 builders. No optimizer. Uniform-style logical executor, buffer, slot, event usage. | Assert exact logical regions, online softmax stages and buffer lifetimes against hand-computed coverage; inspect flow and map dumps. |
| **P3 Hopper lowering** | IR03→IR04 GEMM and FA3; explicit native op families, complete synchronization phases, independent per-CTA resource calculation; verifier. | Golden native-op counts/regions; fail on missing WGMMA collective, async waits, barriers and invalid buffer lifetime. No cycle model inside compiler. |
| **P4 DAG lowering + runtime** | IR04→IR05 generic op compiler, `kernel.gtsir` + `program.gtsir`, new kernel materializer and **ProgramExecutor** with one ordered Stream, kernel launch/completion/sync commands, persistent GPUContext/global cycles, per-launch scheduler lifecycle. Preserve legacy single-DAG API and parsers. | Graph parity incl. all FA3 ops; two requested single-launch programs pass. A **GEMM→FA3 sequential test** proves global cycle, cache/memory/tensor persistence; multiple streams rejected. No operation elided or replaced by fake latency. |
| **P5 GEMM sync audit/fix** | Implement smallest correctness fix if tests demonstrate defect, make old/new edge delta explicit and re-run full suite. Keep algorithm and timing profiles unchanged. | Odd/even K and cross-wave tests; no deadlock, no read-before-ready, no SMEM write-before-last-consumer, no store-before-kernel-completion. Edges differences only documented ones. |
| **P6 Validation + report** | Run official 8 GEMM + 6 FA3 cases through original fork builder and new IR runner on **identical fork config**; optionally run official snapshot under its original config separately. Emit CSV/JSON/Markdown report and provenance. | Per-case graph diff, cycles, APE, aggregate MAPE/Pearson/maxAPE; thresholds in §6; existing CTests and profile check pass; no regression masked by waived tests. |
**Commit policy:** P0 evidence separate; P1/P2 pure Python; P3 lowerer; P4 simulator adapter; P5 sync fix separately reviewable; P6 validation. Never combine synchronization fix and new IR implementation into one unexplained commit. The implementing agent should make code changes only on a new feature branch from `serve-sim`, not rewrite branch history or remove old files.
## 6. Validation protocol, thresholds, expected numbers
### 6.1 Three independent gates (do not conflate)
**Gate A — Algorithm/graph parity (highest priority).** Compare candidate IR→DAG with reference C++ builder *running on the same fork + same hardware config*. Compare canonical sorted records: op name/kind/pipeline, node count by type, all TensorRegions and strides, CTAs/SMs/warps, registers/shared bytes, output coverage, issue and complete dependencies (referenced by normalized node identities rather than raw ID alone). For known sync fixes, require a reviewed exact edge delta; **zero tolerance** for unapproved differences in arithmetic/memory traffic or node types. Apply **both** to GEMM and FA3.
**Gate B — Simulator timing parity.** Run old reference and IR candidate on the **same binary/backend/YAML** with unchanged memory placement and deterministic configuration. Expect **identical total cycles** when graph is identical; allow timing differences only if tied to reviewed synchronization changes, in which case present before/after and compare on a corrected reference. No extra invented startup latency. Ensure no per-kernel timing overrides.
**Gate C — External measured-data accuracy.** Use `validation/data/h100_gemm.csv` (8) and `h100_fa3.csv` (6); `APE_i=100*abs(sim_i-measured_i)/measured_i`; `MAPE=mean(APE_i)`, `maxAPE=max(APE_i)`; report Pearson. Compare new implementation against **the same-fork corrected baseline** first, and official public release as a separate historical anchor. For the user's alarm rule, flag and investigate if **MAPE increases >5% relative** (`MAPE_new > 1.05*MAPE_baseline`) **or maxAPE increases >10% relative** (`maxAPE_new > 1.10*maxAPE_baseline`). Also report absolute percentage-point changes to remove ambiguity. Do not accept a coincidentally lower MAPE when graph parity fails.
**If any gate fails:** stop declaring success; emit first diverging node/edge, affected K-stage and CTA, config/digest, and cycles; revisit lowering/sync/operation mapping; rerun complete matrix after fix. Preserve failures in report, do not hide them behind `--allow-measurement-error`.
### 6.2 Reference anchors, confirmed before coding
| Source/profile | GEMM M=512,N=512,K=512 | FA3 B=1,Hq=1,Hkv=1,Q=128,KV=128 | GEMM 8-case MAPE | FA3 6-case MAPE |
|---|---:|---:|---:|---:|
| Official public v0.1.0 | **12,956 cycles** | **7,168 cycles** | **2.65%** | **9.79%** |
| Fork `serve-sim` stored references | **12,249 cycles** | **7,212 cycles** | **5.70198% previously recorded** | **not established; rerun required** |
`validation/v0.1.0.md` in fork still displays the official MAPE figures unchanged: **do not treat them as a measurement of current fork.** Official paper's broad *1.22%–8.71%* accuracy range is not a substitute for the actual public suite above. Current fork changes also include new memory backend paths, making cross-repository cycle equality under different profiles inappropriate.
**Fork GEMM recorded 8-case table** (`docs/tvm-gemm-status.md`, not freshly executed):
| M,N,K | Measured | Fork simulated | APE |
|---|---:|---:|---:|
| 512,512,512 | 13143 | 12249 | 6.802% |
| 512,512,1024 | 24072 | 21473 | **10.797%** |
| 512,1024,512 | 13085 | 12919 | 1.269% |
| 512,1024,1024 | 23999 | 22143 | 7.734% |
| 1024,512,512 | 13077 | 12919 | 1.208% |
| 1024,512,1024 | 24046 | 22143 | 7.914% |
| 1024,1024,512 | 13296 | 14258 | 7.235% |
| 1024,1024,1024 | 24123 | 23482 | 2.657% |
### 6.3 Required test matrix
| Test class | Cases | Assertion |
|---|---|---|
| Reference smoke | GEMM 512³; FA3 (1,1,1,128,128) | Saved references on matching fork profile or documented sync-version delta |
| Measured validation | All official 8 GEMM + 6 FA3 shapes | Gate C with per-case CSV |
| GEMM buffer/sync | K=64,128,192,256; persistent multiwave with 2 SMs; odd/even K-slot phase; multiple output tiles | No hazards, correct complete/issue edges and CTA finalization |
| FA3 algorithm | Hq/Hkv=8/8,16/16, and supported GQA; Q/KV lengths 512,1024,1536 | All QK/softmax/PV/store stages preserved, exact reference morphology |
| Negative support | GEMM M=1; FA3 Q=1; split>1, BF16, D256, alternate CTA order | Reference profile rejects (legacy fork standalone may still support them) |
| Generic compiler | IR01/02/03/04/05 round-trip and determinism; invalid tensor alias, stream, unknown SFU op, barrier cycles | Verifiers produce location-rich failures; no implicit fallbacks |
| Integration | `gtsim-ir run gemm ...`; `gtsim-ir run fa3 ...`, each one stream/one invocation | Every stage emitted; runner performs real launch/complete; program timing matches single-DAG compatibility (zero launch overhead) under same initial state |
| Sequential program runtime | test-only `Stream0: GEMM → FA3 → StreamSynchronize` | Distinct launch/complete events; continuous clock; stable program TensorIDs, GPU/L2/memory state preserved; two kernels do not silently become one DAG or two new GPUs |
| Stream/event failures | empty stream; unknown kernel; wait-before-record; multiple streams; bad tensor binding | Deterministic valid empty-stream handling; reject unsupported multi-stream/cyclic or dangling events; location-rich errors |
| Existing regression | `ctest --test-dir build --output-on-failure`, `python3 scripts/check_workload_profile.py`, `python3 scripts/run_reference.py --build-dir build --verify` | Zero newly introduced failures; TVM tests optional as before; no extra L2 flush or reset in the single-DAG compatibility path |
### 6.4 Commands (target CLI contract, to be implemented)
```bash
# Existing reference executables / scripts (run first)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
python3 scripts/run_reference.py --build-dir build --verify
python3 scripts/run_validation.py --build-dir build --out-dir validation/results/ir-framework/legacy
# New frontend, no TVM import
python3 -m pip install -e ./frontend
# Both accept param overrides, only within reference-supported shapes
gtsim-ir run gemm --m 512 --n 512 --k 512 --dtype fp16 \
  --architecture configs/v0.1.0/h100-sxm.yaml \
  --memory-config configs/v0.1.0/memory_full_hbm-0F5M.yaml \
  --out-dir validation/results/ir-framework/gemm-512
gtsim-ir run fa3 --batch 1 --heads-q 8 --heads-kv 8 \
  --seq-q 512 --seq-kv 512 --dtype fp16 --head-dim 128 \
  --architecture configs/v0.1.0/h100-sxm.yaml \
  --memory-config configs/v0.1.0/memory_full_hbm-0F5M.yaml \
  --out-dir validation/results/ir-framework/fa3-512
# Sequential runtime semantics (new test binary; should NOT use independent Simulator instances)
ctest --test-dir build -R execution_program --output-on-failure
# Stage inspection and full parity/accuracy report
gtsim-ir inspect --input validation/results/ir-framework/gemm-512/manifest.json
gtsim-ir validate --oracle legacy --suite h100-v0.1.0 \
  --out-dir validation/results/ir-framework/validation
```
An offline machine may need the already-required `3rdparty/hbfine` submodule initialized to build this fork. The new Python frontend must **not** require `scripts/tvm/build_tvm.py`, `.tvm-venv`, CUDA SDK or `ptxas`.
### 6.5 Output/artifact manifest
For each run, store these files, deterministically named and versioned:
```text
<out-dir>/
├── run-spec.json            # git refs, yaml hashes, compiler version, shapes, profile
├── program.progir          # IR-01 (versioned JSON)
├── kernel.algoir            # IR-02 (versioned JSON)
├── kernel.flowir            # IR-03 + embedded ResourceMapping (versioned JSON)
├── kernel.nvir              # IR-04 (versioned JSON)
├── kernel.gtsir             # IR-05 kernel DAG (versioned JSON)
├── program.gtsir            # IR-05 execution program (versioned JSON)
├── program.txt              # short inspection dump, optional
├── lowering-manifest.json   # pass names, inputs/outputs digests, object counts
├── graph-summary.json       # node/op counts, CTA resources, regions, dep counts
├── simulation.log          # GTSim raw text
├── kernel-timings.json      # launch/start/complete cycle per invocation
└── metrics.json             # program cycles, per-tier traffic, outcome, validation status
```
CI output for the full suite must also include `validation-summary.json`, `gemm_cases.csv`, `fa3_cases.csv`, `graph-diffs/`, `failures/` and `report.md` (source/number provenance and threshold decisions). Generated multi-MB graphs/results stay under ignored `validation/results/ir-framework/`, while small static fixtures stay tracked under `tests/`.
## 7. Legacy TVM containment and compatibility
- **Do not delete or rename** `scripts/tvm`, `3rdparty/tvm`, `build-tvm` instructions, existing docs/tvm* or `tests/test_tvm*.py`; those are historical results and optional compatibility utilities.
- IR-01 parsing, validation, Program composition and IR-01-to-IR-05 pass logic stay in **`frontend/`**. C++ `src/runtime/` reads the *lowered IR-05 ExecutionProgram* and implements runtime command semantics; it is not a second Program IR compiler.
- `frontend/` may **not import** `tvm`, `scripts.tvm`, `h100_schedule.py`, `native_dag.py`, `reference_gemm_lower.py` or `manual_gemm.py` at runtime. A separate `tests/reference_parity/` utility may call legacy oracles without shipping that dependency to the production frontend.
- Preserve `GTSIM_TVM_PYTHON` as an **optional CMake cache parameter**, default unset. Add `GTSIM_BUILD_IR_FRAMEWORK` option (default ON) for the new runner; do not make core or old reference executables depend on Python.
- Keep C++ legacy `gtsim_tile_ir` v1/v2 reader and binaries operational. Register new `.gtsir` JSON parser/materializer separately, dispatched only by the new program runner using the `format` discriminator. Do not silently reinterpret old textual `.gtsir` files.
- If the GEMM sync defect requires patching `workloads/gemm/gemm_h100.cpp`, change only the minimal relevant synchronization behavior, record graph-diff rationale and cycle impact, update fork `references.json` **only after full verification**, preserving prior baseline numbers and measurements in a comparison report. Do not rewrite official data CSVs or copy official MAPE into fork post-fix documentation.
## 8. Definition of Done (agent verification checklist)
- [ ] Reference fork and official upstream commit/tree IDs recorded; 8 GEMM and 6 FA3 measured CSV hashes checked and identical.
- [ ] Existing build/CTest and legacy baseline executed first; exact hardware/memory YAML and backend provenance saved.
- [ ] IR-01/02/03/04/05 models, validators, deterministic JSON codecs and textual printers work without TVM.
- [ ] GEMM and FA3 each have a **real**, separately authored IR-02 template and full IR-03 Flow program, not a wrapper around a precomputed reference DAG.
- [ ] `IR-01 + IR-02 + fixed schedule → IR-03 → IR-04 → IR-05` emits `program.progir`, `kernel.algoir`, `kernel.flowir`, `kernel.nvir`, `kernel.gtsir` and `program.gtsir` with a verifiable lowering manifest.
- [ ] New IR-05 runner supports FA3's actual SIMD/SFU/SHFL op set and preserves legacy `.gtsir` v1/v2.
- [ ] Two requested **one-stream/one-kernel** top-level GEMM/FA3 programs run with legal variable parameters; unsupported fork-only features are rejected in the official parity profile.
- [ ] GTSim has **actual single-Stream sequential multi-launch semantics** (launch, completion, event, synchronize); multi-stream input is rejected; the new runner does not loop over independent `Simulator` instances.
- [ ] A test-only `GEMM→FA3` program passes with continuous global cycle, per-launch scheduler lifecycle, stable TensorRegistry, retained L2/MemorySystem state and no intermediate cache flush; deterministic launch-overhead accounting (default zero) is verified.
- [ ] Legacy single-DAG executables preserve their original timing and output format unless a separately reviewed synchronization correctness fix changes those results.
- [ ] GEMM sync issue independently diagnosed, minimal fix/test justified; official-vs-fork edge delta documented and no timing-constant workaround.
- [ ] Both new graphs match the independently built C++ reference morphology (or an exact reviewed sync-only edge delta), not just total cycles.
- [ ] All 8+6 measured validation cases and MAPE/maxAPE/Pearson compared against the **same fork build/profile**, with official v0.1.0 as a separate historical reference.
- [ ] If MAPE increase exceeds 5% relative or maxAPE exceeds 10% relative, the agent investigates, fixes and reruns; if unresolvable, marks validation FAIL and reports precise differences.
- [ ] All legacy CTest/profile checks pass; legacy TVM files preserved but unused by the new default path.
- [ ] Final report includes commands, commit/source/config hashes, pass/fail matrix, cycles per case, graph diffs, remaining limitations and concise reproduction instructions.
**Implementation success means a verified IR→DAG→GTSim path for two original H100 algorithms, not an assertion that a full Qwen decode step is already supported.**
## Sources inspected
- Fork `serve-sim`: `CMakeLists.txt`, `workloads/gemm/gemm_h100.cpp`, `workloads/fa3/fa3_h100.cpp`, `workloads/common/workload_options.h`, `src/tile_ir_parser.cpp`, `src/tile_dag_builder.cpp`, `include/tile_ir.h`, `workloads/gemm/gemm_ir.cpp`, `workloads/serve/serve_ir.cpp`, `tests/tile_ir_parity_tests.cpp`, `tests/test_manual_gemm.py`, `scripts/run_validation.py`, `configs/v0.1.0/references.json`, `docs/tvm-gemm-status.md`, `validation/{data,v0.1.0.md}`.
- Official `SJTU-ReArch-Group/GTSim` public main: original `workloads/{gemm,fa3}/*_h100.cpp`, `configs/v0.1.0/{references.json,h100-sxm.yaml}`, `validation/data/*.csv`, `validation/v0.1.0.md`, `scripts/run_validation.py`, README/release scope.
- Project uploads: `llm_serving_ir_design_v0_2.md`, `uniform-sim-ir-main.zip` (`flow/ir.py`, `flow/builder.py`, `flow/codec.py`, `mapping/ir.py`, `examples/gemm_double_buffer.py`, `pyproject.toml`); `gemm_fa3-param-sweep.tar.gz` contains additional *fork HBF/topology* sweeps, which are not substitutes for the official 8+6 reference validation.
