# GTSim Layered IR — Architecture Refactoring Checklist
**Target:** `Hypersonic-cpu/GTSim`, `serve-sim` (reviewed at `9031d1a`; confirm the current HEAD before editing).  
**Purpose:** Make IR-03/04/05 real, readable compiler layers **without changing** the validated Hopper GEMM/FA3 reference behavior.
## 0. Constraints and Definition of Success
- **Preserve:** the five-layer IR design, file extensions (`.progir`, `.algoir`, `.flowir`, `.nvir`, `.gtsir`), existing single-stream `ProgramExecutor`, persistent GPU/L2/memory state, old TVM files, and current reference GEMM/FA3 entrypoints.
- **No new functionality:** do not implement MoE, prefetching, automated DSE, numerical inference, multi-stream concurrency, or a general CUDA compiler in this refactor.
- **Correctness baseline:** the existing fork's *corrected* GEMM synchronization (`pingpong-cross-wave-v2`) and the original-reference FA3 algorithm. Do not restore the upstream GEMM dependency graph wholesale or activate fork-only FA3 extensions.
- **No hidden fallbacks:** a lowerer must either preserve a supported IR construct or reject it explicitly. Never ignore modified schedule or event dependencies merely because node names still match.
- **Mandatory regression gate:** all existing 8 GEMM + 6 FA3 validation cases must retain exact reference graph/cycle parity (except an independently justified, explicitly approved correctness fix). Keep measured-data MAPE/maxAPE unchanged within numerical reporting precision. Preserve the two-SM odd/even-K GEMM synchronization checks and sequential GEMM→FA3 runtime checks.
## P0 — Fix IR-03 → IR-04 Semantic Lowering (Highest Priority)
**Problem:** `frontend/src/gtsim_frontend/passes/lower_hopper_gemm.py` and `lower_hopper_fa3.py` infer execution structure from names such as `cta0.tile0.k0.A`, scan lists, and reconstruct hard-coded TMA/WGMMA/softmax dependency graphs. Actual `FlowOp` event/slot/executor dependencies are only partially consumed. This can silently discard a legal schedule transformation.
**Required changes:**
- Introduce **typed, explicit identities** for `TileId`, `IterationId`, `PipelineStage`, `BufferSlot/Generation`, and dependencies. Do not infer execution semantics by splitting `FlowOp.id` or matching `.A`, `.B`, `cta{k}.k{j}`.
- Define a small typed Flow operation set, e.g. `AsyncCopy`, `MatmulTile`, `ReduceMax`, `ReduceSum`, `Elementwise`, `StoreTile`, `Acquire`, `Await`, `Release`. Retain operand regions, logical executors, SSA values, result events and buffer ownership as first-class fields.
- Implement **shared operation-level lowering patterns**, for example `AsyncCopy → TMA`, `MatmulTile → WGMMA`, `ReduceMax → SIMD/SHFL`, `StoreTile → SMEM store/TMA store`, `Acquire/Await/Release → target synchronization`. Keep GEMM- and FA3-specific **algorithm schedules** separate; a dedicated algorithm pattern is acceptable, rebuilding a fixed DAG from operation names is not.
- Lower **actual dependency events and lifetime constraints** from IR-03, including buffer reuse, producer-ready/consumer-complete, inter-tile ordering and the corrected even-K cross-wave guard. Check that Hopper-specific synchronization remains legal.
- Move common WGMMA fragment/warpgroup expansion, TMA chunk selection, hardware resource calculations and barrier helpers into small reusable modules; do not duplicate them between GEMM and FA3.
- Reject missing/invalid `Mapping` bindings and unknown operations; do not silently use a default engine, layout or route.
**Tests that demonstrate genuine lowering:**
- Take a valid Flow schedule; change one **typed** event/dependency without renaming operations. Verify the downstream native program/dependency graph changes accordingly, or that the transformation is rejected as illegal.
- Test slot reassignment/depth, ready-vs-issue edges, output region changes and executor mappings. Ensure the lowerer honors them or fails with an actionable diagnostic.
- Verify every original GEMM TMA/WGMMA and FA3 QK–online-softmax–PV stage remains represented; graph morphology, resources and dependency parity against the independent C++ oracle remain unchanged for the reference schedules.
## P1-A — Keep IR-03 and IR-04 Structured; Expand Only in IR-05
**Problem:** the current reference `gemm.py` / `fa3.py` eagerly instantiate every CTA, tile and K/KV iteration into Flow objects; Hopper lowerers eagerly build every warp instruction again. `lower_execution.py` merely copies `ArchKernel` into `ExecKernel` (`dict(vars(arch))`). IR-04 is already effectively IR-05, creating multiple huge representations of the same work.
**Required design:**
- **IR-03 (`.flowir`):** keep compact scheduled **TileLoop / PipelineRegion** structures, logical operations, symbolic tile-region expressions, selected BM/BN/BK or Q/KV tiling, pipeline depth, buffer slots/generations, executor roles, and mapping. Iteration count and tail predicates must be explicit.
- **IR-04 (`.nvir`):** keep compact **Hopper-native structured loops/regions**, concrete instruction families (TMA, WGMMA, SIMD/SFU/SHFL), warpgroup assignment, shared/register layouts, native async/barrier/commit/wait semantics. It is a *native program*, **not** a flattened DAG; do not model fake PTX/SASS execution.
- **IR-05 (`.gtsir`):** perform **the only full unrolling** of CTA/wave/K loops and warp fragments; resolve native barrier phases, async issue/complete conditions and buffer-slot generations into explicit `DAGNode` operations, tensor regions, `after_issue` and `after_complete` edges. Existing GTSim timing execution remains unchanged.
- Make `passes/lower_execution.py` a real lowering pass rather than a dataclass copy. Its output should be functionally equivalent to the old flat graph for the original schedules.
- Do **not** introduce lazy DAG execution yet. Accept the large **final** DAG in this phase; avoid duplicating it at two earlier IR levels. Add IR size/node-count measurements to the regression report.
**Important reference behavior:** `workloads/gemm/gemm_h100.cpp` **also eagerly constructs one complete DAG**: nested persistent CTA → wave → K-tile → warp loops call `new DAGNode` via `add_node()`. It does **not** solve DAG size or use dynamic kernel loops. Its advantage is that it constructs the DAG **once**. This refactor should keep GTSim's existing execution model while avoiding repeated full expansion across IR-03, IR-04 and IR-05.
**Tests:** compare final DAGs (nodes, regions, CTA placement/resources, issue/completion edges) and cycles before/after; round-trip every newly introduced structured IR field; check large FA3 cases no longer serialize a full warp-level graph in `.flowir` and `.nvir`.
## P1-B — Separate Per-Kernel Runtime State from Persistent GPU State
**Problem:** `src/runtime/program_executor.cpp` already runs sequential kernels using one GPU, which is correct for v0. However, `Simulator::bind_kernel` in `src/simulator.cpp` directly resets many SM fields, replaces schedulers and rebinds scoreboards. Launch-local state and persistent device state are tightly coupled.
**Required changes (minimal, no runtime redesign):**
- Factor per-launch admission/scheduler/scoreboard initialization and retirement into a focused `KernelExecutionSession` / `KernelDispatcher` abstraction (exact names flexible). Keep shared GPU/MemorySystem/L2 and global TensorRegistry in a persistent context. Prefer extracting helpers from the existing `Simulator` over introducing a second execution engine.
- Make lifecycle invariants explicit: `CREATED → QUEUED → RUNNING → COMPLETED`, required output visibility, no pending launch-local operations on retirement, and a monotonically increasing global clock.
- Ensure kernel rebind supports **different subsequent CTA counts/warp-ID ranges and resource footprints**, not only the already-tested GEMM→FA3 pair with matching reference profiles.
- Keep one ordered stream only, with real launch/complete/event/synchronize semantics. Reject multiple streams; do not silently simulate concurrency.
- For current v0, the conservative complete-memory-drain boundary is acceptable. **Document it as a limitation**, not a future cross-kernel async-prefetch protocol. Preserve L2 lines and memory-controller state; do not flush caches between sequential kernels. Explicit launch overhead stays configurable (default zero for parity).
**Tests:** existing mixed GEMM→FA3 and warm-L2 tests; also run FA3→GEMM and repeated kernels with different CTA/warp counts; check stable addresses, cache retention, event clocks, per-kernel/device cycles and no scheduling-state leakage.
## P2-A — Strengthen IR-02 Semantics Without Building a General Compiler
**Problem:** `ir/algorithm.py` encodes algorithms mostly as immutable strings (`stages` and `computation`) and checks exact string equality. This documents formulas but offers little structured semantics for later transformations.
**Required changes:**
- Replace free-form formula strings *as the authoritative semantics* with a minimal **typed algorithm contract**: operand roles, shapes/index domains, reduction axes, math-operation kinds, initialization, output dtype and validity constraints.
- GEMM: typed matmul/reduction, accumulator initialization and FP16 output conversion. FA3: typed QK, streaming max/sum recurrence, PV, normalization and GQA head mapping.
- Keep exactly the two known algorithm families; **do not** implement a general expression optimizer, symbolic algebra system, or automatic algorithm discovery.
- Keep readable formulas as a **printer/documentation output**, not as the only contract lowerers understand.
**Tests:** invalid dimensions/dtypes, invalid GQA, missing reductions or outputs rejected; original instantiated reference schedule still produces identical final graphs and timing.
## P2-B — Improve Maintainability and Formatting
**Problem:** compressed one-line Python statements, large multi-purpose lowerers, local `emit()` closures with shared mutable arrays, hard-coded constants and verification logic mixed into model classes make the implementation unnecessarily difficult to review.
**Required changes:**
- Split large GEMM/FA3 lowerers into named functions for tile iteration, operand loads, matrix instruction selection, softmax stages, synchronization, epilogue/stores and native-kernel construction. Prefer explicit intermediate records/results over nested closures mutating one global `ops` list.
- Centralize reference profile constants (BM/BN/BK, Q/KV tile, stage count, warpgroup shape, buffer/resource sizes) in typed schedule/target profiles. Keep numerical timing parameters **only** in GTSim hardware config, never in the compiler.
- Separate Flow verification into small checks for SSA/event dominance, tensor regions/coverage, buffer-slot lifetime, mapping capability and output identity. Diagnostics should identify the offending operation/region.
- Preserve module boundaries: `ir/` defines data; `kernels/.../reference/` builds scheduled Flow; `passes/` lowers; `src/execution_ir/` materializes; `src/runtime/` executes. Keep legacy TVM entirely isolated and unchanged unless required by an independent regression fix.
- **Do not manually reformat source files. Invoke formatters.** Use the repository's existing style configs when present; otherwise add minimal pinned formatter config/developer dependencies:
  - Python: `ruff format frontend/src/gtsim_frontend tests/ir_frontend tests/reference_parity` and `ruff format --check ...` (plus `ruff check ...` after agreeing on enabled rules).
  - C++: `clang-format -i` on **changed** `.cpp/.h` files, followed by `clang-format --dry-run --Werror` on those files. Do not reformat unrelated legacy code.
  - If Markdown/JSON is reformatted, use a suitable formatter (e.g. Prettier for Markdown) or the project's deterministic JSON serializer; avoid manual alignment chores.
- Record formatter/tool versions and enforce formatting checks in CI. Formatting-only commits should be separate from semantic changes for readable diffs.
## Execution Order and Acceptance
1. **Freeze evidence:** record current `serve-sim` SHA and existing graph/cycle/MAPE results; preserve the GEMM sync audit and 14-case oracle tools. Run existing tests before modifications.
2. **P0:** implement typed semantics and dependency-preserving IR-03→04 lowering, including targeted mutation tests.
3. **P1-A:** introduce structured IR-03/04 and move full expansion plus dependency legalization to IR-04→05; preserve exact original reference DAGs.
4. **P1-B:** extract launch-local state management with stable persistent GPU ownership; run mixed-kernel tests.
5. **P2:** strengthen typed Algorithm IR, split large methods, run formatters and add formatting CI checks.
6. **Final validation:** run complete 8 GEMM + 6 FA3 graph and cycle parity, two-SM K=64/128/192/256 sync audit, sequential-kernel and cache-retention tests, existing CTest and optional legacy tests where available. Compare measured-data MAPE/maxAPE with the frozen same-fork baseline; if MAPE worsens by >5% relative or maxAPE by >10% relative, investigate and do **not** label the refactor PASS.
**Definition of done:** compact, interpretable `.flowir` and `.nvir`; a genuine `.nvir → .gtsir` expansion pass; lowerers driven by typed Flow operations/events rather than string naming; a maintained sequential Program runtime; all prior **reference** graph/timing behavior preserved; automatically formatted code. Provide before/after file sizes, node counts, test output, any intentional deviations, and reproduction commands.
