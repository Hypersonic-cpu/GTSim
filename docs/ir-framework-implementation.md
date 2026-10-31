# Layered IR implementation

The implementation contract remains
[the supplied plan](serving-ir/GTSim_Layered_IR_Implementation_Plan_v0.1.md).
Implemented formats are documented in [the wire specification](ir-framework-spec.md);
commands and supported shapes are in [compatibility and reproduction](ir-framework-compat.md).

- P0: pinned source/config/data hashes, original CTest/reference runs and full
  8+6 measured baseline in `serving-ir/baseline/`.
- P1/P2: immutable typed IR records, strict deterministic JSON codec, inspection
  printers, mathematical templates, and independently authored logical reference
  Flow schedules. No vendor code was copied from Uniform.
- P3: independent native Hopper expansion, resources, four-warp collectives,
  TMA chunk geometry, softmax operations and synchronization dependencies.
- P4: generic execution serialization/materialization and real ordered-stream
  runtime on one persistent GPU context; launch/event/synchronize commands,
  continuous clocks and stable program allocations. Existing single-DAG
  Simulator behavior and legacy textual parsers remain available.
- P5: the minimal GEMM even-K cross-wave buffer-1 correction, committed separately
  before using the fork as the new graph oracle. See
  [the synchronization report](serving-ir/gemm-sync-fix.md).
- P6: independent graph/cycle comparison and 14 measured cases, plus actual
  mixed GEMM→FA3, shared-allocation warm-cache, overhead and rejection tests.
  See the final small reports under `serving-ir/results/`.

The persistent context uses the existing Simulator's GPU ownership rather than
a second GPUContext wrapper. `Simulator::bind_kernel` reinitializes only
per-launch scheduling/residency state; ProgramExecutor retains ownership and
all materialized DAG/tensor lifetimes. The GPU, SM pipelines, L2 and MemorySystem
are not reconstructed at kernel boundaries. Cache retention is tested using
resident line counts and reduced HBM read traffic, not only pointer equality.

Reference builders are linked only into test-oracle utilities. Production
lowerers do not invoke them, import TVM code, or load precomputed DAG fixtures.
A separate pinned old-builder comparison uses the current simulator backend
for both versions, making synchronization cycle deltas attributable to the
builder rather than a different memory model.

Generated graphs, stage JSON, raw logs, hashes and detailed per-case reports
remain under ignored `validation/results/ir-framework/`. Small CSV/JSON/Markdown
summaries are tracked. No simulation figures are presented as physical GPU
measurements or numerical execution proof.

## Structured refactor

The architecture follow-up is implemented in the structured schema revision.
Reference loop bodies are in `kernels/hopper/reference/templates.py`; typed
geometry is centralized in `ir/profiles.py`. Flow verification is split between
domain, tensor/coverage, mapping, SSA/event and lifetime checks under `verify/`.
Native pattern verification checks instruction families, resource/layout records,
collectives and async protocol. IR models contain data and delegate verification.

The former flattened native DAG model is now the execution graph model in
`ir/graph.py`. Full expansion lives under `passes/hopper/expand_*` and is invoked
only by `lower_execution.py`. Early IR size is independent of CTA/iteration/warp
counts apart from scalar loop bounds and tensor dimensions.

`KernelExecutionSession` now owns initialization/retirement/rebind procedures for
launch-local scheduling, scoreboards and admission. Both the legacy Simulator
constructor and subsequent launches use that initialization seam. Simulator still
owns the persistent GPU and clock; no second timing engine was introduced. Session
retirement rejects pending pipeline, TMA, SRAM, retry or remote work. Tests cover
both kernel orders, growing/shrinking CTA/warp ranges and varying IR-05 resources.

**Runtime limitation:** every launch conservatively waits for the full memory
system to drain. L2 lines and memory-controller state persist, but this is not a
cross-kernel asynchronous prefetch or overlap protocol.

The refactor report and before/after sizes are in `serving-ir/refactor-results/`.
Formatting-only commits precede semantic changes; final checks use Ruff 0.14.4
and clang-format 23.1.3, pinned in the developer extra and formatting CI workflow.
