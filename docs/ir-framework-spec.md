# Layered IR wire contracts (v0.1)

The design narrative is [the serving IR spec](serving-ir/llm_serving_ir_design_v0_2.md).
This document records the implemented interchange. JSON is authoritative;
inspection text is deterministic and intentionally has no parser.

| Layer                    | File             | `format`               | Top-level `record` |
| ------------------------ | ---------------- | ---------------------- | ------------------ |
| Program                  | `program.progir` | `prog-ir/0.1`          | `Program`          |
| Algorithm                | `kernel.algoir`  | `algo-ir/0.2`          | `Algorithm`        |
| Scheduled Flow + Mapping | `kernel.flowir`  | `flow-ir/0.2`          | `FlowKernel`       |
| Hopper native            | `kernel.nvir`    | `arch-ir/0.2`          | `ArchKernel`       |
| Execution kernel         | `kernel.gtsir`   | `gtsim.exec.kernel/1`  | `ExecKernel`       |
| Execution program        | `program.gtsir`  | `gtsim.exec.program/1` | `ExecProgram`      |

Every top-level record contains `schema_version`, `symbol` and `source_hash`.
Algorithm, Flow and native records use schema version 2; Program and Execution
records retain version 1. The Program envelope now embeds the version-2 Algorithm
contract. Regenerate older `.progir` / `.algoir` / `.flowir` / `.nvir` artifacts;
legacy text and version-1 Execution JSON are unchanged.
Nested records carry a required `record` discriminator. Unknown records, fields,
versions and unsupported operations fail. Hashes are SHA256 of canonical,
sorted-key JSON including the final newline. Source hashes identify preceding
IRs; they never affect timing. `lowering-manifest.json` records input/output
hashes, hardware profile identity and invocation-to-graph paths.

Program tensors have symbolic identities, concrete shapes, dtype, tier and
read-only status. Invocations bind named operands to those identities. The
Algorithm record contains explicit stages and mathematical contracts, including
initialization, reduction coverage, GQA and online-softmax recurrence; numerical
execution is false. Reference schedules are explicitly selected, not searched.

Flow contains typed SSA values/producers, symbolic tensor regions, logical
executors, bounded tile/pipeline loops, slot generations and phased dependencies.
Mapping binds executor capabilities, buffers, routes and placement policy.
Reference-profile verification checks domains, output coverage, SSA/event dominance
and buffer lifetime before target selection.

Native kernels contain compact operation patterns and structured native loops,
with shared/register layouts, resource footprints, TMA chunk selection, four-warp
WGMMA collectives and async/barrier/commit/wait protocol. They do not contain a
per-CTA or per-warp flat DAG. Only the Execution kernel contains final CTA records,
explicit output IDs and instructions with `pipeline`, `op`, `cta`, `warp`, `tensor`,
`tile`, `dtype`, `leading_dim`, `after_issue` and `after_complete`. The execution
expander resolves phased events and slot generations without adding synthetic
latency. `sync_semantics_version` retains corrected GEMM provenance.

Execution programs contain exactly one ordered stream, program-global flattened
tensor allocations, launch graph paths, local-to-global integer tensor bindings,
ordered commands and `launch_overhead_cycles` (default zero). They contain no
Algorithm or Flow objects. FA3's logical rank-four tensors become stable,
head-specific two-dimensional allocations named from their Program identities.
Reusing the same logical head in another launch reuses its allocation.

Commands are `LaunchKernel`, `RecordEvent`, `WaitEvent`, `StreamSynchronize`.
Launch IDs are unique; wait-before-record, dangling bindings, output/input
allocation aliases and unsupported streams fail before simulation. Program-global
allocations remain alive for the entire program. Physical addresses are allocated
only by GTSim's TensorRegistry; they are never serialized by the frontend.

The runtime uses one persistent Simulator/GPU context. `bind_kernel` replaces
launch-local schedulers/scoreboards after completion, retaining GPU/SM objects,
L2 contents, memory controllers and global addresses. Completion waits for all
DAG nodes and outstanding memory requests, without flushing dirty L2 lines.
The global clock advances through memory drain. Per-launch device cycles and
program-total completion cycles are reported separately. Launch overhead advances
idle memory time without scheduling kernel instructions.

Legacy textual `gtsim.tile_ir 1/2` remains on the original parser/executables.
Only `gtsim_program` accepts the new JSON formats.

## Structured compiler representation

`AlgorithmContract` stores typed operand/index domains, mathematical operations,
initializers, reduction axes, result axes/dtypes, QK scaling and the GQA mapping.
Readable formulas are derived inspection output, not authoritative input strings.

Flow has a bounded `ScheduleDomain` with CTA/wave loops, a `PipelineRegion`,
explicit full-tile predicates and a last-wave `tile_id < tile_count` predicate.
The loop body contains six GEMM or 29 FA3 typed operations independent of shape.
`TileId`, `IterationId`, `PipelineStage`, `SlotUse`, `ValueId`, `SSAValue` and
phased `Dependency` records carry identity, regions, generation counts, loop-carried
initialization, producer/value dependencies and logical executors. Display names
can change without changing the resulting graph.

Native IR retains these loops and selects operation-level `NativePattern` /
`NativeStep` records: TMA chunk geometry, WGMMA fragment/warp loops, actual
SIMD/SFU/SHFL families, shared/register layouts and an explicit async protocol.
`passes/hopper/patterns.py` owns target selection; shared expansion helpers own
instruction-loop iteration, TMA splitting and phased event resolution.

`passes/lower_execution.py` is the only full unrolling boundary. It expands CTA,
wave, pipeline and warp-fragment loops, resolves forward issue events and protected
buffer generations, and constructs one final `ExecKernel`. A large final DAG is
still expected; there is no lazy DAG execution.

The supported subset is deliberately the two fixed reference schedules. Typed
schedule, dependency, slot, region, mapping or protocol transformations outside
that subset are rejected with a location-bearing diagnostic. They are never
silently replaced with inferred defaults. In particular, this refactor does not
claim support for arbitrary transformed schedules merely because their display
names resemble a reference kernel.
