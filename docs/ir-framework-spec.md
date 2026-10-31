# Layered IR wire contracts (v0.1)

The design narrative is [the serving IR spec](serving-ir/llm_serving_ir_design_v0_2.md).
This document records the implemented interchange. JSON is authoritative;
inspection text is deterministic and intentionally has no parser.

| Layer | File | `format` | Top-level `record` |
|---|---|---|---|
| Program | `program.progir` | `prog-ir/0.1` | `Program` |
| Algorithm | `kernel.algoir` | `algo-ir/0.1` | `Algorithm` |
| Scheduled Flow + Mapping | `kernel.flowir` | `flow-ir/0.1` | `FlowKernel` |
| Hopper native | `kernel.nvir` | `arch-ir/0.1` | `ArchKernel` |
| Execution kernel | `kernel.gtsir` | `gtsim.exec.kernel/1` | `ExecKernel` |
| Execution program | `program.gtsir` | `gtsim.exec.program/1` | `ExecProgram` |

Every top-level record contains `schema_version: 1`, `symbol` and `source_hash`.
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

Flow contains SSA `TileValue` producers, tensor regions, logical executors,
per-CTA buffers, slot generations and phased events. `mapping` binds executors,
buffers, routes and CTA/SM placement. Producer readiness, consumer completion,
slot ownership and complete/non-overlapping output coverage are verified.
The Flow schema is independently implemented and Uniform-inspired; it does not
claim binary interchange with the external Uniform v1.0 package.

Native kernels contain logical tensor metadata, CTA resources, per-warp native
instructions and explicit output tensor IDs. Instructions contain `pipeline`,
`op`, `cta`, `warp`, `tensor`, `tile`, `dtype`, `leading_dim`, `after_issue` and
`after_complete`. WGMMA collectives contain all four warp fragments. TMA chunks,
softmax SIMD/SFU/shuffle operations and output staging are separately represented.
Synchronization events become explicit dependency edges, not synthetic latency.
`sync_semantics_version` distinguishes corrected GEMM from the prior fork graph.

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
