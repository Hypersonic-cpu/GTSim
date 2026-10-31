# LLM Serving Simulation: Layered IR Design
**Status:** Design specification (draft)  
**Version:** 0.2  
**Scope:** Baseline decode-step simulation on Ampere, Hopper, and Blackwell targets  
**Reference:** `uniform-sim-ir-main.zip` (Flow IR v1.0, Mapping IR, Hardware IR, Sim Plan) and GTSim `serve-sim` branch.
## 1. Design Boundary
The framework translates a **decode step** into an executable simulation program. It is not a general-purpose tensor compiler and does not attempt to discover GPU algorithms from scalar mathematics. Kernel algorithm families and baseline schedules are explicitly provided. GTSim remains the single timing backend.
```text
Serving frontend / trace
  └─ DecodeStepContext (batch, per-request KV lengths, routing, model metadata)
       │
       ▼
IR-01  Program IR ─────────────── program-level launch/order/data dependencies
       │ references KernelInvocation(template, arguments)
       ├─────────────────────┐
       │                     ▼
       │               IR-02  Algorithm Template IR ── kernel semantics + implementation family
       │                     │
       │                     ▼  instantiate + select baseline schedule
       │               IR-03  Scheduled Flow IR ─────── Uniform-style FlowKernel
       │                     + ResourceMapping ──────── Uniform-style MappingProgram (adjunct)
       │                     │
       │                     ▼  target legalization / instruction selection
       │               IR-04  ArchNative IR ─────────── target-specific instructions + synchronization
       │                     │
       └─────────────────────┤  lower kernel operations and program events
                             ▼
                       IR-05  Execution Program + Kernel DAGs
                             │
                             ▼  materialize program + per-kernel DAGs
                       GTSim ProgramExecutor
                       ├─ single ordered Stream (v0)
                       ├─ launch / complete / sync semantics
                       └─ persistent GPU Context + global cycle
                             │
                             ▼  per-launch schedulers; shared GPU/memory state
                       GTSim timing backend
```
**Interpretation:** IR-01 and IR-02 are complementary inputs, not a linear sequence in which IR-01 becomes IR-02. IR-01 specifies *which kernel instance runs and when*; IR-02 defines *what that kernel instance means*. Their instantiation produces IR-03 kernels associated with the IR-01 invocations. IR-04 lowers each kernel, and IR-05 composes the resulting kernel DAGs with cross-kernel execution constraints.
The examples below follow Uniform IR's textual conventions: `dialect.operation`, `@symbol` for named objects, `%value` for SSA values, `!event` for events, typed regions, and explicit capabilities/work. **New `prog.*`, `algo.*`, and `arch.*` forms are proposed inspection syntax**, not parsers or APIs already implemented in Uniform or GTSim. `flow.*`, `map.*`, and `sim.*` follow their existing conceptual structure; example operations are abbreviated for readability.
## 2. IR Overview
| Layer | Primary unit | Records | Deliberately excludes | Existing reuse |
|---|---|---|---|---|
| IR-01 Program (`.progir`) | `prog.program` / invocation | streams, launches, tensor identities, events, ordering, dependencies | tiled loops, warp assignment, machine instructions | new lightweight frontend |
| IR-02 Algorithm Template (`.algoir`) | `algo.kernel` | computation semantics, index domains, known stages and implementation family | chosen tile sizes, placement, actual pipeline depth | adapters around GEMM/FA3/etc. |
| IR-03 Scheduled Flow (`.flowir`) | `flow.kernel` + mapping adjunct | tile iteration, executor roles, buffer slots, operations, lifecycle, ordering, resource bindings | concrete PTX/SASS, cycle timings | Uniform Flow IR v1.0, selected Mapping IR structures |
| IR-04 ArchNative (`.nvir`) | `arch.kernel` | target-specific op kinds, layouts, synchronization, resource footprints | runtime cycle outcome, memory queue scheduling | GTSim Hopper-model semantics; new thin target lowerers |
| IR-05 Execution (`.gtsir`) | `exec.program` + kernel DAG | global tensors, ordered stream commands, launch/completion/sync boundaries, CTA/warp ops, issue/complete dependencies | compiler scheduling choices, original tensor algebra | GTSim `DAGNode`, Tile IR, new ProgramExecutor |
| Hardware description | external profile | memory hierarchy, engine properties, capacities, bandwidths, queueing | workload semantics | GTSim configuration; Uniform Hardware IR as design reference |
**Baseline invariant:** each layer retains the mathematical outputs, tensor identities, required visibility ordering, and logical invocation boundaries unless a separately identified kernel fusion/replacement is explicitly selected.
## 3. Serving Frontend (Outside the Five IRs)
The serving simulator or trace reader produces one `DecodeStepContext`. It need not provide instruction timing or a GPU kernel schedule. A request may have a different valid KV length; a MoE model may require a recorded or runtime-provided expert assignment.
```yaml
context_version: decode-context/1
model: qwen_moe_example
batch:
  active_requests: [r0, r1, r2, r3]
  query_tokens_per_request: 1
  kv_valid_lengths: [4096, 8192, 2048, 16384]
routing:
  mode: trace                # or runtime_model; not necessarily known before router executes
  trace_ref: routing_trace_001
weights: model_tensor_catalog
memory_profile: h100_full_hbm
```
**Lowering:** `DecodeStepContext + model layer catalog + selected kernel implementation catalog → IR-01`. The frontend expands model layers and logical tensor shapes, resolves per-request lengths, and refers to the selected operator implementations. It does **not** generate tile loops. Routing is consumed only at the point at which its information is legitimately available; offline trace-driven timing can supply ground truth without implying a predictor knows future expert IDs.
## 4. IR-01 — Program IR
### 4.1 Semantic Contract
A Program is a graph of kernel invocations, tensor versions, stream-command sequences, and events. A `KernelInvocation` references an `AlgorithmTemplate` and one concrete implementation selection; multiple invocations may share that template. A tensor name is program-global, not local to a kernel DAG.
**Principal objects:** `Program`, `TensorDecl`, `KernelInvocation`, `Stream`, `Event`, `Dependency`, `RuntimeParam`, `KernelOutput`.
### 4.2 Textual Form
```mlir
prog.program @decode_layer_0 params(%B: index, %kv_lengths: vector<index>) {
  prog.tensor @hidden        : tensor<?x2048xbf16> storage<HBM>
  prog.tensor @kv_cache      : tensor<?x?x?xbf16> storage<HBM> external
  prog.tensor @expert_weight : tensor<?x?x?xbf16> storage<HBM> external
  prog.stream @compute ordered {
    %norm, !e0 = prog.launch @input_norm
        callee<@rmsnorm> inputs[@hidden] outputs[@norm]
    %qkv, !e1 = prog.launch @qkv_projection
        callee<@gemm> inputs[%norm, @qkv_weight] outputs[@qkv]
        after[!e0]
    !e2 = prog.launch @kv_update
        callee<@kv_cache_update> inputs[@qkv, @kv_cache] after[!e1]
    %attn, !e3 = prog.launch @attention_decode
        callee<@online_attention> inputs[@qkv, @kv_cache]
        attrs{kv_lengths=%kv_lengths, query_tokens=1} after[!e2]
    %expert_ids, %scores, !e4 = prog.launch @router
        callee<@router_topk> inputs[%attn] after[!e3]
    %out, !e5 = prog.launch @moe
        callee<@grouped_gemm> inputs[%attn, %expert_ids, @expert_weight]
        after[!e4]
  }
  prog.return %out after[!e5]
}
```
This is a **structural example**, not a claim that it is a complete Qwen Transformer block or that each logical stage must be a separate CUDA launch. A real program also represents output projection, residuals, norms, activation/gating and MoE combination according to the selected model and actual kernel fusion.
### 4.3 Rules
- `prog.launch` is a **kernel invocation**, not an inlined copy of the kernel's operations.
- A stream gives an ordered submission relation; producer/consumer tensor edges and explicit `after` events define required visibility. A model-level data dependency must never be lost when kernel boundaries change.
- The Program owns global tensor identity, shape, dtype and logical aliases; a kernel owns only its parameter views. Initial storage is a separate property from later temporary buffers.
- Logical runtime dimensions may remain symbolic in IR-01, but every invocation must be fully instantiated or explicitly supported by runtime binding before GTSim executes it.
- Stream synchronization is a semantic barrier, not zero-cost bookkeeping unless the target simulator explicitly models it as such.
- **v0 execution subset:** represent a single ordered stream with sequential `launch`, `record_event`, `wait_event`, and `synchronize` commands. A second stream is rejected as an unsupported backend capability, not silently serialized. The IR schema remains extensible to concurrent streams; this version makes no overlap claim.
- A kernel's `launch` and `complete` are distinct events. Completion requires all output writes needed by its successors to be visible under the memory model; it does not require invalidating GPU caches or resetting memory-controller state.
**Lowering contract:** `IR-01 + IR-02 + schedule configuration → instantiated IR-03 kernels + retained Program skeleton`. Per-invocation arguments define the kernel's tensor regions and runtime dimensions. Program ordering is carried forward to IR-05 rather than being hidden inside IR-03.
## 5. IR-02 — Algorithm Template IR
### 5.1 Semantic Contract
An Algorithm Template states both the **mathematical computation** and a **known implementation family**. It defines logical iteration domains, valid input/output regions, algorithmic stage structure and correctness constraints. It does not ask the compiler to invent online softmax, blocked GEMM, or pipelining from scalar operations.
**Principal objects:** `AlgorithmTemplate`, `Domain`, `TensorAccess`, `Stage`, `ImplementationFamily`, `Invariant`, `Parameter`.
### 5.2 Textual Form: Pipelined GEMM
```mlir
algo.kernel @gemm(%A: tensor<MxKxbf16>, %B: tensor<KxNxbf16>)
    -> (%C: tensor<MxNxbf16>) {
  algo.domain %m in [0, M), %n in [0, N), %k in [0, K)
  algo.compute @matmul
      result<%C[%m,%n]>
      semantics<sum_k(cast_f32(%A[%m,%k]) * cast_f32(%B[%k,%n]))>
  algo.stage @operand_acquisition reads[%A, %B]
  algo.stage @matrix_accumulate after[@operand_acquisition] produces[%C]
  algo.stage @epilogue after[@matrix_accumulate] writes[%C]
  algo.family <tiled_matrix, producer_consumer_pipeline>
  algo.require <same_result, legal_tail_mask, reduction_complete>
}
```
`algo.family` expresses an available implementation pattern; it is **not** a concrete pipeline schedule. In particular, `algo.stage @operand_acquisition` does not decide whether the input is read directly from HBM/HBF, staged through shared memory, or fetched by a particular architecture instruction.
Additional families may be separately defined for `online_attention`, `split_kv_attention`, `grouped_expert_gemm`, `rmsnorm`, `topk`, `silu`, `gather` and `scatter`. Routing and grouped experts require logical token→expert assignments, not only an ordinary GEMM shape.
**Lowering contract:** `AlgorithmTemplate + invocation shapes + chosen baseline schedule → FlowKernel`. Lowering expands known algorithm stages into legal tiled dataflow. All shape/tail handling and output-region coverage must be explicit. The compiler is allowed to instantiate supported templates; it does not invent new mathematical algorithms.
## 6. IR-03 — Scheduled Flow IR with ResourceMapping
### 6.1 Position Relative to Uniform IR
IR-03 should be close to **Uniform Flow IR v1.0**, whose real primitives include `TensorRegion`, `TileValue`, `Executor`, `Buffer`, `SlotHandle`, `Event`, `Operation`, `PipelineRegion`, and `ScheduleRegion`. In Uniform, Flow records logical dataflow and software scheduling while Mapping separately supplies physical bindings. This design preserves those **semantics** but treats the corresponding MappingProgram as an **adjunct to IR-03**, not an additional numbered layer.
Two serialized objects may still be useful:
- `flow.kernel`: logical scheduled program; no physical engine IDs or memory performance parameters.
- `mapping.program`: its target-specific bindings, including `map.place`, `map.execute`, `map.store`, `map.tensor`, and `map.route`.
**Important:** a Uniform `MappingProgram` is **not** the pass that discovers GEMM tile sizes or pipeline depth. Those are already selected when constructing the `FlowKernel`. The project uses fixed baseline schedules initially; it does not require a DSE engine in this version.
### 6.2 Flow Textual Form
```mlir
flow.kernel @gemm_bf16_128x256x64 {
  flow.symbol %M range[1, *]
  flow.symbol %N range[1, *]
  flow.symbol %K range[1, *]
  flow.tensor @A : tensor<%Mx%Kxbf16, row_major>
  flow.tensor @B : tensor<%Kx%Nxbf16, row_major>
  flow.tensor @C : tensor<%Mx%Nxbf16, row_major>
  flow.executor %producer role<producer> count(1)
      organization<independent> progress<ordered> lifetime<per_instance>
  flow.executor %matrix role<matrix> count(4)
      organization<cooperative> progress<dynamic_ready> lifetime<per_instance>
  flow.buffer %a_stage slots(2) access<read_write>
      : buffer<128x64xbf16, staging>
  flow.buffer %b_stage slots(2) access<read_write>
      : buffer<64x256xbf16, staging>
  flow.pipeline @k_loop %k = 0 to ceildiv(%K,64) step 1
      depth(2) instantiate<eager> {
    // An actual Uniform Flow v1.0 dump expands these into flow.iteration blocks.
    %a_slot, %b_slot, !acquired = flow.acquire @acq
        slots[%a_stage.s(%k mod 2), %b_stage.s(%k mod 2)]
    %a, !a_ready:data_ready = flow.load_async @load_a
        @A[0, %k*64]<128x64> to %a_slot by %producer
        requires[bulk_copy] work{bytes=16384}
    %b, !b_ready:data_ready = flow.load_async @load_b
        @B[%k*64,0]<64x256> to %b_slot by %producer
        requires[bulk_copy] work{bytes=32768}
    !inputs_ready = flow.join @input_join after[!a_ready, !b_ready]
    %acc, !mma_done:completed = flow.matmul @mma
        %a, %b, %acc_prev after !inputs_ready by %matrix
        requires[matrix.mma] work{flops=2097152}
    !released = flow.release @release
        after !mma_done release[%a_slot, %b_slot]
  }
  %c, !epilogue_done = flow.compute @epilogue %acc
  !stored = flow.store_async @store_c %c to @C[0,0]<128x256>
}
```
This is **proposed illustrative text in Uniform's printing style**, not an exact `FlowPrinter` output: the current Uniform v1.0 reference builder explicitly materializes pipeline iterations and uses concrete `SlotHandle` generations. The apparent `%acc_prev` is a loop-carried accumulator value whose concrete versions must be represented consistently (`flow.carry` where required). Real kernel generation must split full tensor dimensions into CTA tiles and include boundary predicates; the example focuses on one output tile.
### 6.3 Mapping Adjunct
The corresponding mapping should use Uniform's `mapping.program` representation rather than embedding physical memory names in `flow.buffer` types:
```mlir
mapping.program @gemm_sm90 flow<@gemm_bf16_128x256x64> hardware<@h100_profile> {
  map.place @cta_placement %cta -> @SM[*]
      resolve_at<launch> relation<one_to_one> lifetime<instance>
      mobility<fixed> requires{warps=12}
  map.execute @matrix_bind %matrix capability<matrix.mma>
      -> @H100_WGMMA
  map.execute @copy_bind %producer capability<bulk_copy>
      -> @H100_TMA
  map.store @a_storage %a_stage -> @GPU_SMEM allocation<per_instance>
  map.store @b_storage %b_stage -> @GPU_SMEM allocation<per_instance>
  map.tensor @a_tier @A -> @HBM
  map.tensor @b_tier @B -> @HBF
  map.tensor @c_tier @C -> @HBM
  map.route @input_route capability<bulk_copy>
      @HBM -> @GPU_SMEM via<@tma_route>
}
```
The printed structure follows Uniform's Mapping IR, but the named hardware resources are **hypothetical profile names**, not claims that these particular identifiers exist in the uploaded package. A real Mapping must bind **each supported source memory and operation** to a valid target route; if a target does not support `HBF → GPU_SMEM` directly, the verifier must reject that mapping or the input program must explicitly represent a supported intermediate transfer. Capacity and bandwidth are **hardware-profile properties**, not numbers injected into the Flow operation.
### 6.4 Verification
- `FlowVerifier`: valid tensor regions, value producers, event dependencies, pipeline semantics, buffer-slot ownership/reuse, operation capabilities, acyclic dependencies.
- `MappingVerifier`: referenced executors/buffers exist; bindings select available engines and memory spaces; routes match required capabilities.
- `ResourceVerifier` (proposed adapter): shared-memory/register allocation per CTA, capacity and lifetime constraints, legality of target layouts and instruction families.
**Lowering contract:** `(FlowKernel, MappingProgram, HardwareProfile) → ArchNativeKernel`. Flow determines software partial order; Mapping determines physical binding; target lowering chooses native instruction forms, their legal layout and synchronization protocol. Hardware contention and cycle timing remain in GTSim.
## 7. IR-04 — ArchNative IR
### 7.1 Contract
IR-04 represents a **single concretely scheduled target kernel**. Its operations expose architecture-native instruction categories and the associated synchronization semantics. It is not required to be executable PTX or SASS, and it does not decide a new tile schedule.
**Core objects:** `NativeKernel`, `NativeCTA`, `NativeWarpGroup`, `NativeBuffer`, `NativeOp`, `NativeEvent`, `NativeBarrier`, `NativeRegion`.
### 7.2 Hopper Example
```mlir
arch.kernel @gemm_sm90 target<sm_90a> {
  arch.cta @block0 threads(384) warps(12)
  arch.buffer %a0 : smem<128x64xbf16> stage(0)
  arch.buffer %a1 : smem<128x64xbf16> stage(1)
  arch.buffer %b0 : smem<64x256xbf16> stage(0)
  arch.buffer %b1 : smem<64x256xbf16> stage(1)
  arch.barrier %full0 kind<mbarrier> phase(0)
  arch.op @load_a0 opcode<cp.async.bulk.tensor.2d>
      executor<warp8> reads[@A[0,0]<128x64>] writes[%a0]
      signals[!full0] on_complete
  arch.op @load_b0 opcode<cp.async.bulk.tensor.2d>
      executor<warp8> reads[@B[0,0]<64x256>] writes[%b0]
      signals[!full0] on_complete
  arch.op @mma0 opcode<wgmma.mma_async.m64n64k16>
      executor<warpgroup0> reads[%a0,%b0] after[!full0]
  arch.op @mma_wait opcode<wgmma.wait_group> after[@mma0.issue]
}
```
This is a **compact inspection example**, not a complete legal PTX or a fully lowered kernel. Real lowering must preserve TMA transaction-byte accounting, mbarrier phases, WGMMA collective grouping, shared/register layouts, commit/wait groups, accumulator dependence, epilogue stores and full CTA lifetime.
Target adapters are separately defined for Ampere (`mma.sync`, `cp.async`), Hopper (`wgmma`, TMA, mbarrier) and Blackwell (`tcgen05`, TMA, TMEM). Shared IR syntax does not imply their machine semantics are interchangeable or that the current GTSim fork already fully models all three.
### 7.3 Lowering Obligations
1. **Instruction selection:** map each Flow capability to a legal target operation or reject it.
2. **Resource legalization:** finalize physical buffer layouts, TMA descriptors, collectives and register/shared footprints.
3. **Synchronization legalization:** convert abstract `data_ready`/`release` conditions into the target's barrier and async-group protocol, preserving issue versus completion semantics.
4. **Kernel verification:** every global output region is produced, every consumed input region is ready, and no stage slot is overwritten before its consumers finish.
**Lowering contract:** `ArchNativeKernel → Kernel Execution DAG`, with all target-dependent synchronization interpreted into explicit dependencies. IR-04 should not change mathematical outputs or silently omit memory transactions.
## 8. IR-05 — Execution Program and Kernel DAGs
### 8.1 Two-Level Execution Structure
IR-05 consists of (a) a **Program execution manifest**, retaining stream and launch/event structure from IR-01, and (b) **Kernel DAGs**, obtained from IR-04. This is a single execution layer, but it need not be serialized as one enormous flat graph.
```mlir
exec.program @decode_layer_0 {
  exec.stream @compute ordered {
    exec.launch @input_norm dag<@input_norm.gtsir>
    exec.launch @qkv_projection dag<@qkv.gtsir> after[!input_norm.complete]
    exec.launch @attention_decode dag<@attention.gtsir> after[!kv_update.complete]
    exec.launch @router dag<@router.gtsir> after[!attention.complete]
    exec.launch @moe dag<@moe.gtsir> after[!router.complete]
  }
  exec.tensor @kv_cache allocation<external> lifetime<decode_session>
  exec.tensor @expert_weight allocation<external> lifetime<model>
}
```
This `exec.program` textual form is **proposed**: the proposed IR-05 export uses `.gtsir` for both `program.gtsir` (Execution Program) and `kernel.gtsir` (Kernel DAG), distinguished by a required versioned `format` field. Current GTSim `.gtsir` v1/v2 is a **per-kernel** Tile IR and does not already implement a stream manifest; its legacy syntax and reader remain unchanged. The v0 runtime supports **one ordered stream with multiple sequential kernel launches**, even though the initial public GEMM and FA3 examples each launch only one kernel. An `exec.launch` references a kernel graph; it does not inline its nodes into one global DAG.
A kernel DAG may retain the current GTSim wire format:
```text
gtsim.tile_ir 1
tensor id=0 name=A shape=[128,128] strides=[128,1] dtype=fp16 tier=HBM
tensor id=1 name=B shape=[128,256] strides=[256,1] dtype=fp16 tier=HBF
tensor id=2 name=C shape=[128,256] strides=[256,1] dtype=fp16 tier=HBM
cta id=0 sm=0 warps=12 shared=163840 registers=180224
%0 = cp.dram2sram_tma name=loadA cta=0 warp=8 tensor=0 dtype=fp16 tile=[0,0,128,64] after_issue=[] after_complete=[]
%1 = cp.dram2sram_tma name=loadB cta=0 warp=8 tensor=1 dtype=fp16 tile=[0,0,64,64] after_issue=[] after_complete=[]
%2 = wgmma.m64n64k16 name=mma0 cta=0 warp=0 tensor=2 dtype=fp32 tile=[0,0,16,64] after_issue=[] after_complete=[0,1]
```
This is a **structural excerpt**, not a complete runnable `.gtsir`: a legal WGMMA requires all four collective warp fragments and correct complete output coverage. It demonstrates the actual GTSim fields, not a new parser format.
### 8.2 Cross-Kernel Composition Requirements
- **Tensor identity:** kernel-local tensor IDs map to stable Program Tensor IDs. The same weight/KV allocation must not be recreated for every kernel invocation.
- **Address identity:** global memory region and device tier remain consistent across producers and consumers; Kernel DAGs reference views of shared Program allocations.
- **Kernel completion:** includes required output-store completion/visibility, not merely issuance of the last warp instruction.
- **Stream ordering:** completion/launch dependencies are explicit and cannot be lost during DAG concatenation or lazy kernel instantiation.
- **CTA resources:** each invocation has its own CTA resource budget and launch boundary. ID renumbering alone is not a kernel scheduling model.
- **Runtime state:** KV Cache updates, request lengths, expert routing and residency persist across layer and kernel boundaries.
- **Single-stream command semantics (v0):** submit commands in stream order; a subsequent launch is eligible after the previous kernel's required completion, while event record/wait and stream synchronize observe the same completion clock. Reject multiple streams; do not simulate cross-stream overlap.
- **GPU-state persistence:** one Execution Program owns a persistent GPU/MemorySystem context and monotonically increasing global cycle. Per-launch CTA/warp schedulers and dependency scoreboards are initialized and retired without recreating the GPU, caches, memory controllers, or global tensor allocations.
- **Launch accounting:** kernel launch overhead is explicit and configurable (default `0` cycles for legacy parity), recorded separately from kernel device time; no hidden per-kernel startup cost.
- **Fallback kernels:** a kernel represented by a calibrated latency model must still have input/output identities and boundary events; timing must not silently default to zero.
### 8.3 GTSim Execution Semantics and Ownership
The **Frontend** owns `.progir` construction, IR validation, pass orchestration, serialization, and IR-01 + kernel-DAG composition. The **C++ Runtime** owns stream-command execution and GPU lifetime. Do not place stream scheduling or GPU state inside `frontend/passes/`, and do not place IR-01 compiler passes inside GTSim `src/runtime/`.
```text
frontend/ir/program.py + frontend/passes/compose.py
    → program.gtsir (exec.program; ordered launch/event/sync commands)
frontend/passes/lower_execution.py
    → kernel.gtsir (exec.kernel; per-CTA/warp operation graph)
src/execution_ir/{graph_codec,graph_materialize}.cpp
    → validated per-kernel DAGNode graph, program tensor bindings
src/runtime/program_executor.cpp
    → one stream command queue + monotonic global cycle
src/runtime/kernel_dispatcher.cpp
    → per-launch scheduler/scoreboard lifecycle on persistent GPUContext
GTSim GPU/MemorySystem
    → shared L2, HBM/HBF, device resources and continuous timing
```
The existing `Simulator(DAG*, config)` constructs a new GPU and per-DAG scheduler; repeated `Simulator` instances are **not** valid multi-kernel execution because they reset GPU/cache/memory state and restart the cycle counter. A thin loop around `Simulator::run()` is therefore insufficient. Refactor the lifecycle into persistent `GPUContext` and per-launch `KernelExecutionSession` (exact C++ names are implementation choices), retaining a single-DAG compatibility API for the original executables.
**v0 state machine:** `CREATED → QUEUED → RUNNING → COMPLETED` per launch; `COMPLETE` requires dependency closure and required output visibility. `StreamSynchronize` waits until the stream's preceding commands complete. The baseline single-stream runtime serializes kernels, so no GPU-overlap scheduling is needed yet; completed kernels must not flush L2 or reset MemorySystem. Explicitly account for in-flight writes/requests when defining completion and before admitting dependent kernels.
### 8.4 Execution/Memory Boundary
`TensorRegion` and op metadata lower to a logical memory access: `(tensor_id, region, dtype, layout, operation)`. GTSim allocates its physical base addresses, produces memory requests and simulates cache, interconnect and HBM/HBF timing. Precise physical addresses need **not** be embedded at the Algorithm or Flow level.
Two dependency conditions must be preserved:
- `after_issue`: predecessor was issued; permits legal asynchronous overlap.
- `after_complete`: predecessor has completed; required before consuming data or reusing a protected buffer.
These are the GTSim counterparts of Uniform Sim Plan's `release<issue>` and `release<complete>` conditions. Uniform Sim Plan is useful as a semantic reference, **not** a second timing engine in this system.
## 9. Lowering Matrix
| Conversion | Required inputs | Output | Preserves | Decides / expands | Rejects |
|---|---|---|---|---|---|
| Serving → IR-01 | model metadata, `DecodeStepContext`, implementation catalog | `Program` | model layer dependencies, per-request context | invocation list, tensor shapes/views, initial streams | missing kernel model, incompatible routing/shape |
| IR-01 + IR-02 → IR-03 | invocation, algorithm template, baseline schedule | `FlowKernel` + invocation link | math, tensor identity, value dependencies | tile domains, executor roles, pipeline slots, logical op order | unsupported template, illegal tails, incomplete outputs |
| IR-03 + Mapping + Hardware → IR-04 | FlowKernel, MappingProgram, target profile | `ArchNativeKernel` | dependencies, buffer lifetime, work/traffic | native instructions, layouts, synchronization, resources | missing capability, invalid route, resource overflow |
| IR-04 → kernel DAG | native kernel | `kernel.gtsir` | issue/complete meaning, memory regions | unroll to CTA/warp operation nodes and explicit dependencies | unsupported instruction, barrier deadlock, illegal reuse |
| IR-01 + kernel DAGs → IR-05 | Program skeleton, invocation ↔ DAG mapping | `ExecutionProgram` | stream order, cross-kernel data visibility | global IDs, ordered launch/event/sync commands, kernel graph references and tensor bindings | missing global tensor binding, dangling events, unsupported multi-stream execution in v0 |
| IR-05 → GTSim | ExecutionProgram, GPU/memory profile | per-kernel + program timeline/statistics | launch/completion/sync semantics; persistent tensor and memory state | per-launch scheduler/scoreboard lifecycle on one GPU, global cycle, memory requests, contention and timings | unsupported stream count, unavailable hardware route, impossible allocation |
**Stage separation:** A scheduled `FlowKernel` is already an implementation; `MappingProgram` associates it with physical resources; architecture lowering selects machine-operation semantics; DAG lowering translates those semantics into a simulator-readable partial order. None of these later stages should implicitly retile the kernel.
## 10. Data, Event and Resource Contracts
| Contract | Definition | Ownership |
|---|---|---|
| `TensorID` | stable logical identity across kernels | IR-01; referenced by IR-02..05 |
| `TensorRegion` | offsets, shape, dtype, mask, logical layout | IR-02 / IR-03 |
| `TileValue` | SSA value and producer | IR-03 |
| `SlotHandle` | buffer, slot index, generation | IR-03 |
| `CompletionEvent` | `issued` / `accepted` / `data_ready` / `visible` / `released` as appropriate | IR-01 / IR-03; legalized in IR-04 |
| `Executor` | logical producer/consumer/matrix/vector role | IR-03 |
| `ResourceBinding` | logical executor/buffer → physical engine/memory | Mapping adjunct |
| `CTAResources` | actual SMEM/register/warp requirements | IR-04 / IR-05 |
| `MemoryTransaction` | addresses, bytes, transaction behavior | IR-05 / GTSim |
**Buffer lifetime:** `acquire → producer write → data_ready → consumer read → last consumer completion → release`. Software may express logical `memory_class=staging`; the Mapping adjunct supplies `map.store ... -> @GPU_SMEM`, and the Hardware profile supplies the capacity. Compile-time verification checks per-CTA storage; runtime checks account for concurrently admitted CTAs and global-memory allocations.
**Program-level ordering:** a producer's output becoming visible is not automatically identical to its instruction issue event. Kernel completion and event waits must be modeled at the proper completion point. The v0 runtime implements ordered commands on one stream; cross-stream event dependencies are reserved in the schema but not executed concurrently. Program-global allocations and cache/memory state survive successive kernel launches.
## 11. Formats and Validation
Use two encodings where practical:
- **Machine interchange:** versioned JSON payloads saved with the **layer-specific extensions** below. The extension identifies the IR level; the required `format`/`schema_version` fields identify the wire schema. Existing legacy GTSim `.gtsir` v1/v2 text remains supported unchanged.
- **Human inspection:** deterministic MLIR-like text in the style of `FlowPrinter`, `MappingPrinter` and `SimPlanPrinter`; it is not assumed to be a runnable parser input.
| IR level | Canonical exported file | Payload |
|---|---|---|
| IR-01 Program | `program.progir` | Versioned Program JSON (`prog-ir/0.1`) |
| IR-02 Algorithm Template | `kernel.algoir` | Versioned Algorithm JSON (`algo-ir/0.1`) |
| IR-03 Scheduled Flow + Mapping | `kernel.flowir` | Versioned Flow JSON (`uniform-flow-ir/1.0`-inspired), containing its ResourceMapping adjunct |
| IR-04 ArchNative | `kernel.nvir` | Versioned target-native JSON (`arch-ir/0.1`) |
| IR-05 Execution | `kernel.gtsir`, `program.gtsir` | Versioned kernel-DAG/program JSON with distinct `format` discriminators; legacy `gtsim.tile_ir 1/2` remains a valid `.gtsir` payload |
No additional `.mapir` extension is introduced: ResourceMapping is embedded in IR-03 and may be printed separately only as an inspection view.
IR-01/02 and ArchNative are **new proposed schemas**. Uniform Flow IR v1.0 and Mapping IR already exist in the uploaded project; GTSim Tile IR v1/v2 and `DAGNode` already exist in the fork. Their formats are **not directly compatible today**. An explicit conversion adapter is required at each marked boundary.
Every conversion should emit verifiable provenance: `(source_ir_hash, target_ir_hash, template_id, schedule_id, hardware_profile_id, tensor_mapping, invocation_mapping)`. The hashes are bookkeeping; they do not affect simulation timing.
## 12. End-to-End Example of One GEMM Invocation
```text
IR-01: prog.launch @qkv_projection callee<@gemm> inputs[%hidden,@W_qkv]
    │ reference + shape bindings
IR-02: algo.kernel @gemm family<tiled_matrix,producer_consumer_pipeline>
    │ instantiate fixed reference schedule BM128 BN256 BK64 depth2
IR-03: flow.kernel @qkv_gemm_s0
    │ flow.tensor / flow.executor / flow.buffer slots(2)
    │ flow.pipeline / flow.load_async / flow.matmul / flow.store_async
    │ + mapping.program @qkv_sm90: map.execute/map.store/map.tensor/map.route
    │ legalize to H100 TMA + WGMMA
IR-04: arch.kernel @qkv_sm90
    │ TMA descriptors, SMEM layout, mbarrier phases, WGMMA groups
    │ lower structured control and synchronization to explicit dependencies
IR-05: kernel.gtsir + program.gtsir (exec.program @decode_step)
    │ warp/CTA ops + issue/complete edges + global tensor mapping
GTSim ProgramExecutor: stream command processing, sequential kernel launch/completion, persistent GPU/L2/memory state and global cycles
GTSim timing backend: per-launch CTA/warp scheduling, memory traffic and resource contention
```
For a full Decode Step, the same conversion is applied to each kernel invocation while **one** Execution Program retains the inter-kernel relations, KV Cache state, and weight allocations. Baseline kernels may use existing GTSim C++ DAG builders as an explicit **alternative to IR-02→04 compilation**; the result still enters through the same IR-05 composition and tensor/event contracts. This avoids requiring a new universal compiler before end-to-end timing can be simulated.
## 13. Normative Separation
- IR-01 owns **invocation orchestration**, not tile-level implementation.
- IR-02 owns **kernel computation semantics and known algorithm family**, not a chosen schedule.
- IR-03 owns **scheduled logical compute/dataflow** and its associated resource mappings, not native instruction timing.
- IR-04 owns **architecture-specific execution semantics**, not a new algorithm or tile search.
- IR-05 owns **simulator-executable operations and dependency structure**, not compilation choices.
- GTSim ProgramExecutor owns **stream commands, kernel launch/completion, shared GPU lifetime and monotonic global cycles**; the existing GTSim timing backend owns **runtime resource arbitration, addresses, memory queues and per-launch execution**.
The entire design is intended to support a **fixed-schedule baseline decode step first**. The IR contracts permit multiple schedules, but no automatic DSE, speculative prefetch protocol, or new HBF research mechanism is required by this specification.
