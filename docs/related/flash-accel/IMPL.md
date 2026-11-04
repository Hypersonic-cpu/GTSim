# GTSim FlashAccel Prefetch Reproduction Plan

## 0. Goals and Constraints

**Repository:** `Hypersonic-cpu/GTSim`
**Working branch:** `related/flash-accel` (already created from `serve-sim`)
**Reference:** FlashAccel, arXiv:2607.10186v2, especially Figures 5, 7, 10, 12 and Sections IV–VI.

### Goals

Reproduce FlashAccel's weight placement and asynchronous HBF-to-SRAM prefetch mechanism using GTSim, with a **Co-located H100 configuration**.

Implementation order:

1. GEMM weights in HBF; FA3 historical K/V in HBF.
2. Qwen3-30B-A3B single Transformer Block with static weights in HBF.
3. Complete and validate the no-prefetch baseline.
4. Implement HBF-side SRAM and asynchronous prefetch.
5. Implement automatic tiling, prefetch scheduling, and release insertion.
6. Validate performance with controlled ablations.

### Non-negotiable Constraints

- **No FlashAccel-specific behavior in generic GTSim or HBFine components.** Any shared changes must be reusable mechanisms controlled by configuration or interfaces.
- Do not modify existing standard `frontend/lib/*.algoir` or `frontend/lib/*.flowir`.
- Prefer independent frontend compiler passes and a separate ProgIR over modifying the core compiler.
- Preserve existing GEMM/FA3 execution semantics, GPU scheduling, and memory timing in configurations that do not enable the new features.
- Do not implement OS-level persistent objects, FTL, actual GPU page tables, or TLB shootdown. Preserve the equivalent address-translation and residency behavior in the simulator.
- Reuse the existing TensorRegistry, MemorySystem, HBFine, FlowIR, ArchIR, and IR-05 execution infrastructure wherever possible.
- Keep HBF-side SRAM separate from GPU SM Shared Memory.
- Do not implement alternative prefetch optimization policies yet. First reproduce FlashAccel's original policy.
- Do not modify unrelated components or introduce compatibility-breaking changes.
- Do not create a separate progress-tracking file. **Report implementation results, tests, performance measurements, and blockers directly to the user after each step.**
- Complete and validate each step before proceeding to the next. Never silently substitute an approximation for a required physical mechanism.

## Step 1 — Establish Configuration and Regression Baselines

**Purpose:** Establish a stable H100 Co-located environment without changing existing results.

### Tasks

1. Confirm the working branch is `related/flash-accel`. Read paper and markdown in `docs/related/flash-accel/`.
2. Preserve existing H100 and HBFine configurations.
3. Reuse existing configurations, do NOT create FlashAccel-specific config if it can be done via generic file. (e.g. 2F3M config is generic).
4. Use **2 HBF + 3 HBM stacks (2F3M)** as the initial H100 Co-located configuration.
5. Keep the existing H100 GPU compute parameters. Use a separate HBF profile derived from FlashAccel v2 Table III:
   - 8 Flash dies per HBF stack.
   - 96 planes per die.
   - 32 planes per Megaplane.
   - 4 KiB per NAND page.
   - 4 μs NAND read latency, 75μs program latency.
   - 768 GB/s target read bandwidth per HBF stack.
   - 32 KiB SRAM per plane; 8 MiB additional base-die SRAM per stack.
6. Run existing GEMM, FA3, and serving-related regression tests before modifications.

The new HBF profile must be consistent with its physical geometry and achievable array bandwidth. Do not simply overwrite the bandwidth parameters in the existing HBFlex/OCP model.

The initial 2F3M target is defined in `configs/memconf/2F3M_grouped_planes.yaml`. Its HBM rates are the existing H100 effective rates scaled by 3/5 (1,860 GB/s read and 930 GB/s write). HBF has a 1,536 GB/s aggregate read target (2 × 768 GB/s). The ideal NAND sense ceiling is 1,572.864 GB/s across both stacks (2 × 8 × 96 × 4,096 B / 4 µs), before contention and transfer overhead. The modeled GPU links have separate ceilings of 4,608 GB/s for HBM and 3,072 GB/s for HBF. The target uses the Table III block/page counts, interprets the SRAM capacities as binary units, and disables programming. Use the grouped HBFine backend with Ramulator2 HBM3. The three-stack Ramulator profile models a 2,005.222 GB/s pin ceiling; the legacy effective-rate fields are retained for configuration reporting.

**Acceptance:**

- [ ] Original H100 and HBFine regression baselines remain unchanged.
- [ ] Co-located configuration is defined independently, with all paper-derived parameters and adaptations identified.

## Step 2 — Generalize HBF Geometry and Physical Address Mapping

**Purpose:** Support FlashAccel's physical weight layout from the beginning, avoiding unnecessary read amplification.

### Tasks

1. Inspect HBFine's current geometry constraints, address encoding/decoding, Bank/Plane timing, and page-buffer model.
2. Extend HBFine with a **generic, optional multi-plane command grouping capability**:
   - Configurable dies, planes, and planes per command group.
   - Group members share NAND command control.
   - A grouped command accesses the same block/page offset across its constituent planes.
   - Independent groups can execute concurrently.
   - Preserve existing Bank/Plane behavior for old configurations.
3. Implement a generic physical address mapping/placement interface rather than hardcoding FlashAccel placement into TensorRegistry.
4. Implement the weight layout algorithm specified in Figure 7:
   - Partition static weights into Megapage-sized chunks.
   - Arrange weights according to operator execution order.
   - Assign chunks round-robin across all Megaplanes.
   - Continue the next operator from the Megaplane following the previous assignment.
   - Preserve correct handling of partial final Megapages.
5. For the initial co-located configuration:
   - Megapage = 128 KiB.
   - 24 Megaplanes per stack.
   - 48 Megaplanes across two stacks.
   - Full Hyperpage = 6 MiB.
6. Provide a logical tensor offset → physical HBF address mapping that existing memory operations can use. Keep logical tensor shapes and GEMM indexing unchanged.
7. Add a layout inspection tool that reports physical address, Stack, Die, Megaplane, Block, Page, and logical tensor range.
8. Distinguish useful requested bytes, HBF host transfer bytes, and actual NAND sensed bytes. Avoid forcing full-Hyperpage reads when only a subset is needed.

Do not implement persistent objects, OS-level Megapages, or an FTL. Megapage/Hyperpage are physical organization and mapping abstractions here, not OS-managed objects.

The `grouped_planes` HBFine address codec lays out each Megapage as contiguous plane pages. Consecutive Megapages rotate through stacks, then dies, then groups within a die, before advancing the common page offset. Plane pages are spread over host channels by physical plane index. A miss senses one Megapage and fills one of two buffers in its command group. Only requested beats cross the host interface. Other groups can sense concurrently. These buffers are separate from the SRAM capacity reserved for the later prefetch mechanism.

Use `HBFMegapageAllocator` in operator execution order. Each allocation reserves whole Megapages and exposes only the original logical byte range. `HBFTensorPlacement::map` returns the physical address and usable span up to the next Megapage boundary. Step 3 wires this interface into IR-05. The controller codec and allocator share the same geometry, which is configurable in the memory YAML.

```sh
cmake -S . -B build-backends -DGTSIM_ENABLE_RAMULATOR2=ON
cmake --build build-backends --target gtsim_hbf_layout hbf_grouped_geometry_check
build-backends/hbf_grouped_geometry_check
build-backends/gtsim_hbf_layout configs/v0.1.0/h100-sxm.yaml \
  configs/memconf/2F3M_grouped_planes.yaml A=131136 B=262144
```

The inspection tool takes tensors in allocation order. The CSV reports each logical interval and the corresponding physical Megapage. Simulator diagnostics separate completed kernel logical HBF read bytes, host read bytes and NAND sensed bytes. The NAND count includes all planes in a sensed group, including unrequested lanes of a tail Megapage.

**Acceptance:**

- [ ] Physical mapping, grouped NAND access behavior, parallelism, and address encode/decode pass dedicated tests.
- [ ] Physical traffic and read amplification are measurable; original HBFine geometry and timing tests remain unchanged.

## Step 3 — Enable HBF Tensor Placement in IR-05

**Purpose:** Allow existing GEMM/FA3 and ProgramExecutor paths to access tensors located in HBF.

### Tasks

1. Remove the HBM-only restrictions in:
   - `src/runtime/program_executor.cpp`
   - `src/execution_ir/graph_materialize.cpp`
2. Reuse existing `MemoryTier`, `TensorRegistry`, and segmented HBM/HBF address allocation.
3. Ensure Program-level and Kernel-level tensor bindings preserve the selected memory tier.
4. Integrate the general physical mapping interface from Step 2 with tensor address generation.
5. Preserve existing L2 and MemorySystem request paths for ordinary demand accesses.
6. Ensure HBF-resident input tensors are initialized through existing preload mechanisms without incurring simulated runtime writes.
7. Reject invalid tier bindings, out-of-range addresses, unintended HBF writes, and overlapping allocations.

Do not introduce prefetch behavior in this step.

Implemented IR-05 tier validation and preloaded read-only HBF allocations. In
grouped geometry the Program manifest order determines Megapage placement.
Kernel bindings preserve shape, strides, dtype and tier. The mixed two-GEMM
check transferred 360,448 B from HBF with zero HBF writes. Its all-HBM
counterpart retained the original 14,494-cycle timing.

*Note*:
- Experts and weight should be "flattened" as much as possible.
    - The HBF capacity is large, but you should NOT place all weights on few dies / planes. Instead, you should flatten it accross different pages/planes/dies/stacks, so that than can read in parallel.
    - Design this address mapping and tensor placement according to related papers (search and read!) carefully.
    - Address mapping (how single or multi tensor go accross the physical hierarchies) should be configuable, (e.g. see GPGPU-Sim config for example). Tensor placement (how to assign addresses to each tensor (should do it in software)) should be explicit.
    - Report me about your Mapping and Placement design. Tell me how to change them if I asked.

**Acceptance:**

- [x] IR-05 supports HBM/HBF tensors with correct physical address routing and memory traffic.
- [x] Existing all-HBM programs produce unchanged results in the two-GEMM regression.

## Step 4 — GEMM and FA3 HBF Baselines

**Purpose:** Validate HBF placement and physical layout using small, controlled workloads before attempting a Transformer Block.

### Tasks

1. **GEMM:** Place weight B in HBF; keep activation A and output C in HBM.
2. **FA3:** Place historical K/V cache in HBF; keep Q, intermediate data, and output in HBM. FA3 has no static model weight.
3. Keep original GEMM/FA3 algorithms, tiling, and scheduling unchanged.
4. Use the physical placement infrastructure from Step 2 rather than ordinary contiguous HBF tensor allocation.
5. Test both a single kernel and repeated kernels accessing shared HBF tensors.
6. Validate:
   - Logical addresses and actual HBF physical addresses.
   - Tensor coverage and absence of unintended accesses.
   - HBF read traffic and NAND senses.
   - Effect of physical layout on Megaplane utilization.
   - Persistence of GPU L2 and HBFine internal state across kernels.
7. Compare with the existing HBM baseline and a simple HBF layout without execution-order optimization.

Do not introduce SRAM prefetch, new compute tiling, or runtime scheduling changes.

### Implemented and measured

The independent programs in `frontend/asm/flash-accel/` call the existing
textual GEMM and FA3 library schedules twice with shared input allocations.
GEMM uses `M=256, N=256, K=192`. FA3 uses eight Q heads, two KV heads, 128
query rows per head and 576 K/V rows per KV head. The FA3 reference here is
the existing full-query text schedule. IR expansion now includes a scheduled
iteration ID in instruction names when multiple iterations share a physical
producer warp, which fixes duplicate FA3 output-copy names.

The baseline uses 132 SMs, Ramulator2 HBM3 and grouped HBFine on 2F3M. The
packed comparison sets `hbf.align_tensors_to_command_group: false` in the same
memory YAML, keeping the NAND geometry and link timing fixed. IR-05 timing
reports tensor bases under `tensor_addresses[<tier>]` and per-kernel tier
traffic under `memory[<tier>]`. HBFine supplies the `sensed_bytes` and
`sensed_pages` counters and sense unit activity through the backend interface.
Placement policies are created per tier before the devices start. The
Megapage allocator lives inside the HBFine placement implementation.

| Workload and placement | First launch cycles | Two-launch cycles | First host HBF read | First NAND sense | Active groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEMM all HBM | 6,602 | 12,530 | 0 B | 0 B | 0 |
| GEMM grouped or packed | 13,738 | 19,669 | 98,304 B | 131,072 B | 1 |
| FA3 all HBM | 19,533 | 37,336 | 0 B | 0 B | 0 |
| FA3 grouped | 30,776 | 49,091 | 589,824 B | 786,432 B | 6 |
| FA3 packed | 30,215 | 48,394 | 589,824 B | 655,360 B | 5 |

The standalone grouped launches reproduce the first-launch cycles. An
independent audit enumerated logical read regions, checked exact coverage of
each HBF input and matched the predicted physical groups to the simulator
counters. GEMM uses one tensor and both allocation policies place it at the
same address. For FA3, aligned placement starts K at group 0 and V at group 3.
Packing starts V in the remainder of K's third group, reducing the number of
sensed Megapages from six to five for this input. Host transfer bytes agree.

With the default L2, second launches issue zero HBF reads. In a separate
64 KiB L2 experiment, the second GEMM and FA3 launches transfer 98,304 B and
592,256 B respectively from HBF with zero new NAND sense bytes. The repeated
FA3 host traffic includes 2,432 B of line rereads due to L2 eviction.

```sh
PYTHONPATH=frontend/src python3 -m compile frontend/asm/flash-accel/gemm.progir \
  --lib frontend/lib --param NUM_SMS=132 -o /tmp/flash-gemm.dag.json \
  --runtime-dir /tmp/flash-gemm
PYTHONPATH=frontend/src python3 -m compile frontend/asm/flash-accel/fa3.progir \
  --lib frontend/lib --params frontend/asm/flash-accel/fa3.json \
  -o /tmp/flash-fa3.dag.json --runtime-dir /tmp/flash-fa3
build-backends/gtsim_program configs/v0.1.0/h100-sxm.yaml --memory-config \
  configs/memconf/2F3M_grouped_planes.yaml /tmp/flash-fa3/program.gtsir
```

For the HBM control, compile a copy with the read-only HBF declarations set to
HBM. For a single launch, retain the first launch and its command followed by
the stream synchronization. Logs and manifests are under
`validation/results/flash-accel/step4/`.

**Acceptance:**

- [x] GEMM Weight-in-HBF and FA3 K/V-in-HBF run correctly with audited addresses and traffic.
- [x] The physical layout is verified, and the measured read amplification is explained by Megapage boundaries.

## Step 5 — Qwen3-30B-A3B Single-Block HBF Baseline

**Purpose:** Establish a complete Transformer Block baseline with HBF-resident weights before implementing FlashAccel prefetch.

### Tasks

1. Copy `frontend/asm/qwen3/one_block.progir` into an independent experiment directory, e.g. `frontend/asm/qwen3/flash-accel/one_block.progir`.
2. Preserve existing `frontend/lib/` AlgorithmIR and FlowIR definitions.
3. Set static model weights to HBF in the copied ProgIR:
   - Attention projection weights.
   - Normalization weights.
   - Router weights.
   - MoE expert Gate/Up/Down weights.
4. Keep activation, intermediate tensors, and KV cache in HBM for this baseline.
5. Reuse the existing `prog.launch` input/output declarations to identify tensor consumers and operator order.
6. Apply Step 2's execution-order physical layout for HBF weights. For expert weights, preserve logical expert indexing and the existing routing trace; do not assume a future dynamic expert execution order.
7. Compile through the normal ProgIR → FlowIR → ArchIR → IR-05 pipeline.
8. Validate existing Qwen tensor associations, routing indices, dependencies, and execution behavior.
9. Record end-to-end cycles, per-kernel cycles, HBF/HBM bytes, NAND senses, and Megaplane utilization.

**This step must finish before any SRAM prefetch or compute-tiling changes.** Treat the result as the frozen no-prefetch baseline.

The independent program at `frontend/asm/qwen3/flash-accel/one_block.progir`
changes the storage tier of the nine weight tensors. Their declaration order
matches the order of first consumption. Expert rows retain their original
expert-indexed layout and routing trace. The B=1 validation compares the
compact ArchIR to the all-HBM program after normalizing tensor tiers, and
checks the existing request, head, RoPE position and expert associations.

```sh
PYTHONPATH=frontend/src python3 frontend/tools/run_qwen_simulations.py \
  --source frontend/asm/qwen3/flash-accel/one_block.progir \
  --params frontend/asm/qwen3/b1.json --runner build-backends/gtsim_program \
  --architecture configs/v0.1.0/h100-sxm.yaml \
  --memory configs/memconf/2F3M_grouped_planes.yaml \
  --out validation/results/flash-accel/step5/qwen-b1-hbf
```

For the all-HBM control, omit `--source` and use a separate output directory.
Both runs stream one expanded kernel at a time through a named pipe. The
Program manifests, provenance, simulator logs and timing JSON remain in the
respective output directories.

### Measured B=1 baseline

One decode step at context length 10,000 completes all 18 kernels. Both
runs use H100 132 SM, Ramulator2 HBM3 and 2F3M grouped HBFine.
The HBF run takes 4,291,764 cycles versus 623,329 cycles for the
matched all-HBM control. It reads 113,779,200 B from HBF and
20,679,552 B from HBM, writes 192,896 B to HBM and writes zero B
to HBF. NAND senses total 141,295,616 B across 1,078 commands.

| Kernel | HBM control cycles | HBF weights cycles | HBM read B | HBM write B | HBF read B | NAND sense B | Active groups / 48 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| input_rmsnorm | 1,870 | 9,517 | 4,096 | 4,096 | 4,096 | 131,072 | 1 |
| qkv_projection | 53,256 | 545,285 | 4,096 | 10,240 | 20,971,520 | 20,971,520 | 48 |
| q_norm_rope | 1,312 | 9,117 | 8,192 | 8,192 | 256 | 131,072 | 1 |
| k_norm_rope | 1,374 | 9,105 | 1,024 | 1,024 | 256 | 131,072 | 1 |
| v_extract | 1,171 | 1,204 | 1,024 | 1,024 | 0 | 0 | 0 |
| kv_cache_update | 1,251 | 1,191 | 2,048 | 2,048 | 0 | 0 | 0 |
| fa3_decode_split | 57,574 | 59,478 | 20,490,240 | 67,456 | 0 | 0 | 0 |
| fa3_combine | 1,563 | 1,477 | 66,048 | 8,192 | 0 | 0 | 0 |
| attention_output_projection | 95,668 | 1,085,879 | 8,192 | 4,096 | 16,777,216 | 16,777,216 | 48 |
| attention_residual_add | 1,564 | 1,552 | 8,192 | 4,096 | 0 | 0 | 0 |
| post_attention_rmsnorm | 1,792 | 9,517 | 4,096 | 4,096 | 4,096 | 131,072 | 1 |
| router_projection | 43,573 | 73,806 | 4,096 | 256 | 524,288 | 524,288 | 4 |
| router_softmax_topk | 2,181 | 1,570 | 256 | 256 | 0 | 0 | 0 |
| expert_gemm1_direct | 217,791 | 2,059,274 | 0 | 24,576 | 50,331,648 | 77,332,480 | 48 |
| swiglu | 1,852 | 1,861 | 24,576 | 12,288 | 0 | 0 | 0 |
| expert_gemm2_direct | 135,028 | 417,452 | 12,288 | 32,768 | 25,165,824 | 25,165,824 | 48 |
| weighted_moe_sum_direct | 2,959 | 2,932 | 32,896 | 4,096 | 0 | 0 | 0 |
| final_residual_add | 1,550 | 1,547 | 8,192 | 4,096 | 0 | 0 | 0 |

The audit matched all nine runtime addresses to the sequential Megapage
allocator and matched each weight kernel's active group IDs to its
lowered read regions. Both expert GEMMs accessed only experts
10, 66, 77, 98, 103, 107, 122 and 124. GEMM1 touches 384 unique
Megapages but senses 590 times as its accesses revisit pages after
buffer eviction. GEMM2 senses its 192 unique Megapages once each.
Normalization weights use partial Megapages. FA3 and KV update issue
zero HBF traffic. Active group counts measure coverage over a kernel,
rather than the number of groups simultaneously sensing.

**Acceptance:**

- [x] A complete Qwen3-30B-A3B Transformer Block executes with all intended static weights in HBF.
- [x] The B=1 baseline is validated and reported, including physical layout and memory traffic.

## Step 6 — Generic HBF-side SRAM and Asynchronous Memory Operations

**Purpose:** Add the minimum generic runtime mechanisms needed for FlashAccel's asynchronous prefetch, without changing normal GPU Kernel execution.

### Tasks

1. Implement an optional, configurable **HBF-side read SRAM residency layer**:
   - 32 KiB per plane.
   - 8 MiB base-die SRAM per stack.
   - Independent capacity and allocation state.
   - Preserve the existing HBFine Bank Page Buffers as a separate resource.
2. Support general asynchronous memory operations:
   - Reserve SRAM space.
   - Submit NAND → HBF SRAM prefetch.
   - Query completion/residency.
   - Release SRAM allocation.
3. Define request ownership:
   - Foreground requests belong to GPU Kernels.
   - Background requests belong to the Program-level memory scheduler.
4. Use one shared HBF physical timing model for foreground and background requests, including NAND sense engines, queues, and relevant transfer resources.
5. Keep independent background requests alive across kernel boundaries.
6. Preserve the current foreground kernel completion checks, but do not require background prefetch completion when retiring a kernel.
7. Distinguish `foreground_idle()` from global `all_idle()` or equivalent completion semantics.
8. Advance background memory events with the same global simulation clock as GPU computation.
9. Implement a generic logical-address/region-alias mapping:
   - Before prefetch completion, access the original HBF data.
   - After completion, the prefetch alias resolves to SRAM.
   - Ordinary non-prefetched accesses retain their original mapping.
   - Release reclaims the SRAM only when no outstanding consumer still uses it.
10. Avoid real page-table/TLB implementation. Document the assumed mapping and SRAM access latency; do not silently treat them as zero if the model requires a timing parameter.

The existing `Simulator::run()` waits for `MemorySystem::idle()` after all DAG nodes complete; it does **not** flush L2 or reset HBFine. Change only what is necessary to support separately owned background requests. Avoid removing correctness checks for foreground memory completion.

**Acceptance:**

- [ ] An asynchronous prefetch can overlap Kernel A and remain active when Kernel B begins; ordinary kernels retain their original completion semantics.
- [ ] SRAM allocation, completion, fallback access, release, physical contention, and unchanged non-prefetch behavior pass dedicated tests.

## Step 7 — Automatic Tiling and Prefetch Scheduling Passes

**Purpose:** Generate FlashAccel-style prefetch behavior automatically from the structured compute graph, HBF tensor placement, and SRAM capacity.

### Tasks

1. Implement experiment-specific compiler passes under `frontend/related/flash_accel/`, reusing existing compiler ASTs and lowering interfaces.
2. Keep the copied ProgIR coarse-grained:
   - Kernel invocations and dependencies.
   - Tensor inputs and outputs.
   - Tensor storage tiers.
   - Optional prefetch eligibility and tiling constraints.
3. Use a separate configuration for SRAM capacities, physical layout parameters, and scheduling policy rather than hardcoding them into the standard IR libraries.
4. Run passes after the initial FlowIR construction, before final ArchIR/IR-05 lowering.
5. Analyze the structured FlowIR to determine:
   - HBF-resident tensor usage.
   - Earliest legal prefetch point.
   - Logical tensor regions required by each consumer.
   - Last consumer of each prefetched region.
   - Compute dependencies between kernels and tiles.
6. Implement SRAM-aware compute tiling:
   - Determine legal tile sizes from SRAM capacity and existing GEMM schedule constraints.
   - Split oversized weight-consuming operations.
   - Preserve matrix semantics, including partial accumulation/reduction when splitting a reduction dimension.
   - Reuse existing GEMM lowering wherever possible.
   - Keep physical Megapage layout separate from compute tile size.
7. Generate fine-grained Prefetch/Release operations automatically.
8. Construct a global Prefetch Queue in compute-graph topological order, without fully flattening the entire program into one DAG.
9. Implement FlashAccel v2's initial scheduling policy:
   - Dispatch requests from the queue head whenever sufficient SRAM space becomes available.
   - Allow multiple outstanding prefetch requests.
   - Prefer circuit-die SRAM, then base-die SRAM.
   - Never block Compute merely because prefetch has not completed.
   - Release reserved space after the last consumer completes.
10. Lower the transformed compute schedule into regular kernel DAGs and the memory schedule into generic runtime memory events.
11. Validate the transformed IR before simulation: no cyclic dependencies, missing consumers, overlapping illegal SRAM lifetimes, incorrect tile coverage, or early release.

Do not modify existing `lib/*.algoir` or `lib/*.flowir`. If current lowering cannot represent an operation, add a general-purpose IR transformation or runtime interface rather than FlashAccel-specific handling.

**Acceptance:**

- [ ] Given a ProgIR and SRAM configuration, the passes automatically produce valid compute tiling, Prefetch Queue, and Release events.
- [ ] Transformed GEMM computations preserve original semantics, and the generated memory schedule satisfies SRAM capacity and dependency constraints.

## Step 8 — Prefetch Validation and Performance Evaluation

**Purpose:** Verify the reproduction against FlashAccel v2's mechanism and isolate the performance contributions.

### Tasks

Run tests in increasing complexity:

1. **Two-GEMM Microbenchmark**
   - Kernel A performs computation.
   - Kernel B consumes HBF-resident weights.
   - Prefetch Kernel B's weights while Kernel A executes.
   - Verify temporal overlap and correct fallback when prefetch is late.

2. **Tiled GEMM**
   - Verify automatic weight tiling, multiple outstanding prefetches, SRAM capacity management, and release order.
   - Check compute results/dependencies against the original untiled execution representation.

3. **Dense FFN**
   - Reproduce Figure 12's Gate/Up/Down-style prefetched partial computations.
   - Verify queue ordering and prefetch/compute overlap.

4. **Qwen3-30B-A3B Single Block**
   - Apply the same passes to the validated Step 5 baseline.
   - Preserve existing dynamic expert routing trace semantics.
   - Do not introduce speculative Expert prediction or future-information-based prefetch.

5. **Ablation Experiments**
   - HBM-only.
   - HBF with simple layout and no prefetch.
   - HBF with execution-order layout and no prefetch.
   - HBF with prefetch but without optimized layout.
   - HBF with both optimized layout and prefetch.
   - Selected SRAM capacity and compute tile size sweeps.

Collect:

- End-to-end and per-kernel cycles.
- HBF/HBM transferred bytes and NAND sensed bytes.
- Megaplane utilization and read amplification.
- SRAM occupancy, hit rate, allocation stalls, and release timing.
- Prefetch issued/completed counts and outstanding depth.
- Prefetch/Compute overlap and unhidden latency.
- Demand Read latency and interference from background prefetch.

Check that performance changes come from modeled mechanisms rather than incorrect tensor addresses, missing traffic, free transfers, or incorrect synchronization.

**Acceptance:**

- [ ] Microbenchmarks and the Qwen Block demonstrate correct asynchronous Prefetch/Release behavior with measurable physical traffic and overlap.
- [ ] Ablations and original GTSim regressions pass; performance results and model limitations are reported directly to the user.

---

## Implementation Completion Criteria

The reproduction is complete when:

- [ ] **Baseline:** GEMM, FA3, and Qwen Block correctly access HBF data using Figure 7-compatible physical placement, with measured read amplification.
- [ ] **Prefetch:** Automatic tiling, topology-ordered asynchronous prefetch, HBF SRAM residency, non-blocking fallback, and last-consumer release work across kernel boundaries.
- [ ] **Compatibility:** Existing GTSim/HBFine workloads run unchanged when the new capabilities are disabled. No FlashAccel-specific implementation is embedded in generic libraries or backend code.

**Scope limitation:** This project reproduces FlashAccel's mechanism using a Co-located H100 configuration. It does not attempt to match the paper's H200 throughput numbers, OS-level persistent storage management, KV writeback policy, or full multi-GPU serving evaluation.

For every step, the agent must directly report: **changes made, tests performed, measured results, remaining limitations, and whether the acceptance criteria passed.** Do not proceed past an unresolved correctness failure.
