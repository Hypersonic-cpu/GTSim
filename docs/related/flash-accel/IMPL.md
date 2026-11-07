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

- [x] The covered H100 and HBFine timing baselines remain unchanged. The initial CTest run had one existing profile-boundary false positive. Later steps use focused checks.
- [x] Co-located configuration is defined independently, with all paper-derived parameters and adaptations identified.

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

- [x] Physical mapping, grouped NAND access behavior, parallelism, and address encode/decode pass dedicated tests.
- [x] Physical traffic and read amplification are measurable; original HBFine geometry and timing tests remain unchanged.

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

### Implemented interface and validation

HBFine supports optional SRAM read paths through the same controller as ordinary
demand reads. A prefetch uses existing NAND queues, sense engines and page
buffers. It fills SRAM through a timed internal port, with no GPU-link transfer.
A resident read uses the SRAM port and then the normal host channel and GPU link.
The backend charges capacity in 4 KiB pages, preferring the source plane SRAM
and spilling to that stack's base-die SRAM. Reservation is atomic and release
waits for both fills and accepted consumers.

The read SRAM path is opt-in through `hbf.controller.enable_read_sram`. The
existing `sram_bytes_per_plane` and `base_die_sram_bytes_per_stack` fields set
capacity. `read_sram_access_latency_ns` and
`read_sram_bandwidth_GBps_per_stack` must be positive when enabled. The current
timing model uses one shared read/write SRAM port per stack for both locations.
The experiment uses 20 ns and 1,536 GB/s as explicit modeling assumptions.

IR-05 programs can declare `prefetches` entries with `id`, `tensor`, `offset`
and `bytes`. A `Prefetch` command reserves and starts that range. A launch opts
in with `read_aliases`, and a `Release` command ends the allocation lifetime.
The original logical address remains the L2 identity. At device admission,
an aliased miss uses the original NAND path until the whole range is resident.
An unaliased read uses the ordinary path regardless of SRAM residency.
Manual programs must fit their reservations into available capacity. Automatic
queue scheduling and retry on capacity pressure belong to Step 7.

Kernel drain and retirement wait for foreground requests. Background requests
share the global cycle clock and survive kernel boundaries. Stream
synchronization and program completion drain all outstanding work.

```sh
cmake --build build-backends --target gtsim_program hbf_prefetch_check
build-backends/hbf_prefetch_check
python3 scripts/hbf/check_prefetch.py --build-dir build-backends
```

The two-GEMM check uses 2F3M with Ramulator2 HBM3 and grouped HBFine. Kernel A
completes at cycle 6,602 while the prefetch is outstanding. Kernel B launches
at that same cycle and transfers 64 KiB through fallback and 192 KiB from
SRAM. The prefetch transfers 256 KiB internally and B issues zero new NAND
sense bytes. Total cycles are 29,236 without prefetch and 14,921 with prefetch
for this controlled case. The backend check covers atomic capacity limits,
base-die spill, shared NAND contention, unaliased reads and deferred release.
All ten Step 4 cases reproduced their prior timings, placements and device
counters with SRAM disabled. CTest was skipped.

**Acceptance**

- [x] An asynchronous prefetch overlaps Kernel A and remains active when Kernel B begins.
- [x] SRAM allocation, completion, fallback access, release, physical contention and the non-prefetch baselines pass focused checks.

## Step 7 — Automatic Tiling and Prefetch Scheduling Passes

**Purpose:** Generate FlashAccel-style prefetch behavior automatically from the structured compute graph, HBF tensor placement, and SRAM capacity.

### Tasks

1. Implement experiment-specific compiler passes under `frontend/src/compile/opt/`, reusing existing compiler ASTs and lowering interfaces.
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

The opt-in pass is `frontend/src/compile/opt/prefetch.py`. It runs on bound
FlowIR before ArchIR lowering. It derives read spans from scheduled memory
operations and models physical page charges per plane and stack. Oversized
launches are split along the CTA loop after checking independent output
tiles. Each CTA retains the full reduction loop. A capacity smaller than one
full-reduction CTA is rejected, and split-K is outside this implementation.

The pass emits a side schedule of enqueue boundaries, consumer aliases and
last-use releases. Trace-dependent weight addresses are gated at their
original launch boundary. The generic runtime admits the FIFO head as SRAM
becomes available and lets kernels use the original read path while waiting.
The ProgIR and standard operator libraries contain no prefetch instructions.

Use `--prefetch-config frontend/asm/flash-accel/residency.json` with
`python -m compile --runtime-dir ...` or the streaming simulation runner.
The residency JSON must match the grouped, aligned memory configuration and
its opt-in SRAM capacities. The Qwen run uses the Step 6 SRAM timing assumptions.

| Check | Result |
| --- | --- |
| Capacity-limited GEMM | Two launches preserve all 4,240 instructions and their CTA-local dependencies. Cycles are 13,622 without prefetch and 19,278 with slicing and prefetch. |
| Qwen B=1 | All 18 kernel DAG hashes match Step 5. 23 reservations are generated. Cycles are 4,291,764 without prefetch and 1,604,395 with prefetch. |
| Qwen HBF traffic | 104,058,240 B resident reads and 9,720,960 B fallback reads. Expert reservations are enqueued only after the preceding routing-related launches. |

The focused planner regression and Step 6 runtime checks pass. Logs and
manifests are under `validation/results/flash-accel/step7/`.

**Acceptance**

- [x] The bound FlowIR and residency configuration produce CTA slices, a FIFO prefetch plan and last-use releases.
- [x] The tiled GEMM preserves compute coverage and dependencies, with runtime capacity admission and non-blocking fallback.

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

### Measured results

All Step 8 runs use H100 132 SM with Ramulator2 HBM3 and HBFine 2F3M.
The packed layout aligns tensors to 4 KiB. The execution-order layout
aligns each tensor to a Megapage. The same SRAM timing assumptions from
Step 6 apply to both prefetch variants.

| Qwen B=1 placement | Cycles | NAND sensed B | SRAM hit rate by HBF read bytes |
| --- | ---: | ---: | ---: |
| All HBM | 623,329 | 0 | — |
| Packed HBF | 4,303,770 | 144,179,200 | — |
| Aligned HBF | 4,291,764 | 141,295,616 | — |
| Packed HBF + prefetch | 1,957,243 | 152,961,024 | 89.5% |
| Aligned HBF + prefetch | 1,604,395 | 142,213,120 | 91.5% |

All four HBF variants share identical compute DAGs and transfer 113,779,200 B
from HBF with zero HBF writes. Both prefetch variants start and complete 23
reservations, peak at 8 outstanding fills and reserve at most 48 MiB.
Aligned prefetch overlaps compute for 1,185,197 cycles and transfers
9,720,960 B through fallback. Mean demand read latency is 8,556 cycles
without prefetch and 2,419 with aligned prefetch. Expert requests enter
the queue after routing, and all releases match their last consumer.

| Microbenchmark | Result |
| --- | --- |
| Two GEMMs | 29,236 → 14,921 cycles. A completes at cycle 6,602, with B starting while the fill is outstanding. B reads 64 KiB via fallback and 192 KiB from SRAM. |
| Same-group contention | A's mean demand read latency rises from 2,650 to 5,198 cycles when the background request uses another page in its command group. A completes 7,642 cycles later. |
| Wide GEMM SRAM sweep | 64 KiB/stack rejects the full-K CTA. 128 KiB/stack splits into two launches at 19,278 cycles. 256 KiB/stack keeps one launch at 13,325 cycles. Untiled demand baseline is 13,622 cycles. |
| Dense Gate/Up/Down | 3,264,844 cycles untiled, 6,524,698 with tiling alone, and 706,044 with tiling plus prefetch. Two full-K slices per projection preserve all 493,260 instructions and dependencies. |

Dense FFN's output-column slices share Megapages. Both sliced runs sense
192 MiB, compared with 96 MiB untiled, while host HBF bytes stay at 96 MiB.
The prefetched run starts and completes 24,576 reservations, peaks at 4,096
outstanding fills and 16 MiB occupied, records five queue-head capacity stalls,
and overlaps compute for 289,452 cycles. These experiments measure simulated
timing with a shared SRAM port per stack and CTA slicing that retains full K.

Per-kernel timings, both-tier traffic, sense group counts, read amplification,
SRAM occupancy, fallback, read latency, queue depth and release cycles are in
`validation/results/flash-accel/step8/summary.json`. Overlap counts GPU cycles
with background work in flight. Fallback traffic and demand latency expose
remaining read cost. The contention control measures an exposed kernel delay
directly. The result directory contains the manifests, input profiles and logs.

**Acceptance**

- [x] Microbenchmarks and the Qwen Block demonstrate asynchronous prefetch, release, physical traffic and overlap.
- [x] All five Qwen ablations, the selected capacity sweep and the Dense FFN controls complete as expected.
- [x] Ten prefetch-disabled GEMM/FA3 baselines retain their cycles, traffic and NAND counts. HBFine and frontend focused checks pass. CTest was skipped.

---

## Implementation Completion Criteria

The reproduction is complete when:

- [x] **Baseline:** GEMM, FA3, and Qwen Block correctly access HBF data using Figure 7-compatible physical placement, with measured read amplification.
- [x] **Prefetch:** Automatic tiling, topology-ordered asynchronous prefetch, HBF SRAM residency, non-blocking fallback, and last-consumer release work across kernel boundaries.
- [x] **Compatibility:** Existing GTSim/HBFine workloads run unchanged when the new capabilities are disabled. No FlashAccel-specific implementation is embedded in generic libraries or backend code.

**Scope limitation:** This project reproduces FlashAccel's mechanism using a Co-located H100 configuration. It does not attempt to match the paper's H200 throughput numbers, OS-level persistent storage management, KV writeback policy, or full multi-GPU serving evaluation.

For every step, the agent must directly report: **changes made, tests performed, measured results, remaining limitations, and whether the acceptance criteria passed.** Do not proceed past an unresolved correctness failure.

## Seeded B=4/8 Row-major comparison

The six successful runs use H100 132 SMs, one Qwen3-30B-A3B block,
BF16, BN64/BM16/BK16, Non-Split decode attention and Grouped MoE.
Hybrid runs use the original FlashAccel 2F3M configuration with 128 KiB
Sense commands and the existing FIFO prefetch/SRAM policy. FullHBM uses
five Ramulator2 HBM stacks.

- B=4 contexts are `[10000, 16384, 24576, 32000]`.
- B=8 contexts are `[10000, 16384, 24576, 32000, 12000, 20000, 28000, 30000]`.
- Seed `20261106` resolves SHA-256 counter-based synthetic logits before IR
  expansion. Top-8 selection and selected-softmax normalization generate a
  saved routing trace. The first four requests share routing across batches.
  B=4 has 32 assignments across 29 experts; B=8 has 64 across 52 experts.
  Router, Top-K and assignment kernels retain their normal timed execution.

|   B | Mode        |     Cycles | HBM read MiB | HBF demand MiB | NAND MiB | Prefetch MiB | Not SRAM MiB (%) |
| --: | ----------- | ---------: | -----------: | -------------: | -------: | -----------: | ---------------: |
|   4 | no-prefetch |  7,298,532 |       162.67 |         297.51 |   418.88 |         0.00 | 297.51 (100.00%) |
|   4 | prefetch    |  4,737,434 |       162.67 |         297.51 |   368.00 |       297.51 |    19.45 (6.54%) |
|   4 | full-hbm    |  1,685,980 |       460.17 |           0.00 |     0.00 |         0.00 |                — |
|   8 | no-prefetch | 10,597,123 |       366.79 |         504.51 |   703.38 |         0.00 | 504.51 (100.00%) |
|   8 | prefetch    |  7,697,977 |       374.38 |         504.51 |   624.50 |       504.51 |   52.92 (10.49%) |
|   8 | full-hbm    |  3,085,732 |       863.85 |           0.00 |     0.00 |         0.00 |                — |

|   B | No-prefetch / prefetch cycles | Prefetch / FullHBM cycles | Late-admitted MiB | Not-admitted MiB | Uncovered MiB |
| --: | ----------------------------: | ------------------------: | ----------------: | ---------------: | ------------: |
|   4 |                        1.541× |                    2.810× |             19.45 |             0.00 |          0.00 |
|   8 |                        1.377× |                    2.495× |             52.92 |             0.00 |          0.00 |

Timeliness is measured over actual HBF demand bytes after L2. The non-SRAM
reads in both prefetched runs belong to admitted regions that were not yet
fully ready. The baseline requires a complete prefetch region before serving
its demand reads from SRAM. The remaining misses concentrate in Gate/Up and
Down; `kernels.csv` records every interval. SRAM-capacity slicing gives
22 launches for B=4 and 26 for B=8, versus 19 without prefetch.

### Flash bandwidth over time

|   B | Mode        | Mean NAND completion GB/s | Peak 10 µs NAND completion GB/s | Mean HBF demand GB/s |
| --: | ----------- | ------------------------: | ------------------------------: | -------------------: |
|   4 | no-prefetch |                    119.16 |                          367.00 |                84.63 |
|   4 | prefetch    |                    161.28 |                         1638.40 |               130.38 |
|   8 | no-prefetch |                    137.80 |                          353.89 |                98.84 |
|   8 | prefetch    |                    168.43 |                         1874.33 |               136.07 |

The curves show bursty prefetch fills followed by SRAM-served demand. NAND
Sense completions, HBF-controller demand completions and prefetch SRAM fills
are sampled separately in 1 µs bins; the plot aggregates 10 µs windows.
NAND output is attributed at command completion, so clustered 128 KiB events
can produce finite-window peaks above the steady array rate. These peaks
represent completion throughput. FullHBM has zero Flash traffic.

The optional sampler is isolated in HBFine `statistics.hpp`, defaults to
disabled and adds no scheduled events. Direct tests compare sampling on/off
for ordinary and grouped reads, preserve completion times and reconcile all
four byte streams. GTSim changes only export generic backend samples; routing
and batch/context generation live in the frontend experiment driver.

Artifacts are under `validation/results/flash-accel/b4-b8-seeded/`, including
`summary.csv`, `kernels.csv`, `bandwidth-1us.csv`, `bandwidth-10us.csv`,
`flash-bandwidth.png`, `flash-bandwidth.svg`, parameter/routing JSON, commands,
simulator logs and runner provenance. All bandwidth byte sums reconcile with
the six completed Program timelines and their traffic counters.

```sh
cmake --build build-backends --target gtsim_program
PYTHONPATH=frontend/src python3 frontend/tools/run_flash_accel_batches.py \
  --out validation/results/flash-accel/b4-b8-seeded \
  --seed 20261106 --planes-per-command 32 --jobs 2
uv run --no-project --python 3.14 --with matplotlib==3.11.2 \
  python frontend/tools/report_flash_accel_batches.py \
  --out validation/results/flash-accel/b4-b8-seeded
```

## Rotating SRAM buffers for B=4/8

The compiler prefetch pass now accepts `buffer_count=2` or `4`. It divides
the configured 64 MiB prefetch SRAM into equal fixed slots, including both
per-plane and per-stack quotas. Two buffers use 16 KiB/plane plus
4 MiB/stack each (32 MiB/slot); four use 8 KiB/plane plus 2 MiB/stack
(16 MiB/slot). The shared SRAM/NAND ports retain their configured rates.

The FlowIR pass slices complete-K CTAs using the per-buffer budget and
assigns tile epochs round-robin to slots. The next tile for a slot is
queued after that slot's previous consumers retire. The backend allocator
retains the old lease until its reads and outstanding fill drain, preventing
overwriting live data even if prefetch is late. Routing availability gates
and the existing readiness/fallback rule apply. Qwen selects the buffering
parameter; slicing and ownership are workload-independent.

All ten cases were re-executed with the same contexts, seeded routes,
Row-major layout, 128 KiB Sense geometry and five-stack FullHBM control.
The six original controls reproduced their cycles and per-kernel DAG hashes
exactly. Native opcode counts match across every profile within each batch.

|   B | Mode             |     Cycles | HBM read MiB | HBF demand MiB | NAND MiB | Prefetch MiB | Not SRAM MiB (%) |
| --: | ---------------- | ---------: | -----------: | -------------: | -------: | -----------: | ---------------: |
|   4 | no-prefetch      |  7,298,532 |       162.67 |         297.51 |   418.88 |         0.00 | 297.51 (100.00%) |
|   4 | prefetch         |  4,737,434 |       162.67 |         297.51 |   368.00 |       297.51 |    19.45 (6.54%) |
|   4 | prefetch-2buffer |  2,190,072 |       162.67 |         297.51 |   310.12 |       297.51 |     6.26 (2.10%) |
|   4 | prefetch-4buffer |  2,058,061 |       162.67 |         297.51 |   317.88 |       305.51 |     3.66 (1.23%) |
|   4 | full-hbm         |  1,685,980 |       460.17 |           0.00 |     0.00 |         0.00 |                — |
|   8 | no-prefetch      | 10,597,123 |       366.79 |         504.51 |   703.38 |         0.00 | 504.51 (100.00%) |
|   8 | prefetch         |  7,697,977 |       374.38 |         504.51 |   624.50 |       504.51 |   52.92 (10.49%) |
|   8 | prefetch-2buffer |  3,025,905 |       355.56 |         504.51 |   512.25 |       504.51 |     4.79 (0.95%) |
|   8 | prefetch-4buffer |  3,560,284 |       350.38 |         504.51 |   527.00 |       512.51 |     3.29 (0.65%) |
|   8 | full-hbm         |  3,085,732 |       863.85 |           0.00 |     0.00 |         0.00 |                — |

|   B | Buffers | Original prefetch / buffered cycles | Buffered / FullHBM cycles | Peak total SRAM MiB | Maximum slot occupancy MiB |
| --: | ------: | ----------------------------------: | ------------------------: | ------------------: | -------------------------: |
|   4 |       2 |                              2.163× |                    1.299× |               60.00 |                      30.00 |
|   4 |       4 |                              2.302× |                    1.221× |               57.00 |                      16.00 |
|   8 |       2 |                              2.544× |                    0.981× |               60.00 |                      30.00 |
|   8 |       4 |                              2.162× |                    1.154× |               60.00 |                      16.00 |

All buffered non-SRAM demand bytes are admitted-but-not-ready reads; both
not-admitted and uncovered demand remain zero. The rates use HBF demand bytes
after L2. Full-K slicing with four buffers duplicates 8 MiB of QKV prefetch
page coverage, which appears in the actual prefetch and NAND traffic.

Four buffers reduce misses further but are slower for B=8. Gate/Up slices
have at most 48 CTAs versus 120 with two buffers; Gate/Up time rises from
1,141,740 to 1,490,895 cycles. Total launch counts rise from 34 to 55.
The smaller slices expose less CTA parallelism. Thus the lowest miss rate
does not select the lowest measured stream latency. These are GTSim results
for this routing trace, without hardware calibration of the new buffering policy.

### Buffered bandwidth and occupancy

|   B | Mode             | Mean NAND completion GB/s | Peak 10 µs NAND completion GB/s | Mean HBF demand GB/s |
| --: | ---------------- | ------------------------: | ------------------------------: | -------------------: |
|   4 | no-prefetch      |                    119.16 |                          367.00 |                84.63 |
|   4 | prefetch         |                    161.28 |                         1638.40 |               130.38 |
|   4 | prefetch-2buffer |                    294.00 |                         1756.36 |               282.04 |
|   4 | prefetch-4buffer |                    320.67 |                         1572.86 |               300.13 |
|   8 | no-prefetch      |                    137.80 |                          353.89 |                98.84 |
|   8 | prefetch         |                    168.43 |                         1874.33 |               136.07 |
|   8 | prefetch-2buffer |                    351.47 |                         1743.26 |               346.16 |
|   8 | prefetch-4buffer |                    307.32 |                         1664.61 |               294.20 |

The bandwidth plot overlays no-prefetch, the original shared-pool prefetch,
two buffers and four buffers. It uses the same 1 µs completion samples and
10 µs plot windows as the previous experiment. The occupancy plot and CSV
record actual kernel-completion snapshots, including per-slot epoch IDs and
reserved bytes. Runtime quota and compiler release/requeue checks passed.
Twenty-two frontend tests and the direct SRAM capacity/alias/lease test passed;
CTest was skipped.

Artifacts are under `validation/results/flash-accel/b4-b8-buffered/`:
`summary.csv`, `kernels.csv`, `bandwidth-1us.csv`, `bandwidth-10us.csv`,
`flash-bandwidth.png/.svg`, `sram-buffers.csv/.png/.svg`, `validation.json`,
generated inputs/programs, simulation logs and provenance. Earlier results
remain in their original directories.

```sh
cmake --build build-backends --target gtsim_program hbf_prefetch_check
build-backends/hbf_prefetch_check
PYTHONPATH=frontend/src python3 frontend/tools/run_flash_accel_batches.py \
  --out validation/results/flash-accel/b4-b8-buffered \
  --seed 20261106 --planes-per-command 32 --buffers 1 2 4 --jobs 2
uv run --no-project --python 3.14 --with matplotlib==3.11.2 \
  python frontend/tools/report_flash_accel_batches.py \
  --out validation/results/flash-accel/b4-b8-buffered \
  --baseline validation/results/flash-accel/b4-b8-seeded
```

### B=4 reduced SRAM and K-panel double buffering

The Row-major, 128 KiB FlashAccel sweep completed 30 new cases: 15 each at
8 and 4 MiB per HBF stack. All 15 matching five-stack FullHBM controls were
reused. Batch contexts are `[10000, 16384, 24576, 32000]`, routing seed is
`20261106`, and the two HBF stacks retain their original timing parameters.
Groups A/B share `BM=16`, `BN=32/64/128/256`, and `BK=32/64/128/256`, excluding
`BN256/BK256` for the SMEM limit. Group C stays at `16/64/16`.

| Case             | Fastest BN/BK | Best cycles | Worst cycles | HBM read MiB | HBF demand MiB | NAND sense MiB | SRAM miss |
| ---------------- | ------------- | ----------: | -----------: | -----------: | -------------: | -------------: | --------: |
| FullHBM (reused) | 256/128       |   1,240,994 |    2,212,681 |       460.17 |              0 |              0 |         — |
| 8 MiB/stack      | 32/256        |   1,561,200 |   87,396,475 |       162.67 |         297.51 |         407.00 |    95.47% |
| 4 MiB/stack      | 32/256        |   1,568,953 |  109,060,555 |       162.67 |         297.51 |         351.50 |    98.64% |

Traffic columns describe each row's fastest case. FullHBM at the same
`BN32/BK256` takes 2,104,208 cycles. Against the independently tuned FullHBM
minimum, the two DoubleBuffer minima take 25.8% and 26.4% more cycles.
The slowest candidates issue 74,764 and 79,176 Flash sense commands, versus
3,256 and 2,812 in the fastest cases, despite the same HBF demand bytes.
Most of this extra traffic is repeated sensing during grouped expert GEMMs.

The compiler groups actual native TMA reads into K panels within one-CTA-per-SM
waves. The simulator retires a slot after its last TMA consumer and admits the
next panel into that slot once outstanding fills release it. Requests that
arrive early use the normal demand path. Expert windows are enqueued after
TopK; the full-K compute DAG and 19 launches per block are retained. The
8 MiB configuration uses 8 KiB per plane plus 2 MiB per stack; 4 MiB uses
4 KiB plus 1 MiB. In the latter, whole plane pages alternate slot ownership
rather than splitting a 4 KiB page. Capacity is local to planes and stacks,
so the observed peak can fall below nominal capacity. The inactive write
quota is set to the minimum accepted value in these read-only profiles.

The previous 32 MiB run used whole-K prefetch groups, so its timings cannot
isolate the effect of capacity alone. The 4 MiB `BN256/BK128` case was retried
with a 250-million-cycle execution guard after reaching the default 100
million; its original timeout log is retained. No hardware timing changed.

```sh
cmake --build build-backends --target gtsim_program hbf_prefetch_check
PYTHONPATH=frontend/src python3 frontend/tools/run_sram_tiling_search.py \
  --out validation/results/flash-accel/b4-sram-panel-sweep \
  --max-cycles 250000000
PYTHONPATH=frontend/src uv run --no-project --python 3.14 --with matplotlib==3.11.2 \
  python frontend/tools/report_sram_tiling_search.py \
  --out validation/results/flash-accel/b4-sram-panel-sweep
```

The runner uses at most two jobs and links the existing controls from
`validation/results/flash-accel/b4-shared-ab-tiling`. The output contains
`tiling-cycles-{8,4}mib.png/.svg`, `flash-bandwidth.png/.svg`, summary and
per-kernel CSVs, slot lifetimes, and 10 µs bandwidth series. Raw 1 µs bins
are in each run's `result.json`; whole-stream Flash totals include the
final memory drain. These are GTSim timing estimates.
