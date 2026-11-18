# Experiment 002 — Tiling, Ordering & Prefetch for Q Projection

**Experiment ID:** `002-tiling-order-Qmat`  
**Repository:** `Hypersonic-cpu/GTSim`  
**Branch:** `sim-validation`  
**Reviewed commit:** `24da00978ca7e77474e7018f2886d13120c9fa3c`

## 1. Objective

Explore kernel-level execution dataflows for a single Q-projection GEMM on Full-HBM and 4F1M GPUs.

The experiment must use **enumeration and simulation-based sweep**, without constraint solvers, genetic algorithms or learned optimization models.

The search must operate at the **FlowIR level**, generating genuinely different GEMM execution schedules rather than changing metadata without changing the lowered instructions.

The objective is to understand the interaction among:

- Compute tile geometry and Split-K parallelism
- Persistent CTA task ordering
- HBF weight prefetch granularity
- Prefetch lookahead and SRAM residency
- GPU compute–memory overlap

**This experiment must not assume FlashAccel-style N-slab partitioning or multiple partial-GEMM kernel launches.**

## 2. Isolated Workload

Create one independent `.progir` defining only:

```text
Q = X_norm @ W_q
```

Use the Qwen3-235B-A22B Q-projection dimensions:

| Property | Value |
|---|---|
| M | 1 |
| N | 8192 = 64 Q heads × 128 |
| K | 4096 |
| Input | `X_norm[1, 4096]`, BF16 |
| Weights | `W_q[4096, 8192]`, BF16 |
| Output | `Q[1, 8192]`, BF16 |
| Weight footprint | 64 MiB |
| GPU SMs | 132 |
| Data layout | Row-major (pages should spread over planes, not all in one plane) |

The weights are a **standalone contiguous Q tensor**, not the entire 72 MiB QKV tensor and not an implicit slice with the original QKV row stride. This distinction must be reflected consistently in address generation and memory allocation.

`X_norm` is an external HBM input. The experiment does not execute RMSNorm, K/V projection, RoPE, Attention, Router or MoE.

## 3. Hardware Configurations

**Full-HBM reference**

- `configs/memconf/0F5M_channels.yaml`
- Five HBM stacks, aggregate read bandwidth 3100 GB/s
- `W_q`, input and output in HBM
- No HBF prefetching
- Run one validated C++-reference-equivalent persistent WS GEMM configuration, with `BM=64`, `BN=256`, `BK=64`, `SplitK=1`

**4F1M search**

- `configs/memconf/4F1M_flashaccel_32m.yaml`
- Four HBF stacks + one HBM stack
- `W_q` in HBF; input/output in HBM
- HBF SRAM: 32 MiB per stack, 128 MiB total
- Preserve the existing NAND geometry, physical address mapping, SRAM allocator, bandwidths and latencies

Both configurations must use the same GPU architecture and equivalent GEMM execution semantics. **No HBF→HBM weight staging in Experiment 002.**

## 4. GEMM Execution Baseline: WS + Persistent CTA

The current `small_m_gemm.flowir` must support the proper WS execution model used by the C++ H100 GEMM reference:

- **12 warps per CTA**
- Producer warpgroup: warps 8–11, with TMA issued by the designated producer warp
- Matrix consumer warps: 0–7
- WGMMA instruction geometry: `m64n64k16`
- Software pipeline depth: **2**, with two SMEM input stages
- Correct producer–consumer dependencies, stage reuse, acquire/release and cross-wave hazards

Because logical `M=1`, use **BM=64**, with only one live output row. Other physical rows must be masked or zero-padded correctly. The implementation must not read input rows outside the logical tensor or write invalid output rows.

Every GEMM candidate uses a **persistent CTA work loop**, not one newly launched CTA per logical compute task.

The first experiment fixes WS and pipeline organization. They are **not DSE dimensions**.

## 5. Compute Tiling and Split-K

Define the compute parameters:

```text
BM      = 64                 # fixed, minimum legal physical tile
BN      ∈ {128, 256}
BK      ∈ {32, 64, 128}
SplitK  ∈ {1, 2, 4, 8}
```

All candidates must pass actual TMA, WGMMA, SMEM and register feasibility checks. Unsupported configurations must be rejected explicitly, not silently mapped to different parameters.

Split-K divides the **K dimension**, not the N dimension:

```text
K_per_split = K / SplitK
K_iteration_count = K_per_split / BK
```

A logical compute task is identified by:

```text
(sp_k, m_tile, n_tile)
```

The number of compute tasks is:

```text
numM = ceildiv(M, BM) = 1
numN = ceildiv(N, BN)
numTasks = SplitK × numM × numN
```

For this workload, the meaningful task space is two-dimensional:

```text
(sp_k, n_tile)
```

Each task performs its own complete `K_per_split` reduction using the 2-stage pipeline.

If `SplitK=1`, the task can write directly to the corresponding BF16 output interval.

If `SplitK>1`:

- Independent tasks produce FP32 partial results for the same N interval.
- Partial results must be race-free.
- A dependent reduction combines all Split-K results.
- Convert to BF16 only after the reduction.
- All reduction computation, storage traffic and synchronization overhead must be included in execution time.

The Split-K reduction may require a separate dependent kernel. It must not be modeled as a free operation or as an unsupported global CTA barrier inside the GEMM kernel.

## 6. CompOrdering — Persistent CTA Task Mapping

**CompOrdering defines a permutation of logical compute tasks, followed by their assignment to a fixed set of persistent CTAs.**

Let:

```text
T = SplitK × numN
P = min(NUM_SMS, T)  # baseline: at most one persistent CTA per SM
```

For an ordered task list:

```text
ordered_tasks = [task0, task1, ..., task(T-1)]
```

Persistent CTA `c` executes:

```text
task(c)
task(c + P)
task(c + 2P)
...
```

Equivalently:

```text
task_rank = wave × P + cta_id
```

This is the common and precise definition for all ordering candidates. It preserves the fixed CTA count while changing the sequence of `(sp_k, n_tile)` assignments.

### Required ordering candidates

**A. `n_major` — Split-K varies fastest**

```text
for n in range(numN):
    for s in range(SplitK):
        emit(s, n)
```

For `SplitK=2`, the split-ID sequence is:

```text
0 1 | 0 1 | 0 1 | ...
```

This is equivalent to the C++ GEMM's current output-tile-first Split-K flattening.

**B. `split_major` — N varies fastest**

```text
for s in range(SplitK):
    for n in range(numN):
        emit(s, n)
```

For `SplitK=2`:

```text
0 0 0 ... | 1 1 1 ...
```

**C. `n_grouped(G)` — N groups followed by Split-K**

```text
for n0 in range(0, numN, G):
    for s in range(SplitK):
        for n in range(n0, min(n0+G, numN)):
            emit(s, n)
```

Required `G` candidates: `{2, 4, 8}`, whenever meaningful.

For `G=2`, `SplitK=2`:

```text
0 0 1 1 | 0 0 1 1 | ...
```

This directly captures the ordering distinction in the research hypothesis.

**D. `staggered` — Bank/phase-diversifying order**

For each successive `n`, rotate the split-ID order using `(s+n) mod SplitK`, preserving one occurrence of every `(s,n)` pair.

This provides a deterministic nontrivial ordering for investigating HBF locality and load balance without changing the computation.

Every ordering must be a bijection over the same logical task set. No duplicate or missing tasks.

An ordering does **not** imply that all CTAs globally complete one region before another. Such a barrier is not assumed in this experiment.

## 7. Prefetch Tiling — Independent of Compute Tiling

Introduce two independent logical prefetch dimensions:

```text
PfBN : columns of W_q per prefetch region
PfBK : rows of W_q per prefetch region
```

A prefetch region is a logical rectangle:

```text
W_q[k0 : k0 + PfBK, n0 : n0 + PfBN]
```

Suggested initial candidates:

```text
PfBN ∈ {256, 1024, 8192}
PfBK ∈ {128, 1024, 4096}
```

The following are **not required**:

```text
PfBN == BN
PfBK == BK
PfBN divisible by BN
PfBK divisible by BK
```

A compute tile may require data from several prefetch regions, and one prefetch region may serve several compute tiles.

The logical rectangle must be translated into its exact set of physical NAND pages, respecting:

- Row-major tensor strides
- NAND page granularity
- HBF megaplane/group mapping
- Per-plane SRAM capacity
- Base-die SRAM spill capacity

SRAM capacity is checked using **unique mapped physical pages**, not `PfBN × PfBK × element_bytes` alone.

Overlapping logical prefetch regions may touch the same physical page. These overlaps must be explicitly tracked to prevent incorrect early release, duplicate ownership or unaccounted rereads.

The prefetch organization must not require additional GEMM kernel launches.

## 8. Prefetch Ordering and PfAheadDist

Keep the prefetch policy simple and deterministic for Experiment 002.

First derive the ordered prefetch region list from the **first use of each region in the CompOrdering task sequence**. Ties follow physical HBF address/hyperpage order.

Define:

```text
PfAheadDist ∈ {0, 1, 2, 4}
```

`PfAheadDist` measures lookahead in **prefetch regions**, not cycles or bytes.

### `PfAheadDist = 0`

Disable explicit HBF SRAM prefetch:

- No QueuePrefetch requests
- No speculative SRAM residency reservation
- Normal demand reads remain enabled
- Existing ordinary NAND/page-buffer behavior remains unchanged

### `PfAheadDist = d > 0`

For prefetch regions `G0, G1, ..., Gq`:

- At kernel launch, enqueue the first `d` regions.
- When the first consumer of `Gi` starts its memory-access work, enqueue `Gi+d`, if it exists and has not already been enqueued.
- Enqueueing is nonblocking and uses the existing bounded HBF prefetch engine.
- Requests that cannot obtain SRAM resources remain subject to the existing queue/admission behavior.
- A consumer may use prefetched data only after that data is ready. Otherwise it follows the explicitly modeled demand fallback.
- Release a page/reservation only after its final relevant memory consumer no longer requires it, accounting for concurrent CTAs and overlapping groups.

The trigger is based on **actual runtime issue events**, not only a statically estimated timestamp. Different CTAs may trigger multiple regions in the same cycle; this is legal and must be simulated faithfully.

Prefetch issue time, ready time, first-use time, useful lead time and late/demand-fallback events must be observable.

No fictitious prefetch bandwidth, zero-cost host command path or free pre-launch warmup may be introduced.

## 9. FlowIR-Level Search Contract

All DSE decisions must be represented by a reusable, parameterized GEMM FlowIR and its associated memory-access metadata.

For every candidate, specialization/lowering must determine:

```text
BM, BN, BK, SplitK
CompOrdering, GroupSize
PfBN, PfBK, PfAheadDist
```

The generated execution must correctly reflect:

- Persistent CTA loops and task-ID decoding
- Split-K-specific weight offsets and partial-result destinations
- CTA/wave mapping and valid-task guards
- Two-stage SMEM buffering
- Dedicated TMA producer and WGMMA consumer warps
- Correct K-loop dependencies and final-store conditions
- Physical prefetch regions and consumer relationships
- Buffer reservation, readiness and final release

Parameters are applied through structured compiler specialization/lowering, **not by editing a separate hand-written FlowIR file for each sweep case**.

The original GEMM templates may be reused, but a candidate must produce actual differences in Native IR or prefetch execution when its parameters change.

Current FlashAccel launch-level N-slab partitioning must not be implicitly applied to this isolated GEMM.

Intra-kernel prefetch triggers and SRAM lifetime management must be supported by the actual simulator execution path. Existing launch-boundary-only QueuePrefetch behavior is not sufficient to implement `PfAheadDist` as specified.

## 10. Correctness and Legality Requirements

All successful candidates must satisfy:

1. **Numerical correctness:** compare the complete output against a high-precision GEMM reference, using documented BF16/WGMMA tolerances.
2. **Complete coverage:** every logical output element is produced; no duplicate writes or missing regions.
3. **Split-K correctness:** no lost reductions, early BF16 rounding, stale partial results or cross-CTA races.
4. **Persistent CTA correctness:** CTA/wave/task mapping is bijective and terminates.
5. **WS correctness:** producer and consumer warp sets are distinct, with correct data-ready and buffer-reuse dependencies.
6. **Pipeline correctness:** two-stage SMEM reuse cannot overwrite data still being consumed, including across waves.
7. **Prefetch correctness:** no consumer observes unready data, no premature SRAM release, and physical capacity constraints are enforced.
8. **Memory correctness:** no out-of-bounds accesses for logical `M=1`, no incorrect weight strides, and no missing DMA/reduction costs.

All conditional paths must be checked explicitly, especially:

```text
SplitK = 1 versus >1
PfAheadDist = 0 versus >0
First/last K iteration
First/last persistent wave
Tail task when T is not divisible by P
Different BN/BK/PfBN/PfBK boundaries
```

Reject illegal configurations with a specific recorded reason; never silently fall back to a different kernel geometry.

## 11. Sweep and Measurements

**All candidates are enumerated; there is no solver.**

Deduplicate configurations whose parameters are irrelevant, such as prefetch tiling choices when `PfAheadDist=0`, or ordering choices that generate the same task permutation.

The Full-HBM reference is run independently and serves as the fixed performance reference. The main sweep runs on 4F1M.

Each candidate's output must include:

- Effective parameter set and source commit
- Compile/validation status and rejection reason
- GEMM makespan and total GPU cycles
- Logical compute-task count, persistent CTA count and wave count
- Number of WGMMA/TMA operations
- Split-K partial-output and reduction cost
- HBF NAND sense count and media-read bytes
- HBF prefetch bytes, resident-read bytes and demand-read bytes
- SRAM occupancy, admission stalls and allocation failures
- Prefetch useful lead time, late-prefetch and demand-fallback counts
- GPU memory-wait cycles, TMA/WGMMA activity and achieved bandwidth

Record any unsupported counter as unavailable rather than substituting an inferred value.

Store results under:

```text
validation/results/002-tiling-order-Qmat/
```

The experiment must preserve sufficient provenance to reproduce each candidate and inspect its generated FlowIR/Native IR.

## 12. Expected Findings

This experiment should answer:

**F1 — CompOrdering:** How does the linearization of `(sp_k,n_tile)` change HBF page locality, prefetch effectiveness, memory parallelism and GPU execution time?

**F2 — Prefetch/Compute Granularity:** When is it beneficial for `PfBN/PfBK` to differ from `BN/BK`?

**F3 — Split-K:** Can increased persistent-CTA parallelism compensate for extra reduction traffic and altered HBF access patterns?

**F4 — Prefetch Distance:** What lookahead provides sufficient data readiness without excessive SRAM occupancy, page rereads or queue contention?

**F5 — Joint Optimization:** Does the best combination differ from independently choosing GEMM tiling, CTA ordering and prefetch policy?

The desired output is **a set of measured findings and reproducible optimal candidates**, not a manually hardcoded FlashAccel-like scheduling policy.

The experiment is successful when it can compare multiple legally distinct FlowIR dataflows under identical hardware resources and explain their performance differences using simulator measurements.
