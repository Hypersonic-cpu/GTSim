# Task: GEMM Tiling Search and Tile-major Weight Layout for FlashAccel

## Objective

Continue development on `Hypersonic-cpu/GTSim`, branch `related/flash-accel`.

We need two improvements to the current FlashAccel reproduction:

1. Manually enumerate and search a small set of GEMM compute-tiling candidates for Qwen3-30B-A3B.
2. Implement and evaluate a 4 KiB Tile-major HBF weight layout inspired by TileLens.

These improvements will eventually provide a strong, fair FlashAccel baseline for comparison against our own research.

**Do not implement an automatic tuning framework or introduce a new prefetch policy.** Run the explicitly defined candidate search.

References:

- TileLens: https://arxiv.org/pdf/2607.04031 — Section 3, especially Figure 7 and its address-translation equations.
- FlashAccel v2: https://arxiv.org/pdf/2607.10186v2 — Figure 7 (Megaplane placement) and Figure 12 (Prefetch).
- Current implementation: `docs/related/flash-accel/IMPL.md`.

## 1. Implementation Constraints

- Keep existing Row-major FlashAccel results reproducible.
- Do not modify standard `frontend/lib/*.algoir` or `frontend/lib/*.flowir`.
- Generate independent GEMM FlowIR schedules or parameterized schedule variants for experiments.
- Introduce generic Tensor Layout metadata and address-generation support where necessary. Avoid FlashAccel-specific conditions in shared GTSim/HBFine components.
- Do not modify GPU or HBF timing parameters as part of this experiment.
- Preserve the existing HBF physical geometry, SRAM capacities, and prefetch policy.
- Use H100, 132 SMs, Ramulator2 HBM3, and grouped HBFine for the hybrid experiments. The hybrid topology remains 2F3M. Use five Ramulator2 HBM stacks for all FullHBM experiments.
- Use the same model parameters, BF16 dtype, context length, routing trace, and input tensors across comparisons.
- No fabricated performance estimates. Report simulator measurements only.
- Do not create a separate progress log. Report progress and results directly to the user after each stage.

## 2. Qwen GEMM Groups and Compute-Tiling Candidates

Use `frontend/asm/qwen3/b1.json` as the initial workload configuration.

The GEMMs are:

| Group | Operation                        | GEMM `(M,N,K)`  | Weight       |
| ----- | -------------------------------- | --------------- | ------------ |
| A     | QKV Projection                   | `(1,5120,2048)` | 20 MiB       |
| A     | MoE Gate/Up, per selected expert | `(1,1536,2048)` | 6 MiB/expert |
| B     | Attention Output Projection      | `(1,2048,4096)` | 16 MiB       |
| B     | MoE Down, per selected expert    | `(1,2048,768)`  | 3 MiB/expert |
| C     | Router Projection                | `(1,128,2048)`  | 0.5 MiB      |

For MoE, `M=8` is the total number of assignments across eight selected experts. Each expert has one input token in this B=1 trace. Do not treat the assignments as a single dense GEMM sharing one weight matrix.

### Candidate schedules

Use `(BM, BN, BK)` notation.

| Group | Candidates                                 |
| ----- | ------------------------------------------ |
| A     | `(16,64,16)`, `(16,128,16)`, `(16,256,16)` |
| B     | `(16,64,16)`, `(16,128,16)`, `(16,256,16)` |
| C     | `(16,64,16)`, `(16,128,16)`                |

This produces **3 × 3 × 2 = 18 full-block tiling combinations**, not an independent search for every matrix.

For the initial search:

- Fix `BM=16`, `BK=16`.
- Vary only `BN`.
- Preserve full-K accumulation semantics.
- Preserve the selected experts and logical outputs.
- Keep existing FlowIR operator implementations unless changes are required to support the candidate shape.
- Correctly update CTA counts, matrix tile dimensions, loads, stores, scratch/SMEM requirements, and dependencies.
- Reject candidates exceeding hardware resource constraints. Do not silently truncate tiles or change numerical semantics.
- Do not introduce Split-K in this experiment.

Check that a modified candidate actually changes the lowered Compute Tile and memory access pattern. Merely changing a prefetch window or splitting a kernel is not equivalent to changing GEMM tiling.

## 3. TileLens-style Tile-major HBF Layout

Implement a new optional **2D Tile-major physical layout for read-only HBF GEMM weights**.

### 3.1 Layout definition

Use the initial BF16 Memory Tile shape:

\[
(a,b)=(16,128)
\]

Each Memory Tile contains:

\[
16\times128\times2=4096\text{ bytes}
\]

Therefore, **one contiguous 4 KiB NAND Page corresponds to one complete 16×128 rectangular region of the logical matrix**.

This is a proposed TileLens-style shape for our workload, not a claim that every TileLens experiment uses `16×128`.

Implement the TileLens Figure 7 convention: **column-major ordering of Memory Tiles and column-major ordering of elements within a Memory Tile**.

For a matrix \(W[K,N]\), let:

\[
T_k=\lfloor k/16\rfloor,\quad T_n=\lfloor n/128\rfloor
\]

\[
k'=k\bmod16,\quad n'=n\bmod128
\]

The physical byte offset is:

\[
\boxed{
offset(k,n)=
\left(T_n\cdot\frac K{16}+T_k\right)\cdot4096
+2(n'\cdot16+k')
}
\]

Assume `K` is divisible by 16 and `N` by 128. The current target GEMM shapes satisfy these conditions.

The two levels of ordering matter:

- **Inter-tile:** Sweep K tiles first, then advance to the next N tile.
- **Intra-tile:** Store elements column-major inside each 16×128 Memory Tile.

Do not substitute Row-major ordering inside or between tiles without explicitly defining and testing it as a different layout variant.

### 3.2 Physical placement in HBF

Keep FlashAccel's existing placement mechanism:

```text
Logical Weight Matrix [K,N]
       |
       v
2D Tile-major encoding
       |
       v
Sequence of contiguous 4 KiB Memory Tiles
       |
       v
Group 32 consecutive 4 KiB Pages
into one 128 KiB Megapage
       |
       v
FlashAccel round-robin placement
across HBF Megaplanes
```

Tile-major changes the correspondence between logical `(k,n)` and physical byte offset. It must **not** change the meaning of a Megaplane or make one 4 KiB page independently senseable when the hardware requires a 128 KiB grouped Sense.

Preserve the existing execution-order placement of different weight tensors.

**MoE special requirement:** Treat each expert as an independent `K×N` logical submatrix. Tile-major encode each expert separately, then concatenate expert storage regions in expert-ID order. Do not apply the inter-tile ordering across the entire combined `[E*K,N]` tensor, because that would interleave different experts and destroy expert-level physical locality.

Normalization weights and other non-GEMM tensors should retain their existing layout.

### 3.3 Address generation

The logical GEMM semantics remain unchanged:

\[
C[m,n]=\sum_k A[m,k]B[k,n]
\]

But physical Weight Load addresses must use the Tile-major mapping.

Extend the generic tensor layout description to distinguish at least:

- `RowMajor`
- `TileMajor(tile_k=16, tile_n=128, intra_order=column_major, inter_order=column_major)`

Propagate this layout information through the relevant IR stages to the memory-address generation used by GEMM loads.

**The translation must occur before L2 line generation/coalescing**, not only at HBF device admission. Otherwise, a logical 128 B cache line could incorrectly be treated as a physically contiguous 128 B region after nonlinear remapping.

Keep the existing normal Row-major behavior unchanged.

Since TileLens-HW proposes TMA address-translation support, while GTSim does not currently implement that exact extension, explicitly state how the simulator represents this functionality. Do not incorrectly claim zero-cost transparent hardware support; report any omitted address-generation or TMA conversion overhead.

### 3.4 Layout correctness tests

Create small deterministic tests, including an `8×8` matrix with a smaller illustrative tile and a BF16 GEMM weight matrix with `16×128` Memory Tiles.

Verify:

1. Logical coordinates map to the expected physical byte offsets.
2. Every logical element maps exactly once.
3. There are no overlapping or missing physical storage regions.
4. A full 16×128 tile occupies exactly one aligned 4 KiB Page.
5. GEMM access coverage and dependencies remain equivalent to Row-major.
6. MoE expert-ID mapping remains unchanged.
7. NAND Megapage placement and grouped Sense behavior remain correct.
8. The default Row-major configuration reproduces existing baselines.

Because GTSim is primarily a timing simulator, structural/read-address equivalence alone is not numerical correctness. If necessary, use a small independent CPU reference to verify the layout encoder/decoder and GEMM results.

## 4. Experiments

### 4.1 Joint search

Use two physical layouts:

- **R:** Current Row-major, with FlashAccel 128 KiB Megapage placement.
- **T:** New 16×128 Tile-major, with the same FlashAccel Megapage placement.

Use all 18 compute-tiling combinations for each layout.

For each combination, run:

- **No Prefetch**
- **FlashAccel Prefetch**, keeping the existing FIFO queue, SRAM residency rules, and fallback behavior.

Total maximum:

\[
18\times2\times2=\boxed{72\text{ Qwen Block runs}}
\]

No automatic search algorithm is required. A simple deterministic enumerator is sufficient.

Retain the existing `(16,64,16)` schedule and Row-major layout as the original reference point. Run the original tiling as the matched FullHBM control with five HBM stacks and no HBF or prefetch.

### 4.2 Additional focused tests

For the best candidates, run separate GEMM-level experiments for:

- QKV Projection.
- Attention Output Projection.
- Router Projection.
- Gate/Up and Down for each selected expert.
- Complete Qwen B=1 Block.

Pay particular attention to:

- Whether increasing `BN` decreases NAND Sense traffic but reduces GPU CTA-level parallelism.
- Whether the new Memory Tile shape works better with `BN=128/256` than `BN=64`.
- Whether Tile-major changes the number and distribution of active Megaplanes.
- Whether repeated Sense of the same Megapage is reduced or increased.
- Whether Prefetch remains timely after the access order changes.
- Whether expert accesses still activate the intended physical groups.

Do not change Prefetch Queue policy merely to improve one layout. This experiment is intended to isolate layout and compute-tiling effects.

### 4.3 Metrics

Report at least:

- End-to-end Block cycles.
- Per-GEMM cycles.
- HBF host-demand bytes.
- NAND sensed bytes and grouped Sense count.
- Unique Megapages and repeated Sense count.
- SRAM resident-read bytes and fallback-read bytes.
- SRAM occupancy, pending Prefetch depth, and completion timing.
- Active Megaplanes and their load imbalance.
- GPU CTA count, resource usage, and scheduling behavior.

Distinguish:

\[
A\_\text{kernel}=
\frac{\text{NAND sensed bytes}}
{\text{HBF demand-read bytes}}
\]

from per-tile cold-access amplification and repeated-Sense counts.

If a prefetch run has extra NAND traffic, do not hide it in an SRAM hit-rate metric.

## 5. Result Presentation and Fairness

Produce a compact result table for all 18 tiling combinations under each of the four experiment conditions:

| Layout | Prefetch | Group A BN | Group B BN | Group C BN | Cycles | NAND bytes | Repeated Senses |
| ------ | -------- | ---------: | ---------: | ---------: | -----: | ---------: | --------------: |

Then identify the best configurations using **measured end-to-end cycles** as the primary objective, with correctness and hardware capacity constraints mandatory.

Report these comparisons separately:

1. **Tiling-only gain:** best tiling versus original tiling, same layout and prefetch setting.
2. **Layout-only gain:** Tile-major versus Row-major, same compute tiling and prefetch setting.
3. **Prefetch gain:** Prefetch versus No-Prefetch, same layout and compute tiling.
4. **Combined gain:** best new configuration versus original Row-major FlashAccel + Prefetch.
5. **Remaining HBM gap:** best HBF configuration versus the five-stack FullHBM control.

Avoid claiming the combined improvement comes from only one optimization.

### Fair-baseline requirement

These experiments extend the FlashAccel implementation beyond its specified weight-layout mechanism. **Do not label Tile-major as the original FlashAccel layout.**

Maintain:

- `FlashAccel-original`: original Row-major + Megapage placement + original Prefetch.
- `FlashAccel-tuned`: same FlashAccel Prefetch policy with optimized compute tiling.
- `TileMajor + FlashAccel`: Tile-major with the same Prefetch policy.
- `TileMajor + FlashAccel-tuned`: Tile-major plus optimized compute tiling.

For future comparisons against our proposed research, the FlashAccel baseline must receive the same eligible layout and GEMM scheduling optimizations as Ours unless the experiment explicitly isolates those optimizations.

## 6. Validation and Reporting

Do not overwrite the existing Step 4–8 results. Save new results, configurations, generated schedules, and reproducible commands in a dedicated experiment directory.

Run existing focused tests and relevant regression tests after the changes. Distinguish tests actually executed from tests not executed.

If a candidate cannot be lowered or simulated because of unsupported TMA/SMEM behavior, report the concrete limitation and fix it generically where feasible. Do not silently substitute another tile.

Report directly to the user after each stage:

- Implementation changes and their scope.
- Correctness checks and failures.
- Actual experiments completed.
- Performance tables and exact speedup denominators.
- NAND Sense attribution and any new read amplification.
- Limitations and unresolved modeling assumptions.

Do not invent results or leave placeholder performance claims.

**Final deliverable:** a reproducible, validated joint Compute-Tiling × Physical-Layout sweep, with a clearly separated and competitive FlashAccel baseline suitable for later comparison against Ours.

### Implementation and reproduction

The sweep driver generates independent BN=128/256 schedule files from the
canonical BN=64 schedules, with two A/B stages and a BF16 output scratch tile.
The original library and the FlashAccel FIFO policy are reused. Layout
selection comes from a per-tensor JSON sidecar, is encoded in runtime tensor
records, and is also passed to the prefetch planner so its spans address
physical pages. The C++ cache receives physical lines before coalescing.
Tile-major loads currently require the complete memory-tile height and
line-aligned columns. The simulator models the resulting TMA request stream
without charging separate address-translation or data-conversion instructions.

```sh
cmake --build build-backends --target gtsim_program
PYTHONPATH=frontend/src python3 frontend/tools/run_tile_lens_sweep.py \
  --out validation/results/tile-lens/sweep --jobs 4
PYTHONPATH=frontend/src python3 frontend/tools/run_tile_lens_sweep.py \
  --out validation/results/tile-lens/sweep --jobs 4 --focused
PYTHONPATH=frontend/src python3 frontend/tools/run_tile_lens_sweep.py \
  --out validation/results/tile-lens/sweep --summarize
```

The focused runs isolate the fastest prefetched block for each layout. They
start with cold GPU and NAND caches, retain the block's physical tensor base
addresses, and measure the three projections plus Gate/Up and Down for each
selected expert separately. Their prefetched runs enqueue the isolated
launch's requests at cycle zero, so their cycles should be read separately
from the overlapping full-block timeline.

`summary.csv` reports the complete sweep metrics. `kernels.csv` records
per-launch traffic, resources, SRAM gauges, and Megaplane busy time. A
repeated Sense means the same physical Megapage has been sensed earlier in
the run, including background prefetch. Per-launch counters attribute activity
to the interval in which it occurred. Prefetch completion timing is recorded
at launch-interval granularity. For a cold aligned Tile-major load, a
16×64 BF16 weight tile uses half of one 4 KiB page and senses one 128 KiB
Megapage (64×). A 16×128 tile uses one page and senses one Megapage (32×).
These ratios are distinct from observed NAND bytes divided by host-demand
bytes across a kernel or block.

### Validation

The standalone C++ address test checks an 8×8 bijection and CPU dot product,
BF16 4 KiB tile boundaries, expert isolation, and exact physical cache lines
for BN=64/128/256. The frontend tests compare physical prefetch page sets
against independent coordinate enumeration. The C++ runner validates each
CTA's SMEM and register requirements. CTest is skipped per the project
instruction.

### Measured results

All 72 hybrid cases, the five-stack FullHBM control, and 38 isolated GEMMs
completed. All full-block streams were ordered and drained. Every hybrid
block issued 113,779,200 HBF demand bytes and zero HBF writes.

Each cell reports cycles / NAND MiB / repeated Sense commands.

| BN A/B/C    |                  Row demand |               Row prefetch |                   Tile demand |              Tile prefetch |
| ----------- | --------------------------: | -------------------------: | ----------------------------: | -------------------------: |
| 64-64-64    |    4,291,764 / 134.75 / 206 |   1,604,395 / 135.62 / 213 |    2,790,962 / 390.62 / 2,253 | 1,481,738 / 256.00 / 1,176 |
| 64-64-128   |    4,299,208 / 134.75 / 206 |   1,582,688 / 135.62 / 213 |    2,841,809 / 395.12 / 2,289 | 1,513,261 / 257.12 / 1,185 |
| 64-128-64   |    6,105,974 / 183.50 / 596 |   3,095,498 / 159.88 / 407 |    4,929,123 / 911.75 / 6,422 | 2,040,847 / 399.88 / 2,327 |
| 64-128-128  |    6,113,107 / 185.50 / 612 |   3,109,386 / 159.75 / 406 |    4,938,441 / 913.62 / 6,437 | 2,041,308 / 400.00 / 2,328 |
| 64-256-64   |    8,147,642 / 208.62 / 797 |   4,648,469 / 185.38 / 611 |    5,187,060 / 967.50 / 6,868 | 2,237,583 / 401.88 / 2,343 |
| 64-256-128  |    8,155,409 / 214.38 / 843 |   4,669,338 / 185.38 / 611 |    5,176,378 / 966.88 / 6,863 | 2,255,433 / 398.38 / 2,315 |
| 128-64-64   |    6,199,931 / 165.00 / 448 |   3,509,608 / 162.25 / 426 |    5,870,291 / 884.12 / 6,201 | 3,340,405 / 647.50 / 4,308 |
| 128-64-128  |    6,206,967 / 165.00 / 448 |   3,517,373 / 162.25 / 426 |    5,882,274 / 884.38 / 6,203 | 3,349,167 / 646.88 / 4,303 |
| 128-128-64  |    8,012,442 / 215.75 / 854 |   5,014,997 / 186.50 / 620 | 7,874,338 / 1,395.12 / 10,289 | 3,904,090 / 790.50 / 5,452 |
| 128-128-128 |    8,020,667 / 213.75 / 838 |   5,023,233 / 186.50 / 620 | 7,884,163 / 1,396.88 / 10,303 | 3,910,014 / 790.25 / 5,450 |
| 128-256-64  | 10,055,841 / 238.88 / 1,039 |   6,571,266 / 212.12 / 825 | 8,223,815 / 1,462.50 / 10,828 | 4,107,903 / 791.88 / 5,463 |
| 128-256-128 | 10,062,508 / 238.88 / 1,039 |   6,578,441 / 212.12 / 825 | 8,235,435 / 1,464.25 / 10,842 | 4,113,756 / 791.88 / 5,463 |
| 256-64-64   |    9,758,285 / 221.00 / 896 |   6,550,766 / 208.00 / 792 |    6,415,915 / 982.00 / 6,984 | 3,522,851 / 657.62 / 4,389 |
| 256-64-128  |    9,766,052 / 221.00 / 896 |   6,555,045 / 208.00 / 792 |    6,427,845 / 984.38 / 7,003 | 3,527,773 / 657.75 / 4,390 |
| 256-128-64  | 11,571,594 / 269.75 / 1,286 |   8,052,368 / 232.25 / 986 | 8,427,799 / 1,504.00 / 11,160 | 4,088,165 / 800.75 / 5,534 |
| 256-128-128 | 11,579,296 / 271.75 / 1,302 |   8,060,133 / 232.25 / 986 | 8,435,505 / 1,504.12 / 11,161 | 4,094,703 / 800.75 / 5,534 |
| 256-256-64  | 13,612,968 / 294.88 / 1,487 | 9,609,041 / 257.88 / 1,191 | 8,761,112 / 1,565.75 / 11,654 | 4,289,748 / 802.38 / 5,547 |
| 256-256-128 | 13,620,832 / 294.88 / 1,487 | 9,616,183 / 257.88 / 1,191 | 8,794,740 / 1,565.50 / 11,652 | 4,295,413 / 802.50 / 5,548 |

| Condition     | Best A/B/C |    Cycles | Original BN64 / best |
| ------------- | ---------- | --------: | -------------------: |
| row-demand    | 64-64-64   | 4,291,764 |               1.000× |
| row-prefetch  | 64-64-128  | 1,582,688 |               1.014× |
| tile-demand   | 64-64-64   | 2,790,962 |               1.000× |
| tile-prefetch | 64-64-64   | 1,481,738 |               1.000× |

| Matched comparison                   | Numerator cycles | Denominator cycles |  Ratio |
| ------------------------------------ | ---------------: | -----------------: | -----: |
| Layout (64-64-128, demand)           |        4,299,208 |          2,841,809 | 1.513× |
| Layout (64-64-128, prefetch)         |        1,582,688 |          1,513,261 | 1.046× |
| Prefetch (64-64-128, row)            |        4,299,208 |          1,582,688 | 2.716× |
| Prefetch (64-64-128, tile)           |        2,841,809 |          1,513,261 | 1.878× |
| Layout (64-64-64, demand)            |        4,291,764 |          2,790,962 | 1.538× |
| Layout (64-64-64, prefetch)          |        1,604,395 |          1,481,738 | 1.083× |
| Prefetch (64-64-64, row)             |        4,291,764 |          1,604,395 | 2.675× |
| Prefetch (64-64-64, tile)            |        2,790,962 |          1,481,738 | 1.884× |
| Original FlashAccel / overall winner |        1,604,395 |          1,481,738 | 1.083× |
| Overall winner / five-stack FullHBM  |        1,481,738 |            430,112 | 3.445× |

The Row-major BN64 controls reproduced Step 8 exactly, with 4,291,764
cycles without prefetch and 1,604,395 with prefetch. Router BN128 was
the only winning tiling change. Its own launch took 6,658 more cycles,
but the later Gate/Up interval took 28,601 fewer cycles, with the same
total NAND bytes. These are stream timings with overlapping prefetch.

The best Tile-major block and original FlashAccel both sensed 872 unique
Megapages. Tile-major reached 1,481,738 cycles while increasing NAND
traffic from 135.62 to 256.00 MiB and repeated Senses from 213 to 1,176.
The maximum/mean Megaplane Sense busy time fell from 7.03 to 3.30 across
48 active units. SRAM-resident reads fell from 99.24 to 85.83 MiB, and
fallback reads rose from 9.27 to 22.68 MiB. Thus lower latency coincided
with more balanced activity and higher NAND and fallback traffic.

Wider A/B tiles reduced CTA parallelism and generally raised Sense traffic.
For example, Tile-major BN128 for all groups dropped QKV from 80 to 40 CTAs
and Gate/Up from 192 to 96, while block NAND traffic rose to 790.25 MiB.
The smaller cold-access amplification for a single tile did not predict
the block's repeated Senses.

#### Isolated GEMM cycles

Both configurations use the existing prefetch policy. Each expert range
below summarizes eight standalone runs, with all expert IDs and traffic
in the result CSV.

| Operation          | Row 64/64/128 | Tile 64/64/64 |
| ------------------ | ------------: | ------------: |
| QKV                |        80,678 |        94,684 |
| Attention output   |       113,013 |       119,082 |
| Router             |        59,961 |        52,531 |
| Gate/Up per expert | 49,578–49,642 |        55,270 |
| Down per expert    |        24,052 | 26,817–26,834 |

Artifacts are under `validation/results/tile-lens/sweep/`. The unaggregated
files include each program, simulator log, runner provenance, and timing
JSON. Timings are simulator results for this B=1 trace and the stated
TMA abstraction.
