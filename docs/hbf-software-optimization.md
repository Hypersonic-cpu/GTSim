# HBF software optimization experiments

This experimental `hbf-swopt` branch extends the existing hand-written GEMM
and FA3 DAG builders. Omitting kernel options keeps the original 128x256x64
GEMM and 128x128x128 FA3, both at depth 2. Existing reference cycles are
checked without changing the reference manifest or hardware timing YAMLs.

## Kernel options

Both executables accept `--stages 2|3|4`, `--split 1|2|4|8`, and
`--cta-order POLICY` after the existing workload arguments and memory config.
GEMM additionally accepts `--bm 64|128`, `--bn 128|256`, `--bk 32|64|128`,
and `--group-size N` (default 8). FA3 accepts `--q-tile 64|128` and
`--kv-tile 64|128|256`; head dimension remains 128. GEMM dimensions must divide
the selected tiles and partition lengths. FA3 split KV lengths align to 64
elements; a final 64-element KV tail is decomposed and transferred at its
actual extent (including F2 split 8 with 192-token partitions). Unsupported/infeasible cases exit
with code 2 and a footprint/capacity message before DAG construction.

GEMM original ordering traverses groups of up to eight N tiles, then M.
M-major varies M fastest; N-major varies N fastest. Grouped ordering traverses
groups of M tiles, with M fastest inside each group. These change the tile
sequence executed by persistent CTAs and hence the SMs accessing each tile.
FA3 original traverses batch, query head, Q block; query-major traverses
batch, Q block, query head; KV-head-grouped traverses batch, KV head, Q block,
query heads within that GQA group. Rank determines CTA ID and SM assignment;
the SM queue dispatches CTA IDs in order. Reordering therefore affects execution.

## Buffer and occupancy accounting

For GEMM, stage k waits for all consumers of buffer k modulo depth; next
loads retain nonblocking TMA issue edges. Each compute fragment waits for
its A/B inputs. Persistent CTAs also protect buffers across output tiles.
FA3 K buffers wait for their QK consumers, and V buffers independently wait
for their PV consumers. Variants retain score/online-softmax/PV ordering and
add explicit dependencies on preceding normalization/accumulator state.
WGMMA decomposition scales the counts of 16x64x16 warp fragments with tiles;
softmax work and output transfers scale with query/KV extents.

Shared bytes per GEMM CTA = depth * 2*BK*(BM+BN) + BM*BN*output_element_bytes.
The C staging area is reserved separately because persistent CTAs may overlap
next-tile inputs with C stores. FP32 partials use four output bytes.
Shared bytes per FA3 CTA = 2*QTile*128 + depth*4*KVTile*128. Output reuses input
storage after the final PV; scores/probabilities and accumulators use registers.
Estimated register bytes = FP32 accumulator/score extents + 32 registers per
thread for other state, with the original 12-warp CTA retained.

H100 capacity defaults are 228 KiB shared per SM, 227 KiB per CTA, 65536
32-bit registers per SM, 64 warps per SM, and 255 registers per thread.
See [NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/archive/12.2.1/hopper-tuning-guide/index.html).
These capacity checks are independent of memory/device timing. Register
counts are analytical estimates, not compiler measurements. SM admission
reserves/reclaims resources at CTA dispatch/retirement and respects the YAML
CTA cap (currently one for H100). Hardware timing and that cap are unchanged.
Default GEMM depth 3 fits (208 KiB); depth 4 does not (256 KiB).
Default FA3 depth 3 fits (224 KiB); depth 4 does not (288 KiB).
A fixed smaller-tile depth-2/3/4 control is included to measure depth 4 fairly.
KVTile 256 requires more shared memory than available with both K/V buffered
at depth 2 and is rejected under this retained layout.

## Split reductions

Split-K partitions K across independent CTAs, writes FP32 partial C into HBM,
and dispatches reduction CTAs after all partial stores. Reductions read all
partials, charge split-1 FP32 additions per element, cast, and store FP16 C.
Split-KV partitions each GQA head's KV sequence. Each partition stores its
locally normalized FP32 output and FP32 log-sum-exp (LSE) per query row.
The merge uses m=max(LSE_s), w_s=exp(LSE_s-m), and
O=sum(w_s O_s)/sum(w_s). Reads, max, exp, weight sums, scaling, additions,
division, casts, and output stores are DAG operations on existing pipelines.
Scratch placement follows existing single-tier overrides in FullHBM/FullHBF.
No split merge is zero-cost. Reduction CTAs reserve an estimated 72 KiB of registers including fragments,
normalization state, and loop overhead. Direct register fragment loads bound merge
storage; CTA resources are accounted for separately. Output store issue edges
release register fragments once the payload has issued. All store responses
still retire the CTA; register reuse does not serialize on HBF program acknowledgments.

GTSim models timing rather than tensor values. `kernel_swopt_semantics`
checks arithmetic work counts, global bounds/dtypes, acyclicity, issue versus
completion edges, independent K/V lifetimes, actual CTA/SM mapping, occupancy,
and numerical agreement of the stable merge formula with unsplit softmax.
This is not a CUDA kernel numerical validation.

## Reproduce and represent results

```bash
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
python3 scripts/run_swopt.py --dry-run
python3 scripts/run_swopt.py
python3 scripts/run_swopt.py --combinations
```

The default four simulations run concurrently. Existing result folders are
reused; `--overwrite` reruns selected experiments. `--phase`, `--workload`
(G1/G2/G3/F1/F2/F3), and `--topology` restrict incremental experiments.
The baseline sweep launcher also defaults to `--parallelism 4` and accepts
`--kernel-args='--stages 3' --variant stages3` for custom cases.

Results live under `validation/results/hbf-software-p2-p3/`:

- `original/baseline/WORKLOAD-TOPOLOGY`: the original configuration.
- `p2-depth/stagesN/...`: depth changes at original tiles.
- `p2-controlled-depth/small-tile-stagesN/...`: GEMM 64x128x64 or FA3 Q128/KV64,
  held fixed across depths. Compare against `small-tile-stages2` to isolate depth.
- `p3-tiles/BM*-BN*-BK*` or `Q*-KV*-D128`: tile changes alone at depth 2.
- `p3-cta-order/POLICY/...`: ordering alone.
- `p3-split/splitN/...`: partition factor alone, with charged merges.
- `p2-p3-combined/depth3-.../...`: measured combinations of depth, ordering,
  and (where feasible) tile selection. These are separate from independent effects.

Each leaf contains `run.log` and `result.json`; root CSV/JSON contain all runs.
`configurations.json` gives exact flags, paths, dimensions, input hashes, and
Git revision. `source_snapshot/` and versioned `source_snapshots/` capture uncommitted
implementation files. Each result records its snapshot and binary hash.
Superseded diagnostic timings are excluded from the final CSV.
`summary.md` reports per-topology speedups and independent contributions.
`--combinations` runs depth 3 with the best independently measured CoLocated
CTA ordering; it additionally tests the best tile at depth 3 when that footprint
fits. This phase is chosen only after the independent sweep completes.
Primary comparisons use FullHBM and CoLocated for all six workloads;
G3/F2 representative variants additionally run FullHBF and Cascaded.

Present speedup (original cycles / variant cycles) versus depth, tile shape,
policy, or split factor, with separate panels for topology and workload.
`--variant` selects exact variants for incremental reruns.
Include infeasible points as rejected, never as zero cycles. For controlled
P2, show both speedup against the original and against depth 2 at the same tile.
Use returned HBM/HBF read bytes/request counts as traffic/reuse evidence;
cycle improvements alone do not establish reduced L2 misses. Cross-topology
comparisons include the existing bandwidth/stack differences as well as latency. Report register
estimates, the one-CTA cap, and analytical timing assumptions with the plots.

## Workload helpers

Shared kernel option parsing, feasibility checks, and the numerical attention merge oracle live in `workloads/common/workload_options.h`. The split-reduction DAG builder lives in `workloads/common/split_reduce.h`. These are workload helpers; the simulator core retains only general CTA resource admission. Both workload builders include the helpers directly, so they are not part of the public simulator headers.

## CTA residency and waiting

Like `dev`, each SM admits CTAs up to its configured resident-CTA limit, queues the rest in FIFO order, and dispatches them after resident CTAs retire and the dispatch delay expires. Additional resource checks enforce aggregate shared-memory, register, and warp capacity. A CTA temporarily blocked by resources stays queued; retirement releases its reservation before redispatch. Workload feasibility checks reject individually oversized kernels before simulation.
