# TVM GEMM HBM/HBF placement evaluation

The `tvm-with-hbf` frontend now preserves tensor device identity through native
scheduled TVM TIR, `.gtsir`, device address allocation, DAG materialization, L2
requests and the existing heterogeneous memory system. The schedule is unchanged.

## Changes and design

| Files | Change |
|---|---|
| `scripts/tvm/h100_schedule.py`, `lower_gemm.py` | Declare and validate matching `global.hbm` / `global.hbf` buffer scopes and `gtsim.tensor_tiers`; preserve the complete native instruction/barrier pattern. |
| `scripts/tvm/tile_ir.py`, `src/tile_ir_parser.cpp` | Carry explicit HBM/HBF device identity in each version-1 tensor record. Existing HBM wire output remains byte-identical. |
| `include/dag_node.h`, `src/tile_dag_builder.cpp` | Add device allocations in disjoint HBM/HBF address apertures; reject devices absent from the selected topology. Keep the reference builders' legacy allocator and routing unchanged. |
| `scripts/tvm/gemm.py`, `workloads/gemm/gemm_ir.cpp` | Expose `--placement baseline/hybrid`, topology-aware preflight and `--inspect` for actual allocated tensor addresses, CTA resources and DAG operations/dependencies. |
| `configs/memconf/colocated-2F3M.yaml` | Derive the requested second topology from the existing 3F2M profile. |
| `tests/tile_ir_tests.cpp`, `tests/test_tvm_placement.py`, `tests/test_tvm_gemm.py`, `CMakeLists.txt` | Test scope/metadata agreement, serialization, address isolation, real controller/link completion, missing-device rejection, graph equivalence and multi-wave synchronization. |
| `scripts/experiments/run_tvm_placement.py` | Reproduce the eight matched runs, twice each, through native TVM generation; retain inputs, DAG inspections, commands, hashes and raw logs. |
| `docs/tile-ir.md`, `docs/tvm-gemm-status.md`, this report | Update the frontend contract and distinguish the earlier FullHBM execution record. |

DAG nodes retain tensor identity through `matrix_id`; the immutable registry
supplies addresses and device identity. L2 resolves the device from each actual
global address, including writebacks. CoLocated uses separate `gpu_to_hbm` and
`gpu_to_hbf` links, controller queues and responses. HBF requests reach
`HBFModel::try_submit` and its independently bandwidth-limited read/write queues.
No GEMM-specific routing or new timing policy was introduced in `MemorySystem`.
FullHBM's legacy automatic override remains available to old reference builders;
the Tile IR path rejects an HBF allocation on FullHBM before simulation.

## Two native TVM representations

These are declaration excerpts from the generated `scheduled.py` files; each has
the same complete scheduled body. The serialized native module is reloaded before
lowering, and passes SSA/well-formedness checks plus real simplification,
flattening and intrinsic-lowering passes.

```python
# Baseline
@T.prim_func
def gemm(A: T.Buffer((512, 512), "float16", scope="global.hbm"),
         B: T.Buffer((512, 512), "float16", scope="global.hbm"),
         C: T.Buffer((512, 512), "float16", scope="global.hbm")):
    T.func_attr({"gtsim.tensor_tiers": ["HBM", "HBM", "HBM"]})
    # Complete common scheduled body in the generated artifact.

# Hybrid
@T.prim_func
def gemm(A: T.Buffer((512, 512), "float16", scope="global.hbm"),
         B: T.Buffer((512, 512), "float16", scope="global.hbf"),
         C: T.Buffer((512, 512), "float16", scope="global.hbm")):
    T.func_attr({"gtsim.tensor_tiers": ["HBM", "HBF", "HBM"]})
    # Complete common scheduled body in the generated artifact.
```

Both allocate `stage_a[2,128,64]`, `stage_b[2,4,64,64]` and
`stage_c[4,128,64]` in shared memory. Both use TMA tensor-map loads/stores,
`wgmma.m64n64k16`, full/empty mbarriers, WGMMA fence/commit/wait, CTA joins and
the existing cross-wave bulk-store completion wait. BM/BN/BK are 128/256/64,
with two pipeline stages, 12 warps, split=1, the original CTA mapping, 132 SMs,
and the unchanged H100 profile at 1980 MHz. No cost evaluator or schedule search
is involved.

## Hardware assumptions

All bandwidths below are decimal GB/s. Each baseline/hybrid pair uses exactly
the same architecture and memory configuration file.

| Parameter | Existing CoLocated 3F2M | Derived CoLocated 2F3M |
|---|---:|---:|
| HBM read / write ceiling | 1240 / 620 | 1860 / 930 |
| HBF read / write | 1572.864 / 83.88 | 1048.576 / 55.92 |
| GPU–HBM link | 3072 | 4608 |
| GPU–HBF link | 4608 | 3072 |
| HBM read / write latency, cycles | 270 / 270 | 270 / 270 |
| HBF read / write latency, cycles | 7920 / 148500 | 7920 / 148500 |
| Link latency, cycles | 10 | 10 |

Source: `configs/memconf/colocated-3F2M.yaml`. HBM bandwidth scales by 3/2;
HBF bandwidth scales by 2/3. Links retain the documented 1536 GB/s per stack.
Queue capacities, latency, cache, TMA and compute parameters are unchanged.
The inherited HBF latency assumptions are 4 µs reads and 75 µs writes at
1980 MHz; this evaluation adds no calibration or physical HBF measurements.
The existing TVM compiler was copied from the sibling checkout's `build-tvm`
as requested; compiler/runtime hashes and the pinned commit were verified.

## Verification and results

All eight runs completed twice with identical cycles and per-device traffic. Speedup is baseline cycles / row cycles; values below 1 mean slowdown. Traffic is controller payload in MiB (2^20 bytes), excluding link control packets.

| Topology | M×N×K | Placement | Cycles | Speedup | HBM read/write MiB | HBF read/write MiB |
|---|---|---|---:|---:|---:|---:|
| 3F2M | 512×512×512 | baseline | 13,254 | 1.0000× | 1/0.5 | 0/0 |
| 3F2M | 512×512×512 | hybrid | 43,854 | 0.3022× | 0.5/0.5 | 0.5/0 |
| 3F2M | 2048×4096×512 | baseline | 82,801 | 1.0000× | 6/16 | 0/0 |
| 3F2M | 2048×4096×512 | hybrid | 128,562 | 0.6441× | 2/16 | 4/0 |
| 2F3M | 512×512×512 | baseline | 12,696 | 1.0000× | 1/0.5 | 0/0 |
| 2F3M | 512×512×512 | hybrid | 43,296 | 0.2932× | 0.5/0.5 | 0.5/0 |
| 2F3M | 2048×4096×512 | baseline | 65,305 | 1.0000× | 6/16 | 0/0 |
| 2F3M | 2048×4096×512 | hybrid | 116,964 | 0.5583× | 2/16 | 4/0 |


The 512³ case has 8 CTAs and one output wave. The 2048×4096×512 case has
256 output tiles on 132 persistent CTAs: two waves, with 124 active CTAs in
the final wave. Both execute eight K iterations through the same two shared
stages. Hybrid is slower by 3.3087× / 1.5527× on 3F2M and 3.4102× / 1.7910×
on 2F3M (small / multi-wave). These are simulated timings.

Evidence that B reaches HBF:

- Small hybrid B is allocated at `0x1000000000000000`; A and C remain in the HBM aperture starting at `0x100000000`. Baseline B is `0x100080000`. Allocation is compact within each device, so C’s base also changes with B placement; shapes, strides and access regions remain identical.
- Small hybrid runs complete 4,096 HBF reads / 524,288 bytes, zero HBF writes, and transfer 524,288 return-payload bytes through `gpu_to_hbf`. Both baselines have zero HBF requests and zero HBF link traffic.
- Multi-wave hybrid runs complete 32,768 HBF reads / 4,194,304 bytes, zero HBF writes; the HBF link returns exactly 4,194,304 payload bytes. HBM reads fall from 6 MiB to 2 MiB, with all 16 MiB of C writes still routed to HBM.
- HBF average end-to-end request latency is 7,942 cycles for the small runs, and 7,942.60 / 7,943.21 cycles for the multi-wave 3F2M / 2F3M runs. This includes the inherited 7,920-cycle HBF read latency plus transfer/link/queue effects.
- The C++ boundary test sends a materialized B TMA cache-line address through L2 and requires HBF completion, HBF-link return payload and the HBF latency floor, with zero HBM requests. FullHBM placement rejection prevents the legacy remapping from masking routing errors.

For each matched pair, all materialized operations, regions, CTA/SM/warp mappings,
resource footprints and ordered issue/completion dependencies compare exactly
(same placement-independent SHA-256). The entire `.gtsir` is also byte-identical
after normalizing only `tier=HBF` to `tier=HBM`. The small graph has 9,568 nodes, 216,000
completion edges and 1,424 issue edges; the multi-wave graph has 306,176 nodes,
6,914,480 completion edges and 45,568 issue edges. TMA loads / WGMMA fragments /
shared stores / TMA stores are 320 / 8,192 / 1,024 / 32 for small and
10,240 / 262,144 / 32,768 / 1,024 for multi-wave.

Logical TMA payload is A/B/C = 1/2/0.5 MiB for small and 32/64/16 MiB for
multi-wave. Controller reads are lower because existing L2 hits and request
merging reuse A/B data. Moving B changes its device without changing logical
traffic. Raw per-device latency and link utilization are retained in the logs.

There are no stall-reason counters in this runner. Its existing `CpAvg` statistic
measures each K-round’s span from first TMA-load start to last load completion;
the following averages across output tiles are transfer/wait proxies, not
additive stall-cycle counts:

| Topology | Case | Baseline mean CpAvg | Hybrid mean CpAvg |
|---|---|---:|---:|
| 3F2M | small | 1324.88 | 8974.88 |
| 3F2M | multi-wave | 2232.36 | 7857.73 |
| 2F3M | small | 1324.88 | 8974.88 |
| 2F3M | multi-wave | 2246.71 | 7885.73 |


Checks passed: Release build; **14/14 CTests**, including the new placement
integration test; workload-profile check; unchanged deterministic references
**GEMM 12,249 / FA3 7,212 cycles**. Public validation remains GEMM MAPE
**5.70198%** (8 cases) and FA3 MAPE **4.95376%** (6 cases). The earlier GEMM
measurement outlier remains outside this placement work; no references or
calibration parameters were updated.

The small-case increase in `CpAvg` is exactly 7,650 cycles, matching the
configured HBF minus HBM read-latency difference (7,920 − 270). This supports
latency as the main cause of the small-case slowdown.

## Reproduction and artifacts

After the normal pinned-TVM Python setup, reuse the existing compatible TVM
build or follow the setup in `tile-ir.md`. This evaluation reused the sibling
`build-tvm`; it did not require a new compiler build.

```sh
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
python scripts/experiments/run_tvm_placement.py --repeats 2
```

Use the Python interpreter configured with the pinned TVM dependencies.
Artifacts are under `validation/results/tvm/placement/`: `results.json`,
`comparison.md`, `verification.json`, plus each run’s `scheduled.py`, native
`scheduled.json`, `lowered-tvm.py`, `kernel.gtsir`, `lowering.json`,
`materialized-dag.json` and two simulation logs. These generated artifacts are
ignored by Git; this report and the reproducer are tracked changes.

## Limitations and next steps

- This remains the bounded persistent Hopper GEMM frontend, with FP16 A/B/C and FP32 accumulation. It does not lower arbitrary TVM schedules or provide a physical HBF runtime allocator. Native TIR imports must declare matching device scopes; regenerate older global-only TIR artifacts.
- CUDA numerical execution and physical HBF timing validation remain unperformed. HBF results inherit the repository’s controller abstraction and the stated latency/bandwidth assumptions; the 2F3M derivation assumes linear stack bandwidth.
- This experiment preserves the shared 50 MiB L2 and existing TMA behavior. It does not test capacity pressure, physical HBF bank/channel layout, HBF writes or alternate placements.
- Next useful checks are an above-L2-capacity matched workload and independent HBF/link microbenchmarks. Neither requires changing the GEMM schedule.
