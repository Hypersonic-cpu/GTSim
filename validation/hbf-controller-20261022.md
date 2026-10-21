# HBFine integration and evaluation — 2026-10-22

Implemented on `dev-ramulator2`, starting from `e8950e5`. Work began October 21
and completed October 22. The detailed device is a standalone library in the
registered `3rdparty/hbfine` submodule (`git@github.com:Hypersonic-cpu/HBFine.git`),
namespace `hbfine`, target `HBFine::hbfine`. `HBFineBackend` is the GTSim adapter.
The parent gitlink pins the independently published HBFine implementation at
`4a1a592`. No TVM build, schedule tuning, Ramulator vendor edits, or calibration
changes were performed.

## Design and changed files

| Area | Files and purpose |
|---|---|
| Standalone device | `3rdparty/hbfine/{CMakeLists.txt,cmake/,include/hbfine/,src/,tests/,README.md,LICENSE}`: dependency-free C++20 physical-time API, geometry validation, reversible mapping, sparse events, channel transfers, Bank caches, NAND legality, statistics, tests, install/export package |
| Adapter and routing | `src/hbfine_backend.cpp`, `include/memory_backend.h`, `src/memory_backend.cpp`, `include/memory.h`: clock conversion, original response metadata, one shared L2 path, admission/drain/errors, HBF preload, channelized naive HBM |
| Physical allocations/config | `include/{memory_layout.h,dag_node.h,mem_conf.h,arch_conf.h,hbf_address.h}`: explicit device segments, independent allocations, read-only/preloaded inputs, validated typed YAML profiles |
| Workloads | `workloads/common/placement.h`, `workloads/{gemm/gemm_h100.cpp,fa3/fa3_h100.cpp}`: explicit placement; FA3 batched/GQA ID fix; compute schedule unchanged |
| HBM stack scaling | `include/hbm_address.h`, `src/ramulator_backend.cpp`, `scripts/ramulator/export_h100.py`, `configs/ramulator/h100_hbm3_{2,4}stack.yaml`: retain per-channel timing with correct reduced-stack counts |
| Integration/evaluation | `CMakeLists.txt`, `tests/{hbf_backend_tests.cpp,workload_placement_tests.cpp,hbm_address_tests.cpp}`, `configs/memconf/`, `scripts/hbf/`, `README.md`, `docs/{hbf-controller.md,ramulator-hbm3.md}` |

HBFine does not include GTSim headers or own tensors/GPU clocks. It accepts
64B-multiple physical requests, backpressures bounded queues, orders same AXI-ID
responses, and returns exactly one completion per accepted parent. Whole-page
sensing, pinned two-buffer Bank caches, sequential programming, page-0 erase and
partial-page timeout are separate modeled behavior. Reads start cold from valid
preloaded input pages; initialization has no simulated write cost. Programming
requires explicit erase/timeout assumptions and is disabled in benchmark profiles.
The host must place updates in fresh blocks; there is no implicit FTL or GC.

`hbflex_research` assumes 512 Banks/stack with one Plane/Bank. `ocp_reference`
assumes 256 Banks/stack with two Planes/Bank and one serialized sense engine/Bank.
Both contain 512 Planes. Consequently their ideal cold-array read ceilings are
524.288 and 262.144GB/s/stack, rather than obtaining extra parallelism from page
buffers. This mapping is an assumption, not a source-established equivalence.
Sources and full model boundaries are in the [library README](../3rdparty/hbfine/README.md)
and [integration guide](../docs/hbf-controller.md).

## Placement and matched comparison

| Workload | FullHBM | Mixed configurations |
|---|---|---|
| GEMM | A/B/C in HBM | A/C in HBM; B preloaded read-only in HBF |
| FA3 | Q/K/V/O in HBM | K/V/O in HBM; Q preloaded read-only in HBF |

FA3 Q is a **synthetic per-invocation input**, not a model weight or persistent
query. All cases retain the reference two-stage schedules, tiles, CTA mapping,
1980MHz core and compute parameters. Graph comparisons cover operations, both
dependency types, tiles/strides, warp/CTA/SM assignment and multi-wave GEMM.
Mixed comparisons use separate physical HBM and HBF segments, not tensor labels.

The five-stack FullHBM reference here is `0F5M_channels.yaml`: distinct 80 HBM
channels, 3100/1550GB/s aggregate, 270-cycle HBM latency. 3F2M has 32 HBM/48 HBF
channels; 1F4M has 64 HBM/16 HBF channels. HBM remains 620GB/s read per stack;
HBF host targets are 488/27.2GB/s per stack with 4us sense/75us program. Links
use 1536GB/s per stack and the existing 10-cycle link latency. This matched
FullHBM baseline has a 7680GB/s link; the historical default retains its original
3100GB/s link and cycle references. No read-side SRAM hiding, uncalibrated ECC or
die bus limit is enabled. Optional Ramulator runs use its existing 383ps half-CK
and are separate from the primary naive-HBM comparisons.

## Verification

- Standalone HBFine C++20 configure/build/CTest passed; installation and an
  external `find_package(HBFine CONFIG REQUIRED)` consumer passed without GTSim.
- GTSim CTest: **12/12 passed** without Ramulator; **15/15 passed** with Ramulator.
  Workload-profile boundary and deterministic cycle references passed unchanged.
- All **8 measured GEMM + 6 measured FA3** reference cases retained exactly the
  pre-change executable's simulated cycles. This checks regression preservation,
  not measured accuracy of hypothetical mixed-memory systems.
- Focused tests cover mapping/striping, Bank/Plane conflicts, pinned buffers,
  cache hits/evictions, same-ID ordering, clock conversion, rejection/retry,
  conservation, illegal/partial writes, late-piece timeout, failed writebacks,
  capacity/segment mismatches, and batched/GQA placement/strides.
- Four mixed tiny smokes drained successfully: GEMM 128×256×64 took 12009/11966
  cycles on 3F2M/1F4M; FA3 B1 H1 S128 took 14774 on both. Each generated 32768B
  of actual HBF reads, 256 accepted/completed parents and 8 NAND senses, with
  zero writes/programs/erases/failures. These counters come from HBFine itself.
- Optional two/four-stack Ramulator plus HBFine GEMM/FA3 smokes passed with
  correct 32/64 physical HBM channel declarations and parent/burst conservation.
  Default five-stack Ramulator GEMM 512³ remained **12821 cycles**.
- All **35 comparative runs** completed and drained. Detailed runs conserved
  accepted/completed parents and all per-stack/channel/Bank/Plane read counters;
  every mixed run had HBF reads, and none had timed HBF writes or failures.

## Predicted cycles

Research and OCP columns select HBFine; naive columns use the existing aggregate
HBF service law with matched placement, target bandwidth and latency. FullHBM
has no HBF accesses. GEMM 2048×4096×512 uses 256 tiles and two persistent waves.

| Case | FullHBM | 3F2M naive | 3F2M research | 3F2M OCP | 1F4M naive | 1F4M research | 1F4M OCP |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemm_512x512x512 | 12,777 | 44,350 | 44,344 | 44,344 | 43,519 | 43,512 | 43,512 |
| gemm_1024x1024x1024 | 27,321 | 89,717 | 89,560 | 89,560 | 87,749 | 87,763 | 87,763 |
| gemm_2048x4096x512 | 53,557 | 140,797 | 115,861 | 115,861 | 278,455 | 198,386 | 243,440 |
| fa3_1x8x8x1536x1536 | 42,299 | 57,648 | 54,170 | 54,170 | 61,172 | 56,939 | 64,267 |
| fa3_1x8x8x512x512 | 17,012 | 26,780 | 25,884 | 25,884 | 27,973 | 26,839 | 26,839 |

| Case | 3F2M research / FullHBM | 1F4M research / FullHBM | 1F4M OCP / FullHBM |
|---|---:|---:|---:|
| gemm_512x512x512 | 3.471× | 3.405× | 3.405× |
| gemm_1024x1024x1024 | 3.278× | 3.212× | 3.212× |
| gemm_2048x4096x512 | 2.163× | 3.704× | 4.545× |
| fa3_1x8x8x1536x1536 | 1.281× | 1.346× | 1.519× |
| fa3_1x8x8x512x512 | 1.522× | 1.578× | 1.578× |

## Detailed-device traffic and host cost

Every row below has zero HBF write bytes, programs, erases, failed requests and
channel admission rejections. HBF read GB/s uses total drained simulation time.
Wall time/RSS include graph construction, simulation, output and final drain;
wall ratios are single samples under at most two concurrent simulators.

| Case | Profile | HBF read MiB | HBF GB/s | Cache hit beats | Evictions | Wall s | Peak RSS MiB | Wall / matched naive |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gemm_512x512x512 | 3F2M_research | 0.50 | 23.41 | 0 | 0 | 0.686 | 34.1 | 0.81× |
| gemm_512x512x512 | 3F2M_OCP | 0.50 | 23.41 | 0 | 0 | 0.696 | 34.2 | 0.82× |
| gemm_512x512x512 | 1F4M_research | 0.50 | 23.86 | 0 | 0 | 0.690 | 34.5 | 1.03× |
| gemm_512x512x512 | 1F4M_OCP | 0.50 | 23.86 | 0 | 0 | 0.709 | 34.1 | 1.06× |
| gemm_1024x1024x1024 | 3F2M_research | 2.00 | 46.36 | 0 | 0 | 1.931 | 149.2 | 1.04× |
| gemm_1024x1024x1024 | 3F2M_OCP | 2.00 | 46.36 | 0 | 0 | 1.863 | 148.4 | 1.01× |
| gemm_1024x1024x1024 | 1F4M_research | 2.00 | 47.31 | 0 | 0 | 1.871 | 150.7 | 0.95× |
| gemm_1024x1024x1024 | 1F4M_OCP | 2.00 | 47.31 | 0 | 0 | 1.848 | 147.5 | 0.94× |
| gemm_2048x4096x512 | 3F2M_research | 4.00 | 71.68 | 16384 | 0 | 4.430 | 690.8 | 0.88× |
| gemm_2048x4096x512 | 3F2M_OCP | 4.00 | 71.68 | 16384 | 0 | 4.133 | 689.4 | 0.82× |
| gemm_2048x4096x512 | 1F4M_research | 4.00 | 41.86 | 16384 | 0 | 5.919 | 684.5 | 0.79× |
| gemm_2048x4096x512 | 1F4M_OCP | 4.00 | 34.11 | 0 | 1024 | 6.915 | 685.2 | 0.92× |
| fa3_1x8x8x1536x1536 | 3F2M_research | 3.00 | 114.98 | 0 | 0 | 3.156 | 752.9 | 1.02× |
| fa3_1x8x8x1536x1536 | 3F2M_OCP | 3.00 | 114.98 | 0 | 0 | 3.303 | 746.9 | 1.07× |
| fa3_1x8x8x1536x1536 | 1F4M_research | 3.00 | 109.39 | 0 | 0 | 3.323 | 744.6 | 1.04× |
| fa3_1x8x8x1536x1536 | 1F4M_OCP | 3.00 | 96.92 | 0 | 256 | 3.368 | 743.3 | 1.06× |
| fa3_1x8x8x512x512 | 3F2M_research | 1.00 | 80.21 | 0 | 0 | 0.673 | 142.7 | 0.99× |
| fa3_1x8x8x512x512 | 3F2M_OCP | 1.00 | 80.21 | 0 | 0 | 0.673 | 142.7 | 0.99× |
| fa3_1x8x8x512x512 | 1F4M_research | 1.00 | 77.36 | 0 | 0 | 0.704 | 141.8 | 1.01× |
| fa3_1x8x8x512x512 | 1F4M_OCP | 1.00 | 77.36 | 0 | 0 | 0.722 | 141.6 | 1.04× |

OCP and research results coincide when page working sets avoid same-Bank
conflicts. On 1F4M multi-wave GEMM, OCP incurs more cache evictions and serialized
array service; its runtime is higher. The same mechanism affects larger FA3.
Cache hits can make detailed service faster than a naive model that charges
sense latency on every access. There is no schedule optimization behind these
changes. Small differences without cache hits also reflect per-channel 64B
transfer rounding and miss coalescing versus aggregate naive service. Low achieved throughput relative to target is expected for a kernel
with compute time, latency, and limited outstanding requests.

Raw evidence under `validation/results/hbf-controller-20261021/` includes
`accuracy-preservation.json`, both CTest logs, optional Ramulator smoke logs,
and `final-sweep-verified/{comparison.csv,cycles.csv,resources.csv,provenance.json}` plus
one log/JSON per simulation. `comparison.csv` retains every case × topology ×
backend, both-device read/write traffic, host cost, queue stalls, aggregated
Bank/transfer wait ps, request failures and ratios. `resources.csv` retains each
Stack/Channel/Bank/Plane utilization and optional die/ECC counters. Stack
utilization normalizes summed Bank busy time by the number of Banks; channel
read/write occupancies are separate. Queue stalls are admission failures;
Bank/transfer wait totals are resource waiting, not direct GPU stall cycles.

Reproduce with `python3 scripts/hbf/compare_backends.py --build-dir build`.
Results are retained in an ignored directory; profiles and runner reproduce them.
The cached local GTSim build uses the pre-existing sibling Glaze include path
because this checkout lacks that registered header dependency. HBFine standalone
and installed-consumer builds require neither Glaze nor the sibling checkout.

## Limits and next steps

These are timing predictions without numerical GEMM/attention payload checks or
mixed-memory hardware ground truth. Bank/Plane mapping and host striping remain
explicit assumptions. OCP's strict reference geometry cannot sustain the 488GB/s
cold-array target without independently justified additional parallelism.
Programming is tested synthetically; no production erase latency, host timeout,
GC, wear model, FTL or read-side SRAM latency hiding has been fabricated. Optional
die/ECC resources remain disabled until defensible rates are available. Public
segmented workloads reject incompatible single-device placement rather than
silently overriding address ownership; legacy lower-level routing tests remain.

Further validation should use real device traces or microbenchmarks for Bank
mapping, internal transfers and erase behavior,
and repeated isolated host timing runs for a precise simulator-overhead estimate.
