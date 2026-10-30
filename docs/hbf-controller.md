# HBFine integration and mixed memory placement

GTSim uses the standalone [HBFine library](../3rdparty/hbfine/README.md) through
`HBFineBackend`. The submodule is `3rdparty/hbfine`, upstream
`git@github.com:Hypersonic-cpu/HBFine.git`, namespace `hbfine`, CMake target
`HBFine::hbfine`. GTSim owns tensors, address segments, GPU clocks and D2D links;
HBFine owns NAND mapping, channel transfers, Bank queues/caches, physical-time
events and completion/error accounting. Neither library includes the other's
implementation headers. GTSim links the public HBFine target for configuration
and statistics types; no simulator-specific routing algorithm lives in HBFine.

The existing L2 → `MemoryBackend` path handles both devices. Rejected admissions
remain upstream, accepted IDs complete once, and failed read/writeback responses
raise errors. No L2 slice model or second memory-request path was introduced.
The adapter converts integer GPU cycles to picoseconds with a kHz clock ratio;
it retains request metadata until completion. HBF physical traffic cannot fall
back to HBM. Ramulator HBM remains optional and uses the same backend contract.

## Placement and address ownership

Default public workloads explicitly use `all_hbm` and preserve historical
addresses (4GiB allocation offset) and cycle references. Mixed configs use
separate HBM/HBF bump cursors, zero offset, 4KiB-aligned reservations and contiguous
physical device segments:

| Profile | HBM segment | HBF segment | HBM channels | HBF Host Channels |
|---|---|---|---:|---:|
| 0F5M | [0, 80GiB) | empty | 80 | 0 |
| 3F2M | [0, 32GiB) | [32GiB, 1568GiB) | 32 | 48 |
| 1F4M | [0, 64GiB) | [64GiB, 576GiB) | 64 | 16 |

Tensor extents, alignment padding, request sizes, resolved address tier and
metadata are cross-checked. Holes, overlapping reservations, capacity overflow,
cross-device requests and writes to read-only tensors are rejected. Cache-line
transactions may include a tensor's final reserved padding, but must start in
its actual byte extent.

| Placement | HBM tensors | HBF tensors |
|---|---|---|
| `all_hbm` | GEMM A/B/C; FA3 Q/K/V/O | none |
| `gemm_b_hbf` | A/C | B, preloaded read-only |
| `fa3_q_hbf` | K/V/O | Q, synthetic preloaded read-only per-invocation input |

FA3 has no weight tensor. Q changes between invocations; this experiment makes
no claim about persistent Q or free repeated initialization across decode steps.
GQA K/V IDs are unique across batches and shared only by the appropriate query
head group. TMA shapes/strides, two-stage buffering, compute nodes, CTA placement,
and dependencies are identical across placement profiles. The regression test
compares both graph dependency types and tiles, including persistent multi-wave
GEMM. This branch uses the reference builders; this change adds no TVM frontend
or schedule search.

## Configurations and assumptions

All primary comparisons use the existing 1980MHz H100 compute profile. Naive HBM
uses the existing 620GB/s read and 310GB/s write per stack, 270-cycle device
latency, split across distinct 16-channel-per-stack service instances. Aggregate
legacy naive profiles remain unchanged. New matched 0F5M uses those same channel
instances and 3100/1550GB/s aggregate, so it is an explicitly defined comparison
baseline rather than a change to the original calibration.

Mixed research and OCP profiles share HBFlex's 488/27.2GB/s per-stack host target
and 4/75us sense/program parameters. They differ only in the declared Bank/Plane
mapping; see the library README for spec-derived versus assumed behavior. The
OCP geometry does not guarantee its host target for cold misses: its serialized
array ceiling is 262.144GB/s/stack. Cold caches, no timed initialization writes,
no SRAM read-side latency hiding, and disabled uncalibrated die/ECC limits apply
to all detailed benchmarks. Production profiles disable HBF programming; explicit
synthetic tests exercise program, erase and write timeout behavior separately.

The co-located external links use OCP Grade-2 ceilings (1536GB/s per stack,
10-cycle existing link latency). 0F5M uses 7680GB/s GPU-to-HBM, 3F2M uses
3072/4608GB/s HBM/HBF and 1F4M uses 6144/1536GB/s. These are link assumptions;
NAND target and array limits remain independent. They do not replace the legacy
3100GB/s GPU-to-HBM default or import H200 HBM3e timings.

Optional `*_ramulator_patch.yaml` files select two/four HBM-stack Ramulator
profiles at the existing 383ps half-CK timing. Their pin ceilings are about
668.407GB/s/stack, distinct from the naive calibrated 620GB/s; they are not used
in the primary matched-naive HBF sweep. Default five-stack Ramulator behavior
and the original H100 comparisons are retained.

## Reproduction

```sh
git submodule update --init 3rdparty/hbfine
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/gtsim_gemm configs/v0.1.0/h100-sxm.yaml \
  --memory-config configs/memconf/3F2M_bank_plane.yaml \
  --memory-config configs/memconf/placement_gemm_b_hbf.yaml 512 512 512
./build/gtsim_fa3 configs/v0.1.0/h100-sxm.yaml \
  --memory-config configs/memconf/1F4M_bank_plane_ocp.yaml \
  --memory-config configs/memconf/placement_fa3_q_hbf.yaml 1 8 8 1536 1536
ctest --test-dir build --output-on-failure
python3 scripts/hbf/compare_backends.py --build-dir build
```

The runner limits children to two, with GEMM ascending and FA3 descending; it
waits for child completion without status polling. Logs and JSON retain commands,
source/binary hashes, wall/CPU time, peak RSS, address declarations, counters and
per-resource arrays. `comparison.csv` includes achieved HBF host bandwidth,
traffic, stalls, failures, and ratios; `cycles.csv` is case-by-configuration;
`resources.csv` includes each Stack/Channel/Bank/Plane occupancy and optional
internal serializers. Bandwidth uses HBF host read bytes over total drained
simulation time, not just device-active time. Wall ratios are single concurrent
samples, not precise isolated microbenchmarks. Zero queue rejection does not
imply zero Bank or transfer waiting.

The simulator reports `Total cycles` at DAG completion and separately reports
`Memory drained at cycle`; issued requests must drain successfully. Resident
dirty L2 lines are not implicitly flushed. These are timing simulations without
numerical payload validation. Hypothetical mixed-memory predictions must not be
reported as error against measured FullHBM H100 hardware.

See the [implementation and evaluation report](../validation/hbf-controller-20261022.md).
