# Optional Ramulator2 HBM3 backend

The H100 GEMM and FA3 reference builders can select a timing-only Ramulator2
HBM backend. The default remains `naive`; the shared L2/MSHR model is
unchanged. The separate [HBFine adapter](hbf-controller.md) supplies optional HBF timing. There are no modeled L2 slices, partitions, or slice contention.

## Build and run

Use the submodule revision registered by the current branch. Python bindings,
TVM, CUDA, and numerical payloads are not needed:

```sh
git submodule update --init 3rdparty/ramulator2
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGTSIM_ENABLE_RAMULATOR2=ON
cmake --build build --parallel
./build/gtsim_gemm configs/v0.1.0/h100-sxm.yaml --memory-config configs/ramulator/h100-sxm-ramulator.yaml 512 512 512
./build/gtsim_fa3 configs/v0.1.0/h100-sxm.yaml --memory-config configs/ramulator/h100-sxm-ramulator.yaml 1 8 8 512 512
ctest --test-dir build --output-on-failure -j2
```

Ramulator fetches its declared yaml-cpp and fmt dependencies during CMake
configuration. Standard `FETCHCONTENT_SOURCE_DIR_*` overrides can use local
copies of those same versions. `cmake/Ramulator2.cmake` applies a required
dependent-template spelling fix through a build-directory header overlay and
disables fmt 10 consteval checking on AppleClang 21+. Neither workaround changes
the submodule checkout. Library products also stay in the build directory.

The generated timing profile is committed. Regenerate it from the checked-out
Ramulator Python DSL with `python3 scripts/ramulator/export_h100.py`; this does
not compile or load Python bindings. Config paths resolve relative to their
containing GTSim YAML file. An unavailable compiled backend, malformed profile,
unsupported topology, transaction size, or out-of-capacity address fails
explicitly. The adapter supports FullHBM and CoLocated H100-derived profiles with 1–5 HBM stacks.
The default five-stack profile is unchanged; optional two/four-stack profiles
are generated with `--stacks 2` or `--stacks 4` using the same per-stack timing.
Use the corresponding `configs/memconf/*_ramulator_patch.yaml` after a mixed
profile. These hypothetical reduced-stack systems have no measured H100 target.

## Topology and mapping evidence

[NVIDIA's Hopper architecture description](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
documents five active HBM3 stacks, ten 512-bit controller groups, and 80GB on
H100 SXM. The [HBM3 organization in Ramulator's registered DSL](../3rdparty/ramulator2/python/ramulator/dram/hbm3.py)
models 16 physical channels per stack and two 32-bit pseudochannels per channel.
Thus the adapter instantiates 80 HBM34 controllers, each owning one channel and
two pseudochannels. The ten logical GPU MC groups are metadata: eight physical
channels per group. They add no second controller queue or contention model.

The [Accel-Sim H100 profile](https://github.com/accel-sim/gpgpu-sim_distribution/blob/dev/configs/tested-cfgs/SM90_H100/gpgpusim.config)
uses 40 memory partitions, two subpartitions each, `IPOLY_MODULO`, and `dramid@8`.
Its [address decoder](https://github.com/accel-sim/gpgpu-sim_distribution/blob/dev/src/gpgpu-sim/addrdec.cc)
hashes a partition/subpartition index and further folds routing by the issuing
SM's chiplet. These abstractions are not a documented bijection to physical
HBM3 channels and pseudochannels. GTSim has neither that L2 hierarchy nor
issuing-SM routing metadata at this interface, so a faithful translation is not
available. The adapter deliberately does not claim to reproduce that hash.

The reproducible fallback is **128B line striping**, an explicit modeling
assumption built on the documented physical hierarchy:

```text
channel = floor(physical_byte_address / 128) mod 80
local_byte_address = floor(physical_byte_address / (128*80))*128 + address mod 128
stack = floor(channel / 16)
logical_mc_group = 2*stack + floor((channel mod 16) / 8)
local bits, low to high: byte[5], PC[1], burst-column[5], bank[2], BG[2], SID[1], row[14]
```

Channel removal uses quotient/remainder, not a power-of-two mask, and is
reversible. `PassThroughChannelMapper` and `PassThroughAddrMapper` consume the
explicit native address vector; no subsequent mapper can overwrite it. A 128B
line becomes four 32B bursts across both pseudochannels of the same physical
channel. The 16Gb/8-high organization gives 1GiB per channel and 80GiB overall;
this is the binary-capacity interpretation of the advertised 80GB. NVIDIA's
private physical hash, bank mapping, SID layout, and page policy remain unknown.

## Timing provenance and assumptions

[NVIDIA's current H100 specification](https://www.nvidia.com/en-us/data-center/h100/)
lists 3.35TB/s for SXM. Ramulator provides `HBM3_6400Mbps` with a 625ps full CK,
32-bit pseudochannels, and BL8 (32B transactions). That unmodified speed bin
would provide 4.096TB/s over five stacks and is not used at that operating rate.

The exporter conservatively downclocks this existing bin to the published
aggregate bandwidth: four transfers per DQ per CK, 5120 DQ bits overall,
`CK = ceil_even_ps(4*5120*1000/(8*3350)) = 766ps`. This gives a **383ps
half-CK tick** and **3342.037GB/s** peak pin bandwidth (0.24% below 3350 due to
integer-ps rounding). Non-refresh timing minima retain the preset's CK counts;
they therefore become longer in ns, not faster. Refresh periods and all-bank
refresh latency are recomputed using the DSL's existing JEDEC formulas. The
per-bank refresh and other device timings remain those of that conservative bin.
This projection is a shared operating-point assumption, not measured H100
DRAM timing or a fit to kernel runtimes.

HBM34 per-bank refresh, FR-FCFS, open rows, and the existing 32-entry per-channel
read/write queues are explicit controller assumptions. The core remains at
1980MHz. Integer clock phase conversion advances every elapsed DRAM half-CK
tick; it never rounds an independent per-core-cycle ratio or doubles CK twice.
The existing GPU-to-HBM link remains identical between comparison runs
(3100GB/s, 10 core cycles by default), as do L2 latency and all compute settings.
The naive HBM latency/bandwidth fields are ignored by the Ramulator device.

## Request lifecycle

`MemoryBackend` carries parent ID, physical address, byte count, operation,
original tensor tier, and optional L2/MC/channel routing hints. MemorySystem
chooses the physical device before admission. Retaining the original tier in
responses preserves FullHBM overrides of HBF-tagged tensors and MSHR identity.

The bounded adapter splits aligned 32B-multiple parents into 32B bursts. It
retries each physical channel independently after Ramulator queue rejection.
Rejected bursts retain ownership and identity. Callback bookkeeping joins all
bursts before emitting exactly one parent response. A request cannot dispatch
before its arrival edge. In this submodule revision read callbacks include
read-data latency, while write callbacks occur at WR issue. The adapter delays
write acknowledgements by the profile's `nCWL+nBL`; same-address traffic waits
for that data completion to avoid early write coalescing/forwarding callbacks.
Timing-only writes need no payload or numerical correctness claims.

After the final DAG completion, the simulator drains all issued requests,
eviction writebacks, links, and pending L2 completions. `Total cycles` remains
the final DAG cycle for compatibility; `Memory drained at cycle` reports the
separate end of issued traffic. Resident dirty cache lines are not implicitly
flushed. Internal periodic refresh need not stop for the user-request drain.
CLI timeouts and drain failures now produce a nonzero exit status.

## Tests and experiments

Tests cover reversible byte addressing and boundaries, all 80 channels/160 PCs/
10 groups, balanced distribution, fractional clocks, aligned 32/64/128B
reads/writes/writebacks, adapter and controller backpressure, callback identity,
exact parent/burst/byte conservation, FullHBM placement overrides, and dirty
eviction drain. Smoke tests run tiny GEMM and FA3 through the actual builders.
They verify timing simulation completion, not numerical GEMM/attention results.

```sh
python3 scripts/ramulator/compare_backends.py --build-dir build --out-dir validation/results/ramulator-comparison
```

The runner uses at most two simulator processes: GEMM ascending in size, FA3
descending. It retains commands, status, cycles, wall time, CPU time, peak RSS
where supported, and raw logs; `comparison.csv`, `comparison.md`, and
`summary.json` use the same completed measured cases for both MAPEs. Negative
change in absolute error means improved accuracy. Wall ratios above one mean
more host time. `--naive-build-dir` can preserve original baseline executables;
`--phase` allows separate baseline/new runs and regenerating reports.

See the [H100 evaluation report](../validation/ramulator-hbm3-20261021.md)
for the measured subset, per-case errors, host costs, and discrepancy audit.

This is a DRAM backend extension to the release scope, not a general frontend,
automatic scheduler, or a newly calibrated H100 model. Remaining uncertainties
include NVIDIA's private mapping, DRAM timings, controller policy, unsliced L2
contention, and the shared pre-existing link/TMA/compute approximations. Hardware
microbenchmarks should resolve those before further parameter changes.

Ramulator2 and the generated timing data are MIT licensed; retain
[`3rdparty/ramulator2/LICENSE`](../3rdparty/ramulator2/LICENSE) when redistributing
them. No simulator source or timing data is copied from Accel-Sim.
