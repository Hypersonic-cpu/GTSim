# Ramulator2 HBM3 integration: H100 SXM evaluation

Implemented and tested on `dev-ramulator2`, using its registered Ramulator2 revision without changing the submodule. The original naive binaries and touched sources are retained in `validation/results/ramulator-hbm3-20261021/baseline/`. No TVM build, scheduling change, L2 slicing, or per-workload calibration was introduced.

## Implementation and shared configuration

- `include/memory_backend.h`, `src/memory_backend.cpp`, and `src/ramulator_backend.cpp`: selectable naive/Ramulator HBM interface, physical addresses, optional routing hints, bounded admission, 32B burst splitting, asynchronous completion, and counters from the actual DRAM controllers.
- `include/hbm_address.h` and `include/memory_clock.h`: reversible 128B channel striping and exact integer core/DRAM phase conversion, including coincident arrival edges.
- `include/memory.h` and `src/simulator.cpp`: preserve L2/MSHR semantics, route through the selected device, and drain issued traffic after the kernel. Kernel cycles remain the last DAG completion; drain time is reported separately.
- `include/mem_conf.h`, `include/arch_conf.h`, `configs/ramulator/`, and `scripts/ramulator/export_h100.py`: explicit backend/topology/profile selection and reproducible timing export from the registered DSL.
- `cmake/Ramulator2.cmake` and `CMakeLists.txt`: optional backend, source-preserving compiler compatibility overlay, and focused tests. The CLI mains only gain ownership cleanup and error/timeout exit handling; reference graph builders are unchanged.
- `scripts/ramulator/compare_backends.py`: at most two simulation processes, raw logs, commands, binary/source/configuration hashes, wall/CPU time, RSS, paired-case errors, and CSV/Markdown reports.

Both backends use the same 1980MHz core, reference graphs, CTA mapping, two-stage schedule, shared L2 settings, and existing 3100GB/s + 10-cycle GPU-to-HBM link. Naive retains 270-cycle latency and 3100/1550GB/s read/write parameters. Ramulator replaces those device parameters with HBM3 command timing; HBF is unchanged.

The new topology is five stacks, ten logical MC groups, 80 physical channels, and 160 pseudochannels. The capacity-consistent 16Gb/8-high organization is an assumption. Ramulator's existing 6400Mbps bin is conservatively downclocked to the published 3.35TB/s operating point: 766ps CK, 383ps half-CK, 3342.037GB/s pin bandwidth after rounding. Refresh follows the DSL formulas; other CK minima are retained. This is not a measured H100 timing table. Accel-Sim's partition/hash hierarchy cannot be faithfully translated to physical channels here, so the explicit interleave is also an assumption. See [mapping and timing rationale](../docs/ramulator-hbm3.md) for primary sources and details.

## Accuracy and host runtime

Positive signed error means slower than measured hardware. Negative Δ absolute error means improvement. Host wall time includes process launch, graph construction, simulation, drain, and captured output. Each row is one paired run; two workload families may overlap, so these are host-cost observations rather than a statistical performance study.

| Case | Measured | Naive | Ramulator | Naive error | Ram error | Δ abs error (pp) | Naive wall (s) | Ram wall (s) | Ram/naive wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemm_512_512_512 | 13143 | 12249 | 12821 | -6.80% | -2.45% | -4.35 | 0.228 | 0.449 | 1.97× |
| gemm_512_512_1024 | 24072 | 21473 | 22625 | -10.80% | -6.01% | -4.79 | 0.381 | 0.690 | 1.81× |
| gemm_1024_1024_512 | 13296 | 14258 | 15925 | +7.24% | +19.77% | +12.54 | 0.384 | 0.812 | 2.12× |
| gemm_1024_1024_1024 | 24123 | 23482 | 27547 | -2.66% | +14.19% | +11.54 | 0.648 | 1.390 | 2.14× |
| fa3_1_8_8_1536_1536 | 44766 | 41712 | 43334 | -6.82% | -3.20% | -3.62 | 2.550 | 3.639 | 1.43× |
| fa3_1_8_8_1024_1024 | 29771 | 29090 | 29963 | -2.29% | +0.64% | -1.64 | 1.198 | 1.885 | 1.57× |
| fa3_1_8_8_512_512 | 17052 | 16469 | 16217 | -3.42% | -4.90% | +1.48 | 0.437 | 0.739 | 1.69× |

| Paired set | Cases | Naive MAPE | Ramulator MAPE |
|---|---:|---:|---:|
| GEMM | 4 | 6.87% | 10.61% |
| FA3 | 3 | 4.18% | 2.91% |
| Combined | 7 | 5.72% | 7.31% |

Peak RSS (MiB), retained per process with `wait4`:

| Case | Naive | Ramulator |
|---|---:|---:|
| gemm_512_512_512 | 28.0 | 298.3 |
| gemm_512_512_1024 | 37.4 | 322.0 |
| gemm_1024_1024_512 | 98.2 | 404.6 |
| gemm_1024_1024_1024 | 136.7 | 456.8 |
| fa3_1_8_8_1536_1536 | 641.1 | 978.7 |
| fa3_1_8_8_1024_1024 | 307.7 | 639.2 |
| fa3_1_8_8_512_512 | 102.0 | 417.2 |

## Verification and discrepancy audit

Both tiny smoke cases terminated and drained: GEMM 128×256×64 took 3834 kernel/drain cycles (896 parents, 3584 bursts); FA3 B1/H1/S128 took 6741 cycles (1024 parents, 4096 bursts). These are timing-only executions, not numerical-output checks. The Ramulator build passes all 12 CTests, including existing deterministic references; the optional-backend-off build passes all 9 tests. Profile-boundary and generated-profile reproducibility checks pass. Requesting Ramulator from the disabled build returns a clear error and status 1.

GEMM MAPE exceeds 10%, so the integration was audited before interpreting model limitations. For every selected case: parent accepted/completed counts agree; burst submitted/completed counts agree with Ramulator controller accepted/served reads plus writes; bytes equal 32×bursts; all 80 channels carry traffic; and elapsed DRAM ticks equal floor(drain_cycle×10^9/(1980000×383)). Focused tests force queue rejection/retry and cover 32/64/128B reads/writes/writebacks, callback uniqueness, same-address ordering, high-address uniqueness/capacity, fractional and coincident clocks, placement overrides, and dirty eviction drain. The initial high-address test caught and corrected a capacity-inconsistent smaller organization before the reported experiment.

| Case | Parent requests | 32B bursts | Channel burst min/max | Admission retries |
|---|---:|---:|---:|---:|
| fa3_1_8_8_1024_1024 | 65536 | 262144 | 3276/3280 | 686489 |
| fa3_1_8_8_1536_1536 | 98304 | 393216 | 4912/4916 | 1145815 |
| fa3_1_8_8_512_512 | 32768 | 131072 | 1636/1640 | 271679 |
| gemm_1024_1024_1024 | 49152 | 196608 | 2456/2460 | 665422 |
| gemm_1024_1024_512 | 32768 | 131072 | 1636/1640 | 399403 |
| gemm_512_512_1024 | 20480 | 81920 | 1024/1024 | 124371 |
| gemm_512_512_512 | 12288 | 49152 | 612/616 | 163777 |

The two 1024×1024 GEMMs have +19.77% and +14.19% cycle error. The channel totals are balanced; the 1024³ run records 665422 admission retries, 189244 row hits, 7224 row misses, and 140 row conflicts. Retries count repeated admission attempts, not unique stalled GPU cycles. The evidence supports queue pressure in this model, but does not establish which private H100 policy or remaining simulator approximation explains the hardware discrepancy. No timing or queue parameter was adjusted against these errors.

## Reproduction and limits

Build and timing provenance are in the per-run JSON and `provenance_all.json`; logs, `comparison.csv`, `comparison.md`, and `audit.json` are under `validation/results/ramulator-hbm3-20261021/`. The checkout has a pre-existing missing Glaze gitlink; this machine used the already available sibling Glaze headers through `CMAKE_CXX_FLAGS`. The long fmt Git clone was stopped and replaced by the exact declared 10.2.1 release archive through CMake's standard source override. These local paths/settings are recorded in provenance; no dependency revision was changed.

```sh
cmake -S . -B build/ramulator-hbm3 -G Ninja -DCMAKE_BUILD_TYPE=Release -DGTSIM_ENABLE_RAMULATOR2=ON
cmake --build build/ramulator-hbm3 --parallel 6
ctest --test-dir build/ramulator-hbm3 --output-on-failure -j2
python3 scripts/ramulator/compare_backends.py --build-dir build/ramulator-hbm3 --naive-build-dir validation/results/ramulator-hbm3-20261021/baseline --out-dir validation/results/ramulator-hbm3-20261021 --phase all
```

All seven selected cases completed; no selected case failed or timed out. Explicitly unrun measured cases: GEMM (512,1024,512), (512,1024,1024), (1024,512,512), (1024,512,1024); FA3 B1/H16/S512, S1024, S1536. Larger sweeps, non-FullHBM Ramulator topologies, numerical payload validation, and installation/package validation were not performed.

Next steps require independent memory microbenchmarks or published reverse engineering for the private channel/bank hash, H100 operating timings, controller queue/write-drain policy, and L2 contention. The present backend is a working, conserved HBM3 timing model; its mixed accuracy results do not justify replacing the default calibrated naive model.
