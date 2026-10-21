# Simulator host-runtime optimization

Evaluated on `dev-ramulator2`, starting from `dae4c83`. This work changes host
execution overhead, with no changes to modeled hardware, workload graphs,
calibration, scheduling policy or Ramulator.

## Obvious overhead removed

| Location | Before | After |
|---|---|---|
| `include/memory.h`, SRAM/TMEM completion | Rebuild both pending vectors every cycle, copying nested node-ID vectors. | Stable in-place compaction; identical completion order, including non-ordered deadlines. |
| `include/memory.h`, L2 completion | Scan and copy every pending completion on every cycle. | Drain a deque's ready prefix. Both producers append at the current cycle plus the same fixed L2 latency, so deadlines are nondecreasing. |
| `include/simulator.h`, initialization | Scan the full DAG separately for each SM and each of its four subpartitions, then scan again for CTA initialization. | Partition nodes by SM once, partition each SM's nodes by subpartition, and reuse the SM lists for CTA initialization. Preserve DAG order and legacy broadcast-node behavior. |
| `src/scheduler.cpp`, collective issue | Allocate four temporary vectors on each four-warp readiness attempt. | Four fixed-size stack arrays, after the existing four-warp guard. |
| `src/scheduler.cpp`, GTO | Allocate and sort candidates before trying the greedy warp. | Rank only if the greedy attempt fails, retaining the original comparator and tie behavior. |

The simulator still advances cycle by cycle. No event skipping, operation
reordering, latency adjustment or CTA dispatch-policy change was introduced.
Other possible opportunities, such as caching address expansion or skipping
idle SM work, remain outside this small pass.

## Measurement

The original Release executables are preserved in `build/perf-baseline`; the
optimized Release executables are in `build/perf-optimized`. They use the same
compiler, `-O3 -DNDEBUG`, hardware configuration and workload arguments.

`scripts/benchmark_simulator.py` runs each binary once to warm up, then takes
three sequential samples per binary with alternating run order. Times cover
the entire executable, including graph construction, initialization, simulation
and reporting. The table reports medians; host speedup is before / after time.
Every warm-up and measured run must have byte-identical stdout and stderr,
covering simulation cycles, completion totals, memory traffic and tile statistics.

| Case | Simulated cycles (both) | Before seconds | After seconds | Host speedup |
|---|---:|---:|---:|---:|
| gemm_1024 | 23482 | 0.9305 | 0.6019 | 1.55x |
| gemm_2048 | 26717 | 2.0143 | 1.1317 | 1.78x |
| fa3_8h_1024 | 29090 | 2.0519 | 1.1292 | 1.82x |
| fa3_16h_1536 | 83402 | 9.1317 | 4.6601 | 1.96x |
| gemm_colocated | 88701 | 1.9597 | 1.5879 | 1.23x |
| fa3_colocated | 60291 | 2.6415 | 1.6917 | 1.56x |

FA3 arguments are batch, query heads, KV heads, query length and KV length:
`fa3_8h_1024` = `1 8 8 1024 1024`; `fa3_16h_1536` =
`1 16 16 1536 1536`. GEMM cases are `1024 1024 1024` and
`2048 2048 512`. The two additional CoLocated 3F2M cases use GEMM 1024 cubed
and FA3 `1 8 8 1024 1024`, with the reference builders' unchanged placements.

## Verification and reproduction

**8/8 CTests pass** in both the isolated optimized build and the refreshed
normal `build/`, including the new completion-queue/partition test. All
**14 validation cases** (8 GEMM, 6 FA3) produce byte-identical before/after
result files. GEMM MAPE remains **5.70198%**, FA3 **4.95376%**. The workload-profile
check passes; deterministic references remain **GEMM 12,249 / FA3 7,212 cycles**.

Focused tests cover stable SRAM/TMEM compaction against the original scanning
algorithm, queue backpressure, bank conflicts, equal-deadline L2 hit/miss
completions, zero-latency completion phases, SM/SP order and legacy broadcast
ownership. The existing topology tests cover heterogeneous controller routing.

Build the baseline from the unmodified starting commit before building the
optimized tree. This branch's CMake references Glaze headers but the Glaze
checkout/gitlink is absent; both local builds used the same existing Glaze
headers through `-DCMAKE_CXX_FLAGS=-I<existing-glaze>/include`. No dependency
source or Ramulator files were copied or changed.

```sh
python3 scripts/benchmark_simulator.py \
  --baseline-build-dir build/perf-baseline \
  --optimized-build-dir build/perf-optimized \
  --out-dir validation/results/simulator-perf-20261021/benchmark --repeats 3
ctest --test-dir build/perf-optimized --output-on-failure
python3 scripts/check_workload_profile.py
python3 scripts/run_reference.py --build-dir build/perf-optimized --verify
python3 scripts/run_validation.py --build-dir build/perf-optimized
```

Raw logs, individual timing samples, exact commands and executable hashes are
in `validation/results/simulator-perf-20261021/benchmark/`. Source backups from
before editing are in the adjacent `before-sources/` directory. Wall-clock
speedups are local measurements and depend on workload and host load.
