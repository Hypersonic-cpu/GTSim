# C++ execution/memory refactor validation

Validation was run from the `6d37892` source snapshot plus the uncommitted
changes in this change set. The scope is limited to Execution IR, the C++
streaming frontend, MemorySystem routing/prefetch below L2, and HBM/HBF
backends. No GPU, L2 cache, MSHR, SM, scheduler, subpartition, TMEM, TMA, NoC,
or GPU timing source was modified.

## Implemented

- Dense vectors/bitmaps replace integer `map`/`set` state in the graph builder;
  opcode lookup is a resident string-view hash table and dtype parsing is done
  once per instruction.
- Streaming manifest membership/sequence checks use unordered sets while
  ordered placement policy state remains ordered.
- MemorySystem reuses D2D request/response scratch buffers and uses dense node
  to read-alias bindings. D2D link output-parameter APIs preserve the old
  value-returning wrappers for callers that need them.
- EventLab reuses stack/channel admission scratch and uses direct `.at()` lookup
  for native completion IDs; request batching and admission order are unchanged.
- HBF read-SRAM slots use a vector and transactional per-request deltas instead
  of copying every plane/stack counter. Ready-page membership is unordered;
  region submission order remains the ordered region-map order.

## T1--T5 actual simulations

Every command returned `0` and the cycle value below was read from the current
run's `result.json` (`timing.total_cycles`), not inferred from a static report.
Python/C++ CPU are user+system seconds; combined RSS is the sampled simultaneous
peak.

| Case | Configuration | Cycles | Python CPU | C++ CPU | Python RSS MiB | C++ RSS MiB | Combined RSS MiB | Wall s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | two GEMMs, 1 SM, FullHBM | 38,948 | 1.03 | 1.57 | 24.74 | 41.02 | 65.77 | 2.68 |
| T2 | two GEMMs, 132 SM, FullHBM | 14,914 | 1.00 | 0.79 | 24.46 | 41.27 | 65.73 | 1.82 |
| T3 | 128x256x64 GEMM, 4F1M grouped planes, streaming prefetch | 11,769 | 0.31 | 0.51 | 26.49 | 38.20 | 60.52 | 0.88 |
| T4 | 512x512x512 GEMM, HBFine 4F1M, streaming prefetch | 25,964 | 3.71 | 1.59 | 28.52 | 62.82 | 91.35 | 3.71 |
| T5 | Qwen3-30B-A3B one block, FullHBM | 397,765 | 158.48 | 46.11 | 469.96 | 688.68 | 1,012.70 | 153.32 |

The cycles are identical to the previously validated values: T1 38,948;
T2 14,914 (=6,824+8,090); T3 11,769; T4 25,964; T5 397,765.

## Checks

- Release build: `cmake --build build --parallel $(nproc)` — passed.
- Debug build: `build-debug` with RAMULATOR2 and IR enabled — passed.
- Targeted CTest regression: 19/19 passed (including Ramulator, HBFine,
  EventLab/FlashLab online contract, graph sync, and reference regression).
- Frontend unit tests: 24/24 passed.
- Debug direct backend/IR smoke tests, including profile-argumented Ramulator
  and HBF serving tests — passed.
- `scripts/check_workload_profile.py` still reports the four pre-existing
  workload-builder timing lines in `fa3_h100.cpp`/`gemm_h100.cpp`; those files
  were not changed. The full CTest aggregate also contains unbuilt optional
  EventLab experiment executables, so those entries are reported as not run.
