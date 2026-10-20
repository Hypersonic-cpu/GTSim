# TVM optimization across HBM/HBF profiles

This extension combines the native TE/MetaSchedule compiler pipeline with the
device allocation and memory routing from `tvm-with-hbf`. The user workload
remains `workloads/gemm/te_gemm.py`, with only FP16 matrix multiplication,
FP32 reduction, and an FP16 output cast. The TVM pin is unchanged. No simulator
timing, bandwidth, or latency values are overridden by the frontend.

After the [pinned TVM setup](tile-ir.md#fresh-workspace-setup-and-phase-1-smoke),
build and validate from the repository root:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGTSIM_TVM_PYTHON="$PWD/.tvm-venv/bin/python"
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
```

## Compiler and memory semantics

The selected YAML is parsed by GTSim; its read-only hardware query supplies
the topology, available tiers, resource limits, bandwidth availability, and
optional device capacities. FullHBM places A/B/C in `global.hbm`. Hybrid
profiles place A/C in `global.hbm` and the original B in `global.hbf`.
The compiler may allocate a separate `global.hbm` copy of B. Allocation uses
disjoint address apertures and the existing topology-specific memory paths.
FullHBF is rejected because it cannot satisfy A/C placement. CoLocated and
Cascaded routing use the paths already defined by the selected configuration;
the compiler contains no stack-ratio table.

Hybrid scheduling adds a native sampled `gtsim.prefetch` decision to the
existing tile, pipeline, warpgroup, and persistent-grid search space. The
cost model learns from measured simulated cycles, including this decision.
An initial stratified exploration phase requires four measurements per legal
memory strategy for the 32-trial budget, then uses the learned cost model.
Smaller budgets scale this requirement down. This prevents regression on a
single strategy from excluding the other; native MetaSchedule still chooses
every candidate's tiling, pipeline, mappings, and trace mutations.
Supported choices are direct HBF consumption and whole-B staging. Partial-B
staging is not implemented in this initial extension.

For staging, `tier_memory.py` emits an explicit TMA copy kernel: HBF→shared
load, transaction-byte mbarrier completion wait, shared async-proxy fence,
shared→HBM store, async-group commit/full-completion wait, and barrier
invalidation. Its rectangular boxes follow sampled BK and the supported
128-byte TMA width. Parallel copy CTAs cover B exactly once. The generated
host TIR contains explicit `gtsim.launch_kernel` and
`gtsim.stream_synchronize` compiler/runtime operations for the copy and GEMM
launches. These denote stream-ordered launches, not executable CUDA runtime
bindings supplied by this repository.

The DAG lowerer consumes these operations, verifies launch arguments and
order, proves each copy store preserves its loaded source region, and adds
completion dependencies from the copy launch to GEMM roots. It also runs the
existing operation-driven checks on each kernel. The materialized DAG has a
distinct HBM B allocation and explicit HBF-read/HBM-write operations. Reads,
writes, TMA issue, shared memory, CTA admission, and link/controller costs are
evaluated by the existing simulator. There is no free migration or fabricated
direct HBF→HBM link.

Optional `hbm.capacity_bytes` and `hbf.capacity_bytes` are enforced by compiler
preflight and C++ materialization. Zero means **unspecified**, preserving the
existing abstract memory profiles. The supplied three YAMLs do not specify
storage capacity; result records therefore expose `capacity_known: false`
unless capacities are supplied. Such results characterize their latency and
bandwidth model and do not establish fit on a physical HBF device. No storage
capacity is inferred from a stack ratio.

## Production experiments and resumption

```sh
.tvm-venv/bin/python scripts/experiments/run_tvm_memory_search.py \
  --trials 32 --seed 0 \
  --out-dir validation/results/tvm/memory-auto-search
```

Four independent optimizer processes run concurrently, one per GEMM shape:
512×512×512, 1024×1024×512, 2048×2048×512, and 1024×1024×1024. Each tunes
FullHBM, 1F4M, and 3F2M sequentially, using one simulation worker and one TVM
search thread. BLAS/OpenMP worker counts are limited to one. Each experiment
receives the same budget and seed, with its own database, feedback model,
traces, native IR, and simulator output. Best schedules are replayed and
numerically checked before advancing to the next configuration.

Rerun the same command to resume. Completed experiments are reused only with
matching source/configuration hashes and search settings. Interrupted native
searches restart deterministically from the seed, replaying cached completed
measurements into MetaSchedule's feedback stream. Unfinished artifacts and
previous databases are archived in `resume-archives/`; compiled/simulated
candidates with matching native scheduled IR are reused. This is measurement
replay, not a checkpoint of the evolutionary algorithm's internal state.

For one experiment:

```sh
.tvm-venv/bin/python scripts/tvm/tune_gemm.py \
  --m 512 --n 512 --k 512 --trials 32 --seed 0 --jobs 1 \
  --memory-config configs/memconf/colocated-1F4M.yaml \
  --out-dir validation/results/tvm/one-memory-search
.tvm-venv/bin/python scripts/tvm/replay_gemm.py \
  --search-dir validation/results/tvm/one-memory-search
```

The batch produces `best-results.csv/json`. Slowdown is calculated separately
for each shape against that shape's optimized FullHBM cycles. Per-candidate
records include device request/read/write byte counts and average latency;
raw simulator logs retain link statistics. `migration_payload_bytes` records
logical B staging payload separately from cache/controller traffic. The
production plot is generated by a plotting script outside the repository and
saved beside the CSV.

`test_tvm_memory_search.py` checks numerical interpretation, placements,
transaction payloads, charged HBM writes, copy-to-GEMM launch dependencies,
missing synchronization rejection, and capacity rejection at both boundaries.
The existing single/multi-wave tests and deterministic FullHBM references are
retained. Numerical checks are software instruction interpretation with exact
integer test inputs, not CUDA execution. All reported performance is simulated.
