# GTSim — Qwen3-235B-A22B FlashAccel Full Sweep Execution Plan

## 0. Goal and Boundaries

Repository: `Hypersonic-cpu/GTSim`  
Branch: `sim-validation`  
Reviewed baseline: `065e75ca64e6526da9d95e83c4e48970a72b9045`  
Reference: uploaded `2607.10186v2.pdf` (FlashAccel).

**Execute the implementation described here. Do not redesign the experiment or add unrequested optimizations.**

Evaluate Qwen3-235B-A22B decode at three context lengths:

- 10,000 tokens
- 100,000 tokens
- 200,000 tokens

For each context, independently optimize:

1. Full-HBM (0F5M).
2. FlashAccel 4F1M, full-capacity SRAM pool.
3. FlashAccel 4F1M, strict double buffering.

All legal combinations must be submitted to **one concurrent sweep queue**, with at most **150 running simulations**.

After choosing the optimal tiling for each context and architecture, run the Prefetch-OFF controls and generate bandwidth curves.

**The measured workload is one complete Qwen3-235B-A22B Transformer Decode Block, B=1**, not the entire 94-layer model.

Do not modify GPU Core, SM, L1, L2, NoC, HBFine internals, FlashLab, or EventLab.

Do not add backend patches, hidden simulator shortcuts, runtime validation hashes, or artificially favorable hardware parameters.

---

## 1. Fixed Workload and Three Context Lengths

Use the existing source:

`frontend/asm/qwen3/flash-accel/one_block.progir`

Use the existing parameter file as the base:

`frontend/asm/qwen3/qwen3-235b-a22b-b1.json`

Keep all model dimensions and routing information identical:

```python
B = 1
H = 4096
HQ = 64
HKV = 4
D = 128
E = 128
TK = 8
I = 1536
S = 4
TILE = 256
NUM_SMS = 132

CONTEXT_LENGTHS = (10_000, 100_000, 200_000)
```

Only override `CONTEXTS` in memory:

```python
params = load_existing_params()
params["CONTEXTS"] = [context_length]
```

Do not create manually maintained files such as:

- `qwen235b_ctx10k.json`
- `qwen235b_ctx100k.json`
- `qwen235b_ctx200k.json`

Add a CLI override to `frontend/tools/run_qwen_simulations.py`:

```text
--context-length 10000|100000|200000
```

It must override the input parameter dictionary before `compile_program()`.

Retain the original `ROUTE_IDS`, `ROUTE_PROBS`, BF16 precision, attention split count, and all other settings.

The three contexts must perform exactly the same model operations except for the longer KV-cache attention workload.

Use the existing streaming execution path. Do not materialize a full Program-wide Native Instruction Graph in memory.

---

## 2. Physical Hardware Configuration

### 2.1 4F1M

Create:

`configs/memconf/4F1M_flashaccel_32m.yaml`

Copy `4F1M_grouped_planes.yaml` and change only:

```yaml
hbf:
  controller:
    sram_bytes_per_plane: 32768
    base_die_sram_bytes_per_stack: 8388608
```

These are partial YAML fields; retain all other original configuration fields.

The final physical geometry must be:

| Parameter | Value |
|---|---:|
| HBF stacks | 4 |
| HBM stacks | 1 |
| Flash dies/stack | 8 |
| Planes/die | 96 |
| Planes/command group | 32 |
| NAND page | 4 KiB |
| NAND sense latency | 4 µs |
| Per-plane private SRAM | 32 KiB |
| Base-die SRAM/stack | 8 MiB |
| Total SRAM/stack | 32 MiB |
| Total SRAM across 4 stacks | 128 MiB |
| HBF read bandwidth/stack | 768 GB/s |

Preserve existing grouped-plane address mapping, NAND timing, SRAM timing, links, and GPU architecture.

Use the HBFine backend only.

### 2.2 Full-HBM

Use:

`configs/memconf/0F5M_channels.yaml`

- 5 HBM stacks.
- Aggregate 3100 GB/s HBM read bandwidth.
- No HBF Prefetch.
- No HBF SRAM capacity restrictions.
- No HBF N-Slab Partition.

Use the same GPU core model as 4F1M.

The single-block workload fits in the configured HBM, but the complete 235B model does not. Do not present 0F5M as a feasible single-device full-model deployment.

---

## 3. SRAM Policies and Partitioning

### 3.1 Modify `partition.py`

File:

`frontend/src/compile/opt/partition.py`

Current restrictions:

- Only `n_slab_fit_double_buffer`.
- Only `buffer_count=2`.
- Only fixed `16×64×16, stages=2`.

Replace the policy restrictions with:

```python
partition_policy = "n_slab_fit_sram"
buffer_count in (1, 2)
```

Keep the old policy name compatible with existing experiments.

### 3.2 Pool Mode

`mode="pool"`:

```python
buffer_count = 1
```

Allow each Reservation to use the full physical SRAM capacity:

- 8 pages/Plane.
- 8 MiB Base Die/Stack.
- 32 MiB aggregate/Stack.

No explicit alternating Slot ownership.

Multiple Reservations may coexist if the actual allocator accepts them.

### 3.3 Double Mode

`mode="double"`:

```python
buffer_count = 2
```

Use the existing `PrefetchBuffer`/Epoch mechanism and the current `ReadSramAllocator`.

Enforce:

- Maximum 4 pages per Plane per Slot.
- Maximum 4 MiB Base Die SRAM per Stack per Slot.
- Maximum 16 MiB total per Stack per Slot.
- No reuse until the previous Epoch has released the Slot.

Both modes must use the **same physical 32 MiB/Stack SRAM**.

### 3.4 N-Slab Partition

Reuse `_partition_region()` and its existing physical page-footprint calculation.

For every HBF weight GEMM:

1. Determine its actual accessed Expert/Row ranges.
2. Calculate the footprint of each candidate N-slab.
3. Map every page to its physical Stack/Plane.
4. Allocate per-Plane SRAM first.
5. Spill excess pages into the Base Die SRAM.
6. Find the largest N-slab that fits the selected policy.
7. If necessary, split into multiple consecutive N-slabs.
8. Execute one partial GEMM for each slab.
9. Preserve full K reduction and the original parent row stride.
10. Store each partial output at the correct N offset.

Do not introduce Split-K or physically duplicate parent tensors.

Capacity must be checked against the actual mapped page footprint, not merely the Tensor's byte size.

Expected sanity checks for B=1:

| Weight | Approximate Size | Pool | Double |
|---|---:|---|---|
| One Expert GateUp | 24 MiB | Fit | Fit |
| One Expert Down | 12 MiB | Fit | Fit |
| QKV Projection | 72 MiB | Fit when alone | Requires partitioning |
| Output Projection | 64 MiB | Fit when alone | Capacity boundary |

These are expected capacity cases, not hardcoded partition rules. Final partition counts must come from the allocator and compiler.

---

## 4. Implement Real GEMM Tiling Variants

### 4.1 Candidate Space

**Define all candidates directly inside the Python Sweep Launcher.**

```python
BM_VALUES = (16, 32)
BN_VALUES = (64, 128, 256)
BK_VALUES = (16, 32, 64, 128, 256)

STAGES = (2,)
SPLIT_K = (1,)

TILING_CANDIDATES = tuple(
    (bm, bn, bk, stages)
    for bm in BM_VALUES
    for bn in BN_VALUES
    for bk in BK_VALUES
    for stages in STAGES
)
```

Total: **30 candidates** per configuration.

Do not create individual Tiling JSON files.

Do not divide these into Coarse and Fine experiments. Test every legal candidate in the single global queue.

BK=128 and BK=256 are exploration candidates, not prevalidated native schedules. Reject them before simulation if their actual generated TMA/MMA/resources or dependency structure is illegal. Record the exact rejection reason; never silently fall back to BK=16.

### 4.2 Structured Compiler Pass

Create:

`frontend/src/compile/opt/gemm_tiling.py`

Implement:

```python
def apply_gemm_tiling(flow, bm, bn, bk, stages=2):
    ...
```

Apply this pass after `compile_program()` and before HBF `partition_flow()`.

Update structured FlowIR for:

- `small_m_gemm_flow`
- `grouped_gemm_flow`

Change all relevant:

- `ScheduleConfig.tile_shape`
- BM/BN/BK-dependent `static.let` expressions
- CTA counts and mapping
- TMA Load shapes and offsets
- MMA tile loop counts
- K loop iteration count
- Output Store shapes and offsets
- SSA result shapes
- Static requirements where appropriate
- SMEM scratch requirements

For Grouped GEMM, preserve:

- Expert assignment
- `IDS`
- `A_MAP`
- `C_MAP`
- Per-Expert output row mapping

Do not modify the `.flowir` source files on disk for each Candidate.

Use immutable AST transformations through `dataclasses.replace()`.

### 4.3 Memory Resources

For the current fixed two-stage Skinny GEMM schedule, start with the existing scratch accounting generalized to:

```python
scratch_bytes = (
    2 * bk * (bm + bn)
    + 4 * bm * bn
    + 512
)
```

Align to 512 bytes.

The original `16×64×16` configuration must remain 7168 bytes.

Verify the resource calculation against the actual Native IR and existing Buffer semantics, including stage/slot usage, before accepting a candidate. Do not assume this formula alone proves that larger BK values fit.

Retain:

```text
warps = 4
consumer_warps = 4
warpgroup = 1
stages = 2
split_k = 1
```

Check native TMA box limits, tile alignment, shared-memory capacity, register usage, MMA fragment coverage and pipeline dependencies.

### 4.4 Native IR Acceptance

Before launching simulations, compile or lower every unique Tiling Candidate.

For each Candidate save an IR summary containing:

```text
BM, BN, BK
CTA Count
K Iteration Count
TMA Load/Store Shapes
MMA Instructions
Shared Memory Footprint
Register Footprint
Output Coverage
```

Reject illegal candidates before runtime scheduling.

Candidate T0 `(16,64,16,2)` must retain the original reference kernel's behavior.

A different Candidate must produce genuinely different Native GEMM schedules, not just a changed metadata label.

---

## 5. Correct FlashAccel Prefetch Scheduling

File:

`frontend/src/compile/opt/prefetch.py`

The existing Routing-dependent logic effectively delays Prefetch until the Consumer's own launch boundary.

Replace that restriction with the earliest legal Router-completion boundary.

### Required Logic

1. Determine the completion boundary of `router_softmax_topk`.
2. Before that boundary, routing-dependent Expert Prefetch is forbidden.
3. After that boundary, the selected Expert Prefetches are eligible to enter the global FIFO queue.
4. Dispatch from FIFO head when SRAM and Buffer Slots become available.
5. Do not force Prefetch to wait for its own Consumer Launch.
6. Release SRAM when the last Consumer completes.
7. Preserve Demand Fallback for pages that are not yet Ready.

For the fixed routing trace:

```text
Router completes
    ↓
Prefetch Expert 98
Prefetch Expert 107
Prefetch Expert 10
...
    ↓
Compute Expert 98
    ↕  NAND background prefetch continues
Compute Expert 107
    ↕
...
```

Retain existing Router dependency and dynamic allocation behavior.

Do not use future routing information to issue hardware Prefetch before Router completion.

### Pool vs Double

Pool:

- Multiple prefetched Regions can coexist according to actual free SRAM.

Double:

- Fill one Slot while the other Slot is consumed.
- Maintain strict ownership and release dependencies.

Do not introduce additional Hardware Prefetch Dispatch Bandwidth.

Do not introduce a second Prefetch Engine.

---

## 6. Modify the Runner CLI — No Sweep JSON Files

File:

`frontend/tools/run_qwen_simulations.py`

Add CLI arguments:

```text
--context-length INT
--gemm-bm INT
--gemm-bn INT
--gemm-bk INT
--gemm-stages INT
--sram-policy none|pool|double
```

Retain the existing:

```text
--source
--params
--runner
--architecture
--memory
--prefetch on|off
--out
--max-cycles
```

Construct `PartitionPolicy` directly in memory based on these flags.

Do not require new JSON policy files.

Preserve `--flashaccel-config` for backward compatibility, but disallow conflicting policy inputs.

The intended pipelines are:

**FullHBM**

```text
compile_program
→ apply_gemm_tiling
→ lower_flow
→ run without HBF Partition or Prefetch
```

**4F1M Prefetch OFF**

```text
compile_program
→ apply_gemm_tiling
→ partition_flow
→ lower_flow
→ run without Prefetch
```

**4F1M Prefetch ON**

```text
compile_program
→ apply_gemm_tiling
→ partition_flow
→ prefetch.optimize
→ lower_flow
→ run with Prefetch
```

Save the complete effective parameters and policy in existing Result/Provenance outputs.

The restriction against sweep JSON files applies to **hand-maintained experiment definitions**. Normal machine-generated `result.json` and `provenance.json` are still required.

---

## 7. Single Global Sweep Launcher

Create:

`scripts/experiments/sweep_qwen235b_flashaccel.py`

All Sweep parameters and experiment generation logic must be in this one Python file.

### 7.1 Fixed Experiment Definitions

Use the following structure:

```python
from itertools import product

CONTEXT_LENGTHS = (10_000, 100_000, 200_000)

BM_VALUES = (16, 32)
BN_VALUES = (64, 128, 256)
BK_VALUES = (16, 32, 64, 128, 256)
STAGES = (2,)

TILINGS = tuple(product(
    BM_VALUES,
    BN_VALUES,
    BK_VALUES,
    STAGES,
))

MAIN_CONFIGS = (
    ("fullhbm", "none", False),
    ("4f1m", "pool", True),
    ("4f1m", "double", True),
)

MAX_CONCURRENCY = 150
PER_CASE_TIMEOUT_SECONDS = 1800
AGENT_PROGRESS_INTERVAL_SECONDS = 900

CASES = tuple(
    (context, hardware, sram_policy, prefetch, tiling)
    for context, (hardware, sram_policy, prefetch), tiling
    in product(CONTEXT_LENGTHS, MAIN_CONFIGS, TILINGS)
)
```

This generates exactly 270 logical main-sweep cases before legality filtering.

Do not write separate Candidate configuration JSON files.

### 7.2 Main Scheduler

Implement an asynchronous, resource-aware Scheduler using:

- `asyncio`
- `asyncio.create_subprocess_exec()`
- `asyncio.Semaphore()`
- `asyncio.wait_for()`
- `start_new_session=True`

Requirements:

1. Generate all 270 logical cases at launch.
2. Perform deterministic compiler feasibility filtering.
3. Put every legal candidate into one global queue.
4. Launch all available independent jobs concurrently.
5. Never exceed 150 running simulations.
6. As soon as a job finishes, launch the next queued job.
7. Do not serialize by Context Length.
8. Do not serialize by SRAM Policy.
9. Do not wait for all 10k runs before starting 100k/200k.
10. Do not use stage barriers or Coarse/Fine stages.

The queue should interleave context lengths and architectures to avoid a whole configuration being delayed until the end.

### 7.3 Resource-Aware Admission

Set `--max-concurrency` default to 150, but calculate the effective launch capacity from:

- CPU capacity.
- Available RAM.
- Measured representative per-process RSS.
- A memory safety budget of 75% of available RAM.

The effective concurrency must satisfy:

\[
C_{\text{effective}}\le 150
\]

If memory or CPU resources are insufficient, keep remaining cases queued rather than oversubscribing the system.

Use one initial T0 smoke/pilot to establish approximate resource needs and validate the execution pipeline, then submit the entire Sweep. Do not use a multi-stage performance search.

### 7.4 Process Management

Launch each job with:

- Dedicated output directory.
- Redirected stdout/stderr.
- 1800-second wall-clock timeout.
- Independent process group.
- Proper cleanup of Python Driver and its C++ runner children.
- No shared mutable input files.
- No duplicate job execution.

Use `asyncio.wait_for()` to wait on process completion without periodic process polling.

On timeout:

1. Terminate the process group.
2. Escalate to SIGKILL if necessary.
3. Preserve logs.
4. Mark `TIMEOUT`.
5. Release the concurrency slot.
6. Continue scheduling the next job.

Do not abort the entire Sweep due to one failure.

### 7.5 Resume and Deduplication

Compute a stable Case ID from:

- Git Commit.
- Runner Binary SHA256.
- Context Length.
- Hardware Configuration.
- SRAM Policy.
- Prefetch Mode.
- BM/BN/BK/Stages.
- Effective Model Parameters.

On restart:

- Skip completed, valid cases with matching fingerprints.
- Allow explicit retry of failed/time-out cases.
- Never reuse mismatched or incomplete results.
- Preserve independent Result/Provenance files.

### 7.6 Build and Polling

Build Release once:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF

cmake --build build --target gtsim_program --parallel
```

Do not rebuild for every Tiling Candidate.

The Scheduler must be event-driven, not a short-interval polling loop.

**The Agent should inspect overall Sweep progress no more frequently than once every 15 minutes**, except on a terminal completion, fatal error or resource emergency.

---

## 8. Main Experiment Matrix

For each context length:

| Configuration | Candidates |
|---|---:|
| FullHBM, Prefetch OFF | 30 |
| 4F1M Pool, Prefetch ON | 30 |
| 4F1M Double, Prefetch ON | 30 |
| **Total per context** | **90** |

Across three context lengths:

\[
90\times3=\boxed{270}
\]

All candidate simulations belong to the same global concurrent queue.

Use:

```text
--max-cycles 1000000000
```

and the separate 1800-second wall-clock timeout.

Do not interpret a simulator cycle-limit exit as a successful completed result.

For every completed case record:

- Context Length
- Hardware
- SRAM Policy
- Prefetch Mode
- BM/BN/BK/Stages
- Number of generated N-slabs
- Total Cycles
- Simulated Latency
- Simulated Block-level Tokens/s
- HBM/HBF Traffic
- Prefetch Queued/Admitted/Completed
- SRAM Resident Read Bytes
- Fallback Read Bytes
- SRAM Peak Occupancy
- Sense Commands
- Wall Time
- Peak RSS
- Return Status

Select the optimal tiling **separately for every context and configuration**, using minimum `total_cycles`.

Expected: 9 independent winners.

---

## 9. Prefetch-OFF Controls

Once a 4F1M Pool or Double winner has been selected for a particular context, automatically enqueue the corresponding Prefetch-OFF control.

Keep:

- Same Context Length.
- Same 4F1M Hardware.
- Same physical SRAM.
- Same SRAM Policy.
- Same BM/BN/BK/Stages.
- Same N-Slab Partition.
- Same Routing Trace.

Only disable Prefetch.

This produces:

```text
3 contexts × 2 SRAM policies = 6 controls
```

Do not compare an optimized Prefetch-ON case against a different tiling in the Prefetch-OFF control.

The Scheduler should enqueue each control when its dependency is satisfied, without waiting for unrelated contexts to complete.

---

## 10. Correct HBM Bandwidth Sampling

File:

`include/memory/naive.h`

Current `HBMModel` records the entire Read Request size when it completes, rather than the actual byte service rate over time.

Correct the observational sampling:

- Record serviced Read Bytes at the cycle they consume HBM bandwidth.
- Record serviced Write Bytes at the cycle they consume HBM bandwidth.
- Preserve fractional-credit precision.
- Do not change Read/Write Arbitration, latency, bandwidth, queues or completion timing.
- Keep sampling disabled during the primary Sweep.

Use:

```text
statistics_interval_ps = 1_000_000
```

for final bandwidth runs (1 µs bins).

Reuse HBFine's existing bandwidth bins and counters, without modifying the HBFine implementation.

---

## 11. Bandwidth Re-Runs and Figures

After selecting a winner for a context/configuration, automatically enqueue one bandwidth-enabled replay using the same Tiling and Runtime Parameters.

Required bandwidth replay cases:

```text
3 contexts ×
(
  FullHBM best,
  4F1M Pool best,
  4F1M Double best
)
= 9 runs
```

Treat these as dependent tasks inside the same global Scheduler, not a separate manually launched batch.

Create two primary figures.

### Figure A — Full-HBM

`figures/bw_fullhbm.png`

Use **three panels**:

- Context 10k
- Context 100k
- Context 200k

Each panel shows:

- HBM Read Bandwidth.
- HBM Write Bandwidth.
- HBM Total Bandwidth.
- 3100 GB/s theoretical bandwidth reference.

X-axis: simulated elapsed time.  
Y-axis: GB/s.

### Figure B — 4F1M

`figures/bw_4f1m.png`

Use **three panels**, one for each context.

For each context, overlay or separately show the best Pool and Double results using consistent styling.

Plot:

- HBM Read Bandwidth.
- HBF Demand Read Bandwidth.
- HBF Prefetch Fill Bandwidth.
- HBF NAND Sense Bandwidth.
- SRAM Resident Read Bandwidth.

Use correctly labeled bandwidth domains. These values must not be added together as if they represented independent physical traffic.

Mark the 620 GB/s HBM and 3072 GB/s HBF reference bandwidths.

Use consistent bin widths, colors and units.

Also save the sampled values as CSV.

---

## 12. Acceptance Checks Before the Full Sweep

Do these checks using the fixed T0 candidate. Do not require the Agent to design additional verification methods.

### SRAM

- Pool: 32 MiB/Stack available.
- Double: 16 MiB/Stack per Slot.
- Per-Plane capacity enforced.
- Base Die Spill enforced.
- No overlapping Epoch ownership.

### Partitioning

- QKV correctly partitions when it exceeds the Double Slot.
- Every N-slab fits its physical SRAM footprint.
- Partial GEMM covers the complete output.
- Parent Tensor Strides remain unchanged.
- No overlapping or missing output columns.

### Tiling

- T0 retains the original Native IR semantics.
- Larger BN/BK changes actual TMA and MMA schedules.
- Illegal candidates are rejected rather than silently downgraded.
- TMA offsets and masks remain legal.
- SMEM and registers do not exceed configured limits.

### Prefetch

- Router-dependent Prefetch starts only after Router completion.
- Future eligible Prefetches can overlap current computation.
- FIFO and Head-of-Line behavior remain intact.
- Demand Fallback works.
- SRAM Release/Reclaim completes correctly.
- Prefetch-OFF actually issues no background Prefetch.

### Workload

- Context 10k/100k/200k changes KV-cache work.
- The same model, routing and precision are used for every configuration.
- All Kernels complete.
- No unexpected data or address errors.

If a check fails, fix the specific bug before starting the large Sweep. Do not make an unrelated architectural change.

---

## 13. Output Structure

Write to:

`validation/results/qwen235b-flashaccel-sweep/`

Suggested layout:

```text
qwen235b-flashaccel-sweep/
├── manifest.csv
├── summary.csv
├── best_configs.json
├── report.md
├── ctx10k/
│   ├── fullhbm/
│   ├── pool/
│   └── double/
├── ctx100k/
│   ├── fullhbm/
│   ├── pool/
│   └── double/
├── ctx200k/
│   ├── fullhbm/
│   ├── pool/
│   └── double/
├── controls/
└── figures/
    ├── bw_fullhbm.png
    ├── bw_4f1m.png
    ├── bw_fullhbm.csv
    ├── bw_4f1m_pool.csv
    └── bw_4f1m_double.csv
```

The CSV bandwidth files must include a `context_length` column so that all three contexts remain distinguishable.

`best_configs.json` is a machine-generated results artifact; do not confuse it with manually maintained Sweep configuration JSON files.

---

## 14. Required Summary

Produce a table with **one group per Context Length**:

| Context | Configuration | Best BM/BN/BK | Partitions | Cycles | Latency (ms) | Speedup vs HBM |
|---|---|---|---:|---:|---:|---:|
| 10k | FullHBM | | | | | 1.00× |
| 10k | 4F1M Pool | | | | | |
| 10k | 4F1M Double | | | | | |
| 100k | FullHBM | | | | | 1.00× |
| 100k | 4F1M Pool | | | | | |
| 100k | 4F1M Double | | | | | |
| 200k | FullHBM | | | | | 1.00× |
| 200k | 4F1M Pool | | | | | |
| 200k | 4F1M Double | | | | | |

Also separately report the six Prefetch-OFF comparisons.

Answer these questions using measured data:

1. Does the best GEMM Tiling change with Context Length?
2. Does larger BK reduce GEMM overhead or increase resource pressure?
3. How many N-slabs are needed under Pool and Double?
4. Does Double Buffering improve Prefetch/Compute overlap?
5. Does the 4F1M advantage shrink as Attention KV traffic grows?
6. Are the 10k, 100k and 200k cases bandwidth-, compute- or NAND-latency-limited?
7. Does HBF Prefetch benefit change with Context Length?
8. What are the HBM/HBF measured peak and average bandwidths?
9. How many runs succeeded, timed out or were infeasible?
10. Which results are measured single-block results versus analytical estimates?

Do not fill missing measurements with inferred values.

---

## 15. Change Scope

Permitted modifications:

- `frontend/src/compile/opt/partition.py`
- `frontend/src/compile/opt/prefetch.py`
- New `frontend/src/compile/opt/gemm_tiling.py`
- `frontend/tools/run_qwen_simulations.py`
- Relevant frontend tests
- `include/memory/naive.h` for bandwidth sampling
- New stable 4F1M/BW YAML configurations
- `scripts/experiments/sweep_qwen235b_flashaccel.py`
- Reporting/plotting scripts

Do not modify:

- HBFine
- FlashLab/EventLab
- Native Backend Patch chain
- GPU Core, SM, L1/L2 or NoC
- HBF hardware dispatch bandwidth
- SRAM architecture beyond the approved allocation-policy behavior

Prefer existing infrastructure and keep implementation generic. Use C++ `assert` for internal invariants and exceptions only for true external input/API errors.

### Execution Order

1. Implement SRAM Policy support.
2. Implement structured GEMM Tiling Pass.
3. Correct Prefetch Scheduling.
4. Add CLI overrides and Full-HBM compiler path handling.
5. Correct HBM bandwidth sampling.
6. Add the Python Sweep Launcher.
7. Run fixed T0 correctness checks and one resource pilot.
8. Build Release once.
9. Submit **all legal Sweep Candidates into one concurrent queue**.
10. Dynamically enqueue Prefetch-OFF Controls and bandwidth replays as dependencies complete.
11. Generate figures and final report.
12. Commit changes.

**Do not stop after implementing the launcher. Execute the simulations and produce actual results.**

If the current execution session cannot finish the full matrix, retain the resumable manifest and all completed results; report the exact remaining cases without claiming the sweep completed.

The final deliverable is a reproducible, concurrent, resource-bounded evaluation of FlashAccel Pool versus Strict Double Buffering versus Full-HBM across 10k/100k/200k context lengths.
