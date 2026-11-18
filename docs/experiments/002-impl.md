# Implementation Plan — Experiment 002: Tiling, Ordering & Prefetch

## 0. Task and execution rules

Implement and execute the experiment defined in:

`docs/experiments/002-tiling-order-Qmat.md`

Repository: `Hypersonic-cpu/GTSim`
Target branch: `sim-validation`

**Goal:** Build a FlowIR-based GEMM exploration experiment for the isolated Q projection of Qwen3-235B-A22B, then measure Full-HBM reference performance and sweep 4F1M candidates.

This is an **implementation and experimental execution task**, not another proposal or literature review.

Rules:

- Read the complete Spec and inspect the current code before modifying anything.
- Implement the required functionality, tests, simulator integration, experiment scheduler and reports.
- Run the experiment after implementation and correctness verification.
- Do not introduce a constraint solver, genetic algorithm or learned optimizer.
- Do not introduce a hand-designed FlashAccel N-slab policy.
- Search decisions must originate at the FlowIR level.
- Keep the experiment isolated from the original full-Qwen experiments.
- Do not change GPU clock rates, physical memory parameters, NAND timings, hardware bandwidths or other baseline simulation constants.
- Do not silently skip broken functionality, substitute an easier configuration, or claim completion based only on successful compilation.
- Preserve existing working interfaces and regression tests.
- Document only essential implementation deviations and findings. Avoid extensive progress Markdown files.

**Experiment output root:**

`validation/results/002-tiling-order-Qmat/`

Use the exact directory name above.

---

## 1. Implement a reusable sweep scheduler first

Extract and improve the scheduler in:

`scripts/experiments/sweep_qwen235b_flashaccel.py`

Suggested reusable location:

`scripts/experiments/common/sweep_scheduler.py`

Experiment-specific configuration, candidate generation, command construction and result parsing must remain outside the generic scheduler.

### 1.1 General scheduler requirements

The reusable scheduler must support:

- An arbitrary collection of independently executable jobs.
- Asynchronous process launch with bounded concurrency.
- Resource-aware admission control.
- Pilot-based memory estimation.
- Automatic dispatch whenever a running job finishes.
- Process-group cleanup, timeout and error handling.
- Stable case fingerprints, resumption and deduplication.
- Per-case logs and machine-readable status.
- Periodic aggregated progress and event logs.
- Dynamic reduction of concurrency after memory pressure.
- No busy polling of process completion.

Implement it using `asyncio` and event-driven process waiting.

Do not start a new OS thread per candidate.

Retain compatibility with existing experiment scripts where practical. Do not break the older Qwen235B FlashAccel sweep.

### 1.2 CPU resource policy

Available CPU budget:

```text
CPU_BUDGET = 180
```

The scheduler must also inspect actual CPU affinity and cgroup CPU quota.

```text
EffectiveCPUBudget =
    min(180, affinity_cpu_count, cgroup_cpu_quota)
```

Each running simulation consumes its measured or configured CPU slot allocation. Do not launch more simultaneous CPU-demanding jobs than the effective budget.

Default to one CPU slot per simulation unless pilot measurements demonstrate sustained higher parallel CPU usage.

No fixed batch barriers: whenever resources become available, dispatch another queued case.

### 1.3 RAM policy — mandatory

**Hard memory threshold: 95% of total available RAM capacity.**

Do not interpret this as 95% of currently free memory.

Determine effective RAM capacity using physical host memory and applicable cgroup memory limits. Monitor both host-wide memory pressure and cgroup memory usage; whichever constraint is tighter governs admission.

Use:

```text
RAM_HARD_LIMIT = 0.95
RAM_ADMISSION_TARGET = 0.85
RAM_RECOVERY_TARGET = 0.90

PILOT_RAM_MULTIPLIER = 2.0
```

After the pilot:

```text
EstimatedRAMPerJob = 2 × PilotPeakRAM
```

This is the initial per-job memory reservation used for admission control.

The pilot peak must represent the maximum simultaneous memory footprint of the complete simulation process tree, not simply `max(PythonPeakRSS, CppPeakRSS)`.

Collect process-tree RSS while the pilot runs, including child processes. Cross-check peak values against `ru_maxrss` and cgroup statistics where available.

Do not add peaks from independent, non-overlapping process phases as though they occurred simultaneously.

Before launching a job, ensure the conservative total-memory projection remains below the admission target. Account for existing non-experiment memory usage, currently running jobs and their reserved peak budgets without double-counting their current resident memory.

If observed per-case peak usage exceeds the pilot-derived reservation, raise the future admission estimate accordingly. Do not lower the initial 2× estimate based solely on one small observed case.

**Emergency behavior at 95%:**

1. Immediately suspend new job admissions.
2. Terminate the newest running experiment jobs by process group, escalating SIGTERM to SIGKILL when necessary.
3. Continue until memory consumption falls below 90%.
4. Mark interrupted cases `DEFERRED_RAM` and preserve logs.
5. Reduce the effective concurrency before retrying.
6. Never terminate unrelated system processes.

Memory safety monitoring must run internally at approximately 1-second intervals, independent of the agent's slower progress polling.

Where supported and permitted, use a dedicated cgroup v2 subtree to enforce an actual kernel memory ceiling for experiment processes. If kernel-level enforcement is unavailable, use conservative admission and emergency monitoring, but explicitly report that the 95% limit is monitored rather than strictly kernel-enforced.

Do not disable the RAM guard to increase throughput.

### 1.4 Timeout and recovery

Default per-case wall-clock timeout: **1800 seconds**.

Pilot timeout: **1800 seconds**.

Requirements:

- Terminate the complete process group on timeout.
- Preserve stdout, stderr and simulation logs.
- Distinguish `SUCCESS`, `INFEASIBLE`, `FAILED`, `TIMEOUT`, `DEFERRED_RAM` and `OOM`.
- Do not classify arbitrary execution errors as illegal configurations.
- Retry memory-deferred cases only after lowering concurrency; maximum three automatic RAM retries per case.
- Resume successful cases only when their full fingerprints match.
- Abort the sweep if the resource scheduler itself fails or memory safety cannot be maintained.

### 1.5 Scheduler tests

Before any simulation, test:

- CPU admission and RAM admission.
- Two jobs completing in different orders.
- Resume/deduplication behavior.
- Process timeout and process-group cleanup.
- A simulated 95% memory-pressure event.
- Requeue after memory recovery.
- No deadlock when concurrency is reduced while jobs are pending.
- Accurate peak process-tree memory accounting.

Use synthetic short-lived child processes. Do not test emergency termination by deliberately exhausting the production machine's memory.

---

## 2. Isolate the Q-projection workload

Create an independent ProgIR in:

`frontend/asm/qwen3/experiments/q_projection.progir`

Only one GEMM is permitted:

```text
Q = X_norm @ W_q
```

Fixed properties:

```text
M = 1
N = 8192
K = 4096
NUM_SMS = 132
DTYPE = BF16
```

Memory placement:

| Tensor                  | Full-HBM | 4F1M |
| ----------------------- | -------- | ---- |
| X_norm                  | HBM      | HBM  |
| W_q                     | HBM      | HBF  |
| Q                       | HBM      | HBM  |
| Split-K partial results | HBM      | HBM  |

`W_q` must be physically contiguous and row-major, with shape `[4096,8192]`.

Do not simulate the full QKV projection, RMSNorm or any other transformer operator.

Keep the original Qwen source files unchanged.

---

## 3. Implement the WS persistent GEMM FlowIR

Use the current C++ H100 GEMM and `gemm_reference.flowir` as the behavioral references.

The generated GEMM must have:

```text
BM = 64
WarpsPerCTA = 12
ProducerWarps = 8..11
ConsumerWarps = 0..7
PipelineDepth = 2
MatrixInstruction = WGMMA m64n64k16
```

Implement this as a parameterized FlowIR schedule, not 276 separately generated source files.

### 3.1 Persistent CTA

Use:

```text
numM = 1
numN = N / BN
numTasks = numN * SplitK
numCTA = min(NUM_SMS, numTasks)
numWaves = ceildiv(numTasks, numCTA)
```

Each physical CTA repeatedly executes tasks selected by:

```text
rank = wave * numCTA + cta_id
```

Reject or guard ranks beyond `numTasks`.

Only the first logical output row is valid. Physical BM padding must never result in out-of-range memory access or stores.

### 3.2 Parameterized execution

The schedule must support:

```text
BN ∈ {128,256}
BK ∈ {32,64,128}
SplitK ∈ {1,2,4,8}
```

The same FlowIR template must correctly handle all combinations.

Implement proper Split-K partial outputs and a dependent reduction when `SplitK > 1`. Include reduction latency and HBM traffic in measured execution time.

For `SplitK=1`, avoid allocating or executing unnecessary partial-result buffers and reduction operations.

### 3.3 Conditional correctness

Explicitly validate all conditional paths:

- First, middle and last K iterations.
- Odd and even K iteration counts.
- First and subsequent persistent waves.
- SMEM ping-pong reuse across successive tasks.
- WGMMA accumulation dependencies.
- CTA task tails.
- Split-K output and reduction dependencies.
- No producer–consumer race or premature buffer release.

Do not reuse the old small-M FlowIR's overlapping producer/consumer warp mapping.

Lowered Native IR must visibly reflect the selected tiling, ordering and Split-K geometry.

---

## 4. Implement CompOrdering

Represent the logical task space as:

```text
Task = (split_id, n_tile)
```

Implement the following task permutations:

```text
n_major
split_major
n_grouped_2
n_grouped_4
n_grouped_8
staggered
```

Use the exact definitions in the Spec.

After generating the permutation, map tasks to persistent CTAs using:

```text
rank = wave * numCTA + cta_id
task = Permutation(rank)
```

The ordering must affect actual generated instruction addresses and execution dependencies, not only metadata.

For `SplitK=1`, all orderings are equivalent; deduplicate them.

For `SplitK>1`, all six ordering candidates are required.

Test every permutation as a bijection over the logical task space.

---

## 5. Implement HBF Prefetch Tiling and Lookahead

The prefetch dimensions are independent of compute tiling:

```text
PfBN
PfBK
PfAheadDist
```

Implement physical-page-aware Prefetch Region generation, translating each logical `[PfBK,PfBN]` weight rectangle into its actual mapped NAND pages.

Requirements:

- Handle regions spanning multiple compute tiles.
- Handle a compute tile consuming multiple prefetch regions.
- Track shared physical pages correctly.
- Check the real per-plane and base-die SRAM capacity.
- Support actual SRAM admission, occupancy, reuse and release.
- Never assume an SRAM capacity constraint is equivalent to comparing tensor byte sizes.

`PfAheadDist=0` means **no explicit prefetch**, not prefetch of the current region.

For positive lookahead, implement the Spec's launch-time initialization and runtime first-consumer-triggered enqueueing.

Because the GEMM uses one persistent kernel, prefetch enqueue, readiness and release cannot depend exclusively on kernel launch boundaries.

Any necessary runtime-event representation must be added consistently across FlowIR metadata, lowering, the execution protocol and the GTSim consumer.

**Do not model a prefetch as ready at enqueue time.**

Demand fallback, SRAM allocation stalls, NAND reads and buffer release must all incur their modeled costs.

---

## 6. Compile and correctness validation

Do not run the performance sweep until correctness is established.

### 6.1 Static validation

For every candidate:

- Verify legal BM/BN/BK/Split-K values.
- Verify CTA task coverage.
- Validate SMEM and register capacity.
- Validate TMA box shapes and alignment.
- Validate reduction output buffers.
- Validate Prefetch Region bounds and physical page footprint.
- Ensure all conditional branches lower correctly.

Reject invalid candidates during preflight and record the reason.

### 6.2 Execution correctness

Create deterministic synthetic inputs and compare GEMM outputs to a high-precision numerical oracle, with explicitly documented BF16 tolerances.

Check Split-K=1/2/4/8, all ordering families, and Prefetch ON/OFF.

If GTSim runs only timing rather than numerical values, use the appropriate separate execution/structural oracle to validate complete output coverage and reduction behavior; do not claim numerical testing from cycle counts.

Run the relevant existing GEMM, FlowIR, HBF SRAM and Prefetch regression tests.

A parameter change must not silently alter the mathematical GEMM.

---

## 7. Limit the initial sweep to 276 candidates

The complete Cartesian product in the Spec is unnecessarily large for the first findings experiment.

Use the following fixed, reproducible two-phase enumeration.

### Phase A — Compute ordering and baseline prefetch

Compute candidates:

```text
BM      = 64
BN      ∈ {128,256}
BK      ∈ {32,64,128}
SplitK  ∈ {1,2,4,8}
```

For `SplitK=1`: use only `n_major`.

For `SplitK ∈ {2,4,8}`: use all six ordering candidates.

Number of unique compute configurations:

```text
2 BN × 3 BK × (1 + 3×6 ordering/split combinations)
= 114
```

For each compute configuration, evaluate exactly two Prefetch modes:

**A0 — Prefetch OFF**

```text
PfAheadDist = 0
```

`PfBN/PfBK` are irrelevant and must not generate duplicate cases.

**A1 — Moderate Prefetch**

```text
PfBN = 1024
PfBK = 1024
PfAheadDist = 2
```

Phase A total:

```text
114 × 2 = 228 candidates
```

Run compile feasibility checks before GPU simulation and exclude illegal candidates with explicit records.

### Phase B — Prefetch granularity/lookahead exploration

Select **eight compute configurations** from successful Phase A results:

- For each `SplitK ∈ {1,2,4,8}`, select two distinct compute configurations.
- Rank by the minimum measured execution time of the successful A0/A1 cases.
- Prefer distinct ordering families when possible; break ties lexicographically by `(BN,BK,CompOrdering)`.
- If a Split-K group has fewer than two valid configurations, use all available ones and record the shortfall.

For each selected compute configuration, evaluate:

```text
PfGeometry:
  small  = (PfBN=256,  PfBK=128)
  medium = (PfBN=1024, PfBK=1024)
  whole  = (PfBN=8192, PfBK=4096)

PfAheadDist:
  {1,2,4}
```

Deduplicate equivalent execution configurations:

- For `whole`, there is only one Prefetch Region, so positive lookahead distances are equivalent; run only `PfAheadDist=1`.
- `medium, PfAheadDist=2` was already measured in Phase A; reuse it.

Consequently, each selected compute configuration adds:

```text
small:  3
medium: 2 new
whole:  1
----------------
total:  6 new cases
```

Phase B maximum:

```text
8 × 6 = 48 candidates
```

### Total budget

| Group                                | Maximum simulations |
| ------------------------------------ | ------------------: |
| Independent Full-HBM pilot/reference |                   1 |
| Phase A, 4F1M                        |                 228 |
| Phase B, 4F1M                        |                  48 |
| **Total**                            |             **277** |

There are **276 maximum unique 4F1M sweep cases**. Illegal candidates and deduplicated cases are recorded but not simulated.

Do not expand the sweep to the full Cartesian product without explicit authorization.

---

## 8. Full-HBM pilot and reference

The pilot is a standalone experiment, not a member of Phase A or Phase B.

Use the Full-HBM configuration and the C++-reference-equivalent GEMM:

```text
Hardware    = Full-HBM, 0F5M
BM          = 64
BN          = 256
BK          = 64
SplitK      = 1
CompOrdering = n_major
PfAheadDist = 0
PipelineDepth = 2
WS          = reference 12-warp configuration
```

Run the pilot alone, without any concurrent sweep jobs.

It must:

1. Complete the full compilation and simulation pipeline.
2. Validate the generated kernel and result.
3. Measure wall-clock runtime.
4. Measure maximum simultaneous RAM usage of the complete process tree.
5. Measure approximate CPU demand.
6. Record total cycles, simulation parameters, logs and provenance.

Use this successful result as the **Full-HBM reference**. Do not rerun the identical baseline solely because it also served as the pilot.

Compute the initial sweep memory estimate as:

```text
estimated_job_ram = 2 × pilot_max_simultaneous_ram
```

Determine the effective launch concurrency using both the 180-core budget and the RAM admission target.

### Pilot monitoring

After launching the pilot:

- The execution agent checks pilot status at most once every **5 minutes**.
- Internal RAM safety monitoring remains active at approximately 1-second intervals.
- Process completion is detected by the scheduler without waiting for the next agent poll.
- If the pilot fails, stop and fix the problem before launching the sweep.

Do not launch 4F1M jobs while the pilot is running.

---

## 9. Launch the 4F1M sweep

Before launch:

1. Verify the complete candidate manifest and counts.
2. Verify correctness and preflight results.
3. Verify the reusable scheduler tests passed.
4. Verify available CPU/RAM resources and the pilot-derived concurrency limit.
5. Verify the 95% RAM emergency guard is active.
6. Verify each candidate has an independent output directory and fingerprint.
7. Record a reproducible configuration manifest.

Submit the legal Phase A candidates to one event-driven work queue.

Use cautious initial ramp-up (1, 2, 4, 8 concurrent jobs) before reaching the calculated safe concurrency. The scheduler may increase admissions only while memory observations remain within its reservation model.

After Phase A completes, automatically select the eight configurations according to Section 7, construct the Phase B manifest and run Phase B through the same scheduler.

No manual review or parameter selection is required between phases.

### Sweep monitoring

The agent should inspect overall sweep progress **once every 60 minutes** while the sweep is running.

At each inspection, check:

```text
completed / total
running / queued
successful / failed / infeasible / timeout
current concurrency
peak RAM utilization
current estimated time remaining
last scheduler error
```

Do not frequently query hundreds of child processes manually. The scheduler should maintain an aggregated `scheduler_status.json` that can be read cheaply.

The internal scheduler must continue dispatching jobs and enforcing memory limits between agent inspections.

On completion, immediately proceed to report generation; no extra waiting interval is required.

---

## 10. Required experiment artifacts

Under:

`validation/results/002-tiling-order-Qmat/`

Use the following output structure:

```text
pilot/
  result.json
  resource_usage.json
  driver.log
  simulation.log

phase-a/
  <case-id>/
    result.json
    provenance.json
    compile_summary.json
    driver.log
    simulation.log

phase-b/
  <case-id>/
    ...

candidate_manifest.json
case_results.csv
scheduler_events.jsonl
scheduler_status.json
resource_summary.json
best_candidates.json
findings.md
```

Include the source commit, compiler configuration, effective FlowIR parameters, hardware configuration and candidate fingerprint.

Do not save thousands of identical FlowIR source files. Save specialized IR summaries for all candidates and full generated FlowIR/Native IR for the pilot, best-performing candidates and selected correctness diagnostics.

### Required findings

Report:

**Finding 1 — Compute geometry / Split-K**

- Performance versus BN/BK/Split-K
- Logical compute task count and persistent CTA utilization
- Split-K reduction overhead

**Finding 2 — CompOrdering**

- Performance differences between ordering families
- Changes in HBF traffic, memory parallelism and Prefetch effectiveness
- Cases where ordering produces identical behavior

**Finding 3 — Prefetch tiling**

- Prefetch Region geometry versus SRAM occupancy
- Physical NAND page amplification and repeated reads
- Effect of mismatched compute/prefetch tiling

**Finding 4 — Prefetch lookahead**

- Performance versus `PfAheadDist`
- Late Prefetch, useful residency, admission stalls and fallback reads
- Whether more aggressive Prefetch actually helps

**Finding 5 — Joint optimization**

- Best 4F1M candidate
- Full-HBM reference performance
- 4F1M speedup over its own Prefetch-OFF baseline
- Remaining gap between 4F1M and Full-HBM
- Whether the best configuration differs from independently optimizing the parameters

Include plots or compact tables for these findings.

Do not claim a general optimum outside the tested candidate set.

---

## 11. Final acceptance criteria

This implementation is complete only when:

- The independent Q-projection ProgIR works.
- The WS persistent GEMM FlowIR is correct.
- Split-K and all CompOrdering families produce valid execution graphs.
- HBF Prefetch Tiling and runtime lookahead are functional, not metadata-only.
- Full-HBM pilot/reference completes successfully.
- The reusable scheduler enforces the defined admission and RAM protection policy.
- The legal 4F1M Phase A and Phase B simulations have completed, or failures are accurately identified and explained.
- Results are resumable and reproducible.
- `findings.md` summarizes actual measurements and meaningful bottlenecks.

**Execution instruction:** Implement in the numbered order above. Fix implementation and correctness errors before large-scale simulation. After the pilot succeeds, launch the bounded sweep and continue through final result analysis. Do not stop after writing code or preparing a list of commands.

Provide a concise final report containing the implementation summary, number of successful/failed cases, measured Full-HBM reference, best 4F1M configuration, achieved performance and the main findings. Report any deviations from the Spec explicitly; otherwise avoid additional documentation.

