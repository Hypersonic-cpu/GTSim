# Qwen text-IR comparison results

This is the initial, frozen comparison before the IR-only Issue/Decode correction.
See `issue-aligned-results.md` for the corrected schedules and rerun results.

The current branch's legacy Qwen generator is used only as an independent oracle.
Both cases compile the same `one_block.progir` and the same shared text library;
only the parameter JSON differs. No large DAG files are written during comparison.
Each new graph is summarized in memory and released before its legacy counterpart
is constructed. Native signatures use fingerprints; memory coverage uses exact
integer rectangle-corner counters, not statistical sampling.

| Batch | Active kernels | New DAG nodes | Legacy DAG nodes | Logical read/write regions |
| ----: | -------------: | ------------: | ---------------: | -------------------------- |
|     1 |             18 |       589,458 |          588,165 | all kernels equal          |
|    16 |             19 |     4,604,999 |        4,567,304 | all kernels equal          |

Both cases finish full-block DAG construction and in-memory comparison. All 37
kernel comparisons have equal valid logical memory regions and requested read/
write bytes after normalizing packed versus separate allocations.

## Equality and differences

- All five projection/expert GEMMs per case have equal normalized native opcode,
  physical fragment shape, warp mapping, logical memory-region signatures, and
  CTA resource assignments. Full graph equivalence is **false**: the new Small-M
  schedule uses Complete dependencies between K-stage copy batches; the old
  schedule uses Issue dependencies. Both retain buffer-reuse consumer waits.
- Auxiliary kernels and Decode have different native decompositions and
  dependencies. BF16→FP32 casts are explicit; RMSNorm uses division by width
  rather than multiplication by a loaded reciprocal. Scalar constants can be
  instruction immediates instead of separate `simd.fill` operations.
- Decode retains FP32 softmax probabilities for its SIMT PV reduction; legacy
  Decode casts probabilities to BF16. Both model scaled QK reduction, online
  max/sum/output recurrence, GQA and the current-token-inclusive valid lengths.
  Split output remains FP32 and the separate combine kernel uses LSE weights.
- RoPE uses the half-channel layout. New structured static operands retain each
  request's actual position, theta, epsilon and channel coefficients through
  Native IR into ExecIR. Legacy cost graphs do not retain all such operands.
- The new KV and expert-weight allocations are packed. Only selected expert
  regions and each request/head's valid cache region are accessed. Equal logical
  byte coverage does not imply identical L2 set mapping or HBM transactions.
- Full-query `fa3_reference` is a mathematical/schedule text port, not evidence
  of old calibrated full-tile graph/cycle parity. It is separate from Tq=1 Decode.

## Semantic checks

The independent CPU oracle checks online versus dense softmax attention and
Split-KV LSE combination at D=128 with a 513-token tail case. Maximum absolute
errors are approximately 1.4e-17 in this double-precision algebraic oracle.
RMSNorm, half-layout rotation and weighted grouped/direct row mapping pass.
This is not a numerical BF16 Transformer execution by GTSim.

Structured Native address audits check all 128/512 request-head-split scheduling
units, all 36/576 Q/K RoPE position operands, and all 448/4,256 expert tile units
for B=1/16. Expert/tensor association is checked per CTA rather than relying only
on aggregate memory-byte equality. Both residual consumers use the same attention
residual tensor. Gate/Up load subregions are distinct.

B=1 has 8 assignments and 8 active experts: 128 physical rows and 120 padding rows
(6.25% row utilization). B=16 has 128 assignments and 76 active experts: 1,216
physical rows and 1,088 padding rows (10.53% utilization). This is a grouped
Small-M MMA baseline, not a calibrated claim of faster Hopper Grouped GEMM.

## Execution and timing scope

All four new/legacy B=1/16 Programs completed through the same existing C++
ProgramExecutor. Each run retained persistent GPU/MemorySystem/L2 and stable
allocations, with one ordered stream, zero launch overhead and no kernel overlap.
Kernel JSON was transferred through named pipes; no large DAG files were saved.
The first status check was made at least 15 minutes after launch as requested;
all four jobs had completed by that check.

| Batch | Legacy stream cycles | Text stream cycles |  Change | Legacy us |   Text us |
| ----: | -------------------: | -----------------: | ------: | --------: | --------: |
|     1 |              446,472 |            809,360 | +81.28% |   225.491 |   408.768 |
|    16 |            3,078,542 |          5,482,006 | +78.07% | 1,554.819 | 2,768.690 |

Microseconds use the configured 1,980 MHz core clock. All cycle intervals include
required memory drain; device-completion and drain-completion timestamps happen
to coincide for every kernel in these four runs. Each run starts at global cycle
zero and ends at the stream cycles above.

Actual HBM traffic is identical between versions: B=1 reads 134,458,752 bytes and
writes 191,488 bytes; B=16 reads 1,473,946,112 bytes and writes 2,530,944 bytes.
HBF traffic is zero with the selected FullHBM profile. Thus this measured
simulation slowdown is not explained by extra HBM bytes in these cases.

At B=1 the five GEMMs contribute 325,676 of the additional 362,888 cycles
(89.75%); Decode contributes 37,053 (10.21%). At B=16 Decode contributes
1,183,619 of the additional 2,403,464 cycles (49.25%), and the five GEMMs
contribute 1,219,993 cycles (50.76%). Both matrix
dependency ordering and Decode native decomposition differ; these are observed
combined effects, not isolated causal measurements. Auxiliary differences are
small: gather saves 475 cycles at B=16, while split combine has identical B=1
duration despite its different graph.

Detailed structural differences are in `dag-differences.md`. Complete ordered
kernel intervals, duration deltas and per-kernel HBM reads are in
`validation/results/text-qwen/four-runs/comparison.md` and `comparison.json`.
Each case subdirectory retains `result.json`, `simulation.log`, `program.gtsir`
and `provenance.json` with binary/configuration/source hashes. New timings remain
uncalibrated, and no reference MAPE/accuracy figures are transferred to them.

No complete CTest or 8+6 reference validation sweep was run for this task.
The existing C++ timing constants and legacy implementations were not modified.
See `README.md` for exact in-memory reproduction commands. Compact machine-readable
reports are under `validation/results/text-qwen/memory-b{1,16}.json`; CPU and address
checks are `semantic-oracle.json` and `address-audit.json` in the same directory.

## Compiler CPU cost

Removing serialization exposed the main remaining cost: per-instruction symbolic
resolution and repeated scans of prior fragment events. Iteration scopes now share
validated parameter contexts, resolve CTA/K invariants once, and select indexed
last-executor events directly. A 103,120-node QKV expansion dropped from 5.45 to
2.85 seconds in a local spot measurement, with unchanged native/dependency hashes.
This is a compiler measurement, not a simulator latency improvement or a general
performance guarantee. The full comparison reports were produced before this
CPU-only optimization; their numerical graph results remain applicable.
