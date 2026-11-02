# Textual operator library

The normal compiler reads these `.algoir` and `.flowir` definitions directly.
They contain parameter declarations, formulas, scheduled dataflow, and structured
loops. No Python workload generator is invoked. All kernel invocations share the
AST/parser/compiler in `frontend/src/`.

| Files                                              | Contract                                                                                         |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `gemm_reference.*`                                 | Existing fully tiled FP16 Hopper GEMM schedule, copied alongside the other operators.            |
| `small_m_gemm.*`                                   | BF16 projections, physical M=16 tiles and only valid logical input/output rows.                  |
| `grouped_gemm.*`                                   | Expert-specific packed weights and variable per-expert row counts; grouped Small-M MMA baseline. |
| `fa3_reference.*`                                  | Text port of full-query-tile streaming attention, with actual 128-query tiles.                   |
| `fa3_decode*`, `fa3_combine.*`                     | Single-query GQA online softmax; optional split partial outputs and stable LSE combination.      |
| `rmsnorm`, `norm_rope`, `extract`, `kv_append`     | FP32 reductions, learned scales, position/theta operands, Q/K rotation and current-token append. |
| `router_topk`, `expert_histogram`, `expert_gather` | Timed softmax/Top-K/renormalization, histogram/prefix scan and assignment gathering.             |
| `swiglu`, `weighted_*`, `residual_add`             | Separate Gate/Up reads, token/slot-aware weighted expert reduction and residual updates.         |

The matrix schedules currently require N divisible by 64 and K divisible by 16;
the studied Qwen projections satisfy these conditions. Grouped input/output rows
must be contiguous within each expert group. B=1 direct execution has one row per
selected expert; grouped execution uses stable expert-sorted rows. Padding applies
to physical MMA fragments, never additional logical memory accesses.

`fa3_reference` preserves the full-query mathematical algorithm but uses the new
text framework's packetization and vector schedule. It is not graph/cycle parity
or a transfer of the old reference's timing calibration. The original calibrated
C++/legacy full-tile implementation remains unchanged. The Decode variants are
separate schedules and never manufacture 127 fake query tokens.

Decode now matches the legacy precision contract: probability tiles are cast to
BF16 before PV, Q/K/V conversion is implicit in the FP32 arithmetic cost nodes,
and partial outputs/LSE remain FP32. K-buffer reuse waits for the previous QK;
V-buffer reuse waits for the previous PV. Small-M and grouped copies use Issue
ordering between K-stage transfer batches, while retaining Complete waits for
data readiness and consumer completion before buffer reuse. Earlier four-run
results describe the prior, more serialized text schedules. New latency estimates
remain uncalibrated.

`static.args` attaches immediate operands such as position, theta, epsilon and
channel coefficients to scheduled operations. They survive into structured Native
IR and become concrete ExecIR metadata only in the final lowering. The simulator
is a timing engine; arithmetic opcodes can represent a timing-cost class rather
than numerically executing the algorithm formulas.
