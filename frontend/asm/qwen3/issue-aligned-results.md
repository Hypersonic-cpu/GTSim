# IR-only schedule correction and rerun

All four B=1/16 new/legacy single-block Programs executed successfully through
the unchanged C++ ProgramExecutor. The current branch is `serve-sim`; the source
parent commit is recorded in the evidence. No timing constants, runtime code or
Python compiler implementations were changed for this correction.

## Text changes

- `small_m_gemm.flowir` and `grouped_gemm.flowir`: prior K-stage A/B transfer
  dependencies now use Issue, retaining Complete waits for input readiness and
  consumer completion before two-stage buffer reuse.
- `fa3_decode{,_split}.flowir`: the K buffer waits for previous QK completion,
  whereas V still waits for previous PV completion. Online max/sum/output state
  dependencies remain explicit.
- `fa3_decode{,_split}.{algoir,flowir}`: remove standalone Q/K/V conversion cost
  nodes, modeling conversion implicitly in FP32 arithmetic as in the legacy
  schedule; cast each probability tile to BF16 before PV. Split partial outputs
  and LSE remain FP32. No fake query rows, extra requests or padded KV accesses
  were added. The separate stable LSE combination kernel remains present.

## Actual simulator results

| Batch | Legacy cycles | Initial text cycles | Corrected text cycles | Corrected vs legacy |
| ----: | ------------: | ------------------: | --------------------: | ------------------: |
|     1 |       446,472 |             809,360 |               446,631 |     +159 / +0.0356% |
|    16 |     3,078,542 |           5,482,006 |             3,078,394 |     -148 / -0.0048% |

Each case uses one ordered stream and persistent GPU/MemorySystem/L2, stable
allocations and zero launch overhead. The reported interval includes required
memory drain; device and drain completion happen to coincide in these runs.

All five GEMM kernels in each batch now have identical normalized native
instructions, CTA resources and Issue/Complete edge sets to the legacy kernels.
Their individual simulated durations are identical. Decode Attention durations
also match exactly: 50,671 cycles for B=1 Split-KV and 1,652,512 cycles for B=16
Non-Split. B=1 Combine remains 1,814 cycles in both versions.

Attention is not claimed to have an identical graph: operation ordering,
cost-class opcode names, redundant dependency edges and register estimates still
differ. Equal measured simulation duration is distinct from full graph parity.

Remaining cycle differences are confined to auxiliary kernels:

| Kernel                   | B=1 text minus legacy | B=16 text minus legacy |
| ------------------------ | --------------------: | ---------------------: |
| Input RMSNorm            |                   +33 |                    +33 |
| Q RMSNorm/RoPE           |                   +51 |                   +198 |
| K RMSNorm/RoPE           |                   +50 |                    +51 |
| Post-attention RMSNorm   |                   +33 |                    +33 |
| Router Softmax/Top-K     |                    -9 |                     -9 |
| Expert histogram/prefix  |                absent |                    +20 |
| Expert assignment gather |                absent |                   -475 |
| Weighted MoE sum         |                    +1 |                     +1 |

Actual traffic is equal per kernel and in total between versions. B=1 reads
134,458,752 HBM bytes and writes 191,488; B=16 reads 1,473,946,112 and writes
2,530,944. HBF traffic is zero. All 37 kernel comparisons retain equal logical
read/write regions and requested bytes. New/legacy node counts are 588,050 /
588,165 at B=1 and 4,561,895 / 4,567,304 at B=16.

## Evidence and reproduction

- [All per-kernel durations and global intervals](evidence/issue-aligned-comparison.md).
- [Machine-readable timing and traffic](evidence/issue-aligned-comparison.json).
- [Structural comparison](evidence/issue-aligned-structural.json).
- [Binary, configuration, source hashes and graph counts](evidence/issue-aligned-provenance.json).

The full simulator logs and small manifests remain under
`validation/results/text-qwen/issue-aligned-runs/{new,legacy}-b{1,16}`.
The two-GEMM frontend parity regression also passes; Ruff checks and formatting
checks pass. No full CTest or original 8+6 timing-validation sweep was performed
for this focused IR-only change.

Use the four-run commands in `README.md`, replacing the output root
`validation/results/text-qwen/four-runs` with a fresh directory such as
`validation/results/text-qwen/issue-aligned-runs`, then invoke
`summarize_qwen_simulations.py` on that directory. Kernel DAGs are materialized
one at a time in RAM and streamed through FIFOs, not saved as large files.

These are simulator-derived estimates, not newly calibrated H100 measurements.
No numerical Transformer execution or BF16 accuracy result is claimed.
