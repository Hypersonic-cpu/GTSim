# Qwen3-30B-A3B single-step decode in GTSim

Correction status: the reviewed `f554d79` implementation still has a parallel decode compiler and unresolved attention, RoPE, and vector-operation fidelity problems. The residual dataflow has been corrected, but the canonical IR migration and corrected six-case execution have not yet been completed. Previous decode timing reports are historical results of the reviewed implementation, not acceptance evidence for the corrected workload.

This workload models one autoregressive decode token for each request, through one or two Qwen3 MoE Transformer layers on a single H100 stream. It does not model prefill, sampling, the LM head, or later generation steps.

The dimensions and operator behavior follow the [Qwen3-30B-A3B configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json) and the [Transformers Qwen3 MoE implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py): hidden size 2048, 32 query heads, 4 KV heads, head dimension 128, 128 experts, Top-8 routing, expert intermediate size 768, BF16, RMSNorm epsilon `1e-6`, and RoPE theta `1e6`.

## Build and run

From the repository root:

```sh
cmake --build build --parallel 8
PYTHONPATH=frontend/src python3 -m gtsim_frontend.cli decode \
  --batch-size 1 --layers 1 --context-lengths 10000 \
  --runner build/gtsim_program \
  --architecture configs/v0.1.0/h100-sxm.yaml \
  --memory-config configs/v0.1.0/memory_full_hbm-0F5M.yaml \
  --out-dir validation/results/qwen3-30b-a3b-decode/b1-l1
```

Use `--batch-size 8` or `16`, and `--layers 2` for the other configurations. Supply exactly one context length per request; each must be from 10000 through 32000. Defaults cycle through `[10000, 16384, 24576, 32000]`. The attention length is always the request context plus its appended token. `--split-count` overrides the attention split count; default dispatch uses four splits for B=1 and one non-split CTA per query head for B=8/16.

By default, B=1 uses direct execution for its eight selected experts. B=8/16 run an explicit histogram/prefix/gather assignment kernel followed by grouped Expert GEMM1 and GEMM2. The report includes each active expert's token count, physical M=16 rows, padding and row utilization; no balanced token distribution is assumed.

## Routing input

Without a trace, a seeded deterministic Top-8 assignment is generated (`--routing-seed`, default 0). The Router projection and softmax/TopK kernels remain present in the timing graph. GTSim is a timing simulator and does not numerically infer activations or logits; the trace supplies the route values consumed by later kernels.

A trace is JSON with one array per layer and one record per request:

```json
{
  "layers": [
    [
      {
        "expert_ids": [0, 2, 4, 6, 8, 10, 12, 14],
        "probabilities": [
          0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125
        ]
      }
    ]
  ]
}
```

For B=8/16, repeat the record to exactly B rows. For two layers, provide two layer arrays. The loader normalizes each selected probability row and rejects malformed routes, duplicate experts in a token's Top-8, and IDs outside `[0,127]`.

## IR and execution artifacts

The top-level `.progir` contains the ordered single-stream layer program. Each launch writes its own `.algoir`, structured `.flowir`, structured Hopper `.nvir`, and executable `.gtsir`; `program.gtsir` contains the executable stream. `kernel-timings.json` records simulator-derived launch, device-complete and memory-drain-complete cycles for every kernel, per-layer spans, actual HBM/HBF traffic, graph counts, and the selected attention/MoE variants. `simulation.log`, `results.md`, `metrics.json`, and `run-spec.json` accompany each run.

The simulator keeps one GPU, memory system and L2 across launches, including the two layers. A kernel's completion includes the configured conservative memory-drain boundary; the JSON keeps device completion and drain completion separate. Launch overhead is configurable with `--launch-overhead-cycles` and defaults to zero.

## Fidelity limits

- KV histories use separate contiguous request/head allocations with exact per-request lengths and base addresses. Paged KV addressing is not modeled, and there is no free KV gather.
- Expert weights are allocated only for experts selected by the supplied routing trace. Assignment, GEMM and reduction work is represented in timed kernels.
- Decode attention uses one request/head CTA (or one per request/head/split), streams valid KV tiles, and uses online softmax. Split-KV partial outputs carry log-sum-exp statistics into a separately timed stable reduction.
- The decode and grouped-expert patterns use the repository's H100 Hopper simulator profile. These cycle estimates are simulation results; H100 serving-kernel timing calibration or numerical tensor-inference parity has not been established.
