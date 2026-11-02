# One Qwen3-30B-A3B MoE decode block

`one_block.progir` is the actual textual workload. `b1.json` and `b16.json` supply
all dimensions, context lengths, routing traces, hardware SM count and split count.
Batch size and routing are not hardcoded in the compiler. The default sample
inputs use one new token per request, 10K–32K heterogeneous histories, BF16 and
one ordered stream. There is no prefill, LM head, sampling or extra decode step.

The model values follow the [official Qwen configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)
and the [Transformers Qwen3 MoE implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py).
The shared text library contains the algorithms and schedules; the Program only
declares tensors, parameters, invocations and static variant branches.

The attention residual is materialized once, consumed by post-attention RMSNorm,
and used again by the final MoE residual addition. Input Q/K head normalization
precedes half-layout RoPE; V is only extracted. Cache storage is packed contiguous
request/head regions with separate valid lengths, including the new token.
Expert weights are packed by expert ID, but only selected expert regions are read.
Top-K routing work is still executed before consuming the supplied routing trace.

B=1 selects direct experts. B>1 selects histogram/scan, stable assignment gather
and grouped Small-M execution. `S` independently selects split attention and an
explicit LSE-combination kernel. The sample split counts are configuration values,
not a measured optimal dispatch policy.

## In-memory comparison

```sh
PYTHONPATH=frontend/src python3 frontend/tools/compare_qwen_in_memory.py \
  --source frontend/asm/qwen3/one_block.progir --lib frontend/lib \
  --params frontend/asm/qwen3/b1.json \
  --report validation/results/text-qwen/memory-b1.json

PYTHONPATH=frontend/src python3 frontend/tools/compare_qwen_in_memory.py \
  --source frontend/asm/qwen3/one_block.progir --lib frontend/lib \
  --params frontend/asm/qwen3/b16.json \
  --report validation/results/text-qwen/memory-b16.json
```

The tool expands and checks one new DAG, retains compact comparison data, releases
the DAG, and independently expands the old generator's DAG. No large IR/DAG files
are serialized. Exact rectangle-coverage counters compare memory regions despite
packed versus separate allocations and different copy packetization. Native and
dependency fingerprints are reported separately; equal memory does not establish
whole-graph equivalence. Legacy imports exist only in this comparison tool.

Normal in-memory compilation without the legacy oracle:

```sh
PYTHONPATH=frontend/src python3 -m compile.run_text \
  frontend/asm/qwen3/one_block.progir --lib frontend/lib \
  --params frontend/asm/qwen3/b16.json
```

`--emit-ir --out DIR` optionally emits compact FlowIR/ArchIR. Full DAG serialization
is not part of this default path. Per-kernel materialization preserves the source
Program's complete ordered block and avoids retaining all expanded graphs at once.

The independent CPU operator oracle is `frontend/tools/qwen_semantic_oracle.py`;
`audit_qwen_addresses.py` checks request/head/split/expert association and position
operands directly on structured Native IR. Neither claims numerical Transformer
execution by GTSim. See `results.md` for completed comparisons and differences.

The current corrected schedules and completed reruns are documented in
[`issue-aligned-results.md`](issue-aligned-results.md). `results.md` retains the
initial comparison before the IR-only synchronization/precision correction.

## Four simulator runs

Both DAG versions can execute through the same C++ runner without saving giant
kernel DAG files. A named pipe transfers each materialized kernel directly to
the existing ProgramExecutor JSON reader. The small Program manifest, source
hashes, simulator log and result JSON are retained. Each case runs a full block
on its own persistent GPU state; different cases are independent experiments.

Build the runner if needed:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target gtsim_program --parallel
```

Run each version and batch combination (the example loops run sequentially):

```sh
for batch in 1 16; do
  for version in legacy new; do
    PYTHONPATH=frontend/src python3 frontend/tools/run_qwen_simulations.py \
      --version "$version" --params "frontend/asm/qwen3/b${batch}.json" \
      --runner build/gtsim_program \
      --architecture configs/v0.1.0/h100-sxm.yaml \
      --memory configs/v0.1.0/memory_full_hbm-0F5M.yaml \
      --out "validation/results/text-qwen/four-runs/${version}-b${batch}"
  done
done

python3 frontend/tools/summarize_qwen_simulations.py \
  validation/results/text-qwen/four-runs
```

Use fresh output directories when rerunning. The evidence run used the already
built `/tmp/gtsim-qwen-correction-build/gtsim_program`; each `provenance.json`
records its exact binary hash and all compiler/source/configuration hashes.
`result.json` contains global kernel cycle intervals, device completion versus
memory-drain completion, traffic counters and graph counts. The generated
`comparison.md` includes ordered intervals and per-kernel differences.
See `dag-differences.md` for the structural explanation. Python expansion and
pipe transport wall time are distinct from simulated GPU cycles.
