# Single-query FA3 reference workload

The FA3 builder accepts `sequence_q=1`, `--head-dim 128|256` (default 128),
and `--dtype fp16|bf16` (default fp16). Existing full-tile defaults are preserved.
A single live query uses one m64 consumer group per query head: global Q/O and
split scratch accesses touch only the live row, but Tensor Core and softmax work
retain the padded 64-row tile. This is a deliberately underutilized FA3 baseline,
not a reproduction of FlashAttention decode's grouped-head packing or A100 kernel.
K/V are shared by query heads through the existing GQA tensor mapping.
Head dimension changes global traffic, QK/PV instruction counts, output work,
and shared/register footprints. Split-KV normalization reads/stores live rows.

Example: Qwen3.5-9B full attention, batch 1, 16 Q heads, 4 KV heads,
head dimension 256, BF16, one query and 32768 cached positions:

```sh
./build/gtsim_fa3 configs/v0.1.0/h100-sxm.yaml \
  --memory-config configs/v0.1.0/memory_full_hbm-0F5M.yaml \
  1 16 4 1 32768 --head-dim 256 --dtype bf16 --kv-tile 64 --split 8
```

Here 32768 includes all visible KV positions for this attention invocation.
The 64-token KV tile fits the 256-dimensional CTA's 160 KiB shared memory;
the default 128-token KV tile exceeds the profile's shared-memory capacity.
Split=8 is an explicit modeling choice, not recovered from the trace.
The FullHBM topology routes all tensors to HBM, matching the trace's GPU KV
cache residency. CPU MLP weight offload is outside this attention workload.

Model configuration source: https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json
This model has eight full-attention layers among 32 decoder layers.
The simulator executes timing graphs rather than checkpoint tensor values.
Its H100 hardware profile does not reproduce the trace's A100 hardware.
