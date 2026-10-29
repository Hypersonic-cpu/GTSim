# A100 FlashAttention-2 decode workload

`gtsim_fa2` is an experimental, hand-written timing graph for FA2's single-token
inference path. It supports batch 1 and larger batches, BF16, GQA/MQA/MHA,
head dimensions 64/128/256, and up to 128 KV splits. Query length is currently
restricted to one; the Q-to-KV head ratio must be an integer from 1 to 64.
It does not implement general prefill, dropout, backward, or CUDA tensor execution.

```sh
cmake --build build --target gtsim_fa2
./build/gtsim_fa2 configs/v0.1.0/a100-pcie40-simple.yaml \
  1 16 4 1 32768 256 47
```

Arguments after the YAML are `batch heads_q heads_kv seq_q seq_kv head_dim splits`.
The example uses Qwen3.5-9B's full-attention dimensions, not its GDN head dimensions.
The model has 16 Q heads, 4 KV heads and a 256-dimensional full-attention head.
The trace's complete kernel signature confirms a 64x64 tile and four warps;
its main grid `(1,47,4)` specifies 47 splits and four packed KV-head jobs.
Each group of four Q heads becomes four valid rows in a padded 64-row query tile.
The example has 188 main CTAs and four combine CTAs, each with 128 threads.

## Graph construction

The graph uses independent warp `mma.sync.m16n8k16` operations, with four warps
partitioning query rows. Global-to-shared copies use the existing non-TMA
`cp.dram2sram` path with 16-byte/thread accesses. Shared-memory register loads
use 512-byte/warp `ldmatrix.x4` fragments and a swizzled layout. The scoreboard
joins are zero-setup dependency nodes, not four-warp MMA coupling.
There are no WGMMA, TMA, UMMA or TMEM nodes.

Each KV split traverses 64-token blocks backwards, updates row maxima and sums,
rescales an unnormalized FP32 output accumulator, and normalizes at the end.
Completion edges protect shared buffers and register fragments during reuse.
Only valid query rows and KV positions generate global accesses; MMA tiles
retain their physical padding. Empty split partitions produce zero partial
outputs and negative-infinite LSE.

A separate combine graph waits for all main CTAs, computes stable LSE weights,
and accumulates normalized FP32 partial outputs before writing BF16. It models
the source implementation's reusable partial-output register fragment: each
split's load follows consumption of the previous split, rather than keeping
all partial outputs simultaneously live. Final LSE is also stored.

## Hardware and validation boundaries

The Simple Model profile describes A100 PCIe 40GB: 108 SMs, fixed 1410 MHz,
40 GiB HBM at a peak 1555 GB/s, 40 MiB L2, 164 KiB shared memory per SM,
and 163 KiB per CTA. Warp MMA issue capacity corresponds to 1024 FMA/SM/cycle.
L2/DRAM latency, cache bandwidth, MMA latency and interconnect timing remain
uncalibrated model assumptions. No Ramulator2 backend is used.

The graph models the FA2 algorithm and warp-level work, not exact compiler
instruction scheduling. SIMD/SFU operations and register footprints remain
coarse estimates. KV storage is contiguous per head, without vLLM's paged-cache
physical address mapping. No launch overhead, shared-GPU competition, CPU weight
prefetch, cache append, RoPE, projections or complete decoder layer is modeled.

`fa2_decode_semantics` checks small batches, tails, empty partitions, global
bounds, dtypes, dependency acyclicity, padded MMA counts and absence of Hopper
instructions. Batch 1/2 smoke simulations must drain. An independent FP64 CPU
reference verifies online softmax and split merging against dense GQA; this
does not constitute numerical execution of the timing graph on a GPU.

## Sources

- [FA2 main and split-combine implementation](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h)
- [FA2 launch configuration](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_launch_template.h)
- [GQA query/head swap](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp)
- [Qwen3.5-9B configuration](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
- [Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
- [A100 PCIe 40GB product brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/A100-PCIE-Prduct-Brief.pdf)

## Local 32K trace comparison

With batch=1, 16 Q heads, 4 KV heads, D=256, BF16 and 47 splits, the Simple
Model completed all 3,304,520 graph nodes in 166,650 cycles at 1410 MHz:

| Phase | Simulated time | A100 trace mean (2040 calls) |
| --- | ---: | ---: |
| Main | 96.237 us | 113.326 us |
| Combine | 21.955 us | 30.097 us |
| Total | 118.191 us | 143.423 us |

The modeled total is 17.59% shorter. Trace inputs begin at 32K and grow during
decode, while this run holds KV length at 32768. These measurements compare
only the attention main/combine kernels, without projection, norm, MLP or CPU
offload time. The coarse timing and placement assumptions above still apply.
