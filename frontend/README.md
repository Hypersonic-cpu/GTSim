# GTSim frontend

The frontend keeps symbolic program stages compact and expands only the kernel
requested by the simulator:

```text
ProgIR -> FlowIR -> ArchIR -> NativeExpander -> bounded InstructionBatch IPC
```

`frontend/src/ast/` contains typed ProgIR, AlgoIR, FlowIR and ArchIR records.
`frontend/src/parser/` reads and writes their textual forms. The compiler
stages are:

- `compile/prog_to_flow.py` composes ordered program launches.
- `compile/algo_to_flow.py` specializes an algorithm with a supplied schedule.
- `compile/flow_to_arch.py` lowers FlowIR to structured ArchIR.
- `compile/expand_native.py` expands one region through
  `NativeExpander.expand_to(emit_cta, emit_instruction)` callbacks.
- `compile/streaming.py` owns the single Unix framed transport, manifest,
  prefetch plan, and bounded `InstructionBatch` emission.

The Python process retains source/ArchIR and launch metadata. It does not build
or serialize a complete program DAG. The C++ `gtsim_program` process owns one
persistent GPU/MemorySystem/L2 context, receives one kernel at a time, builds
that kernel's DAG with `KernelGraphBuilder`, runs it, retires it, and detaches
it before the next launch.

## Compile and run

```sh
PYTHONPATH=frontend/src python3 -m compile \
  frontend/asm/gemm/reference/two_gemms.progir \
  --lib frontend/asm/gemm/reference \
  --param NUM_SMS=1 \
  -o /tmp/two_gemms.flowir

PYTHONPATH=frontend/src python3 -m compile.run_text \
  frontend/asm/qwen3/one_block.progir \
  --lib frontend/lib \
  --params frontend/asm/qwen3/b1.json
```

The runtime entry point is stream-only:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target gtsim_program --parallel
PYTHONPATH=frontend/src python3 frontend/tools/run_qwen_simulations.py \
  --params frontend/asm/qwen3/b1.json \
  --runner build/gtsim_program \
  --architecture configs/v0.1.0/h100-sxm.yaml \
  --memory configs/v0.1.0/memory_full_hbm-0F5M.yaml \
  --out validation/results/text-qwen/stream-b1
```

For HBF prefetch experiments, pass `--prefetch-config`,
`--tensor-placements`, and optionally `--tensor-layouts`. Prefetch windows are
planned from actual CTA and pipeline-iteration IDs; HBF weight tensors remain
read-only and external.

## Tests

```sh
PYTHONPATH=frontend/src python3 -m unittest discover \
  -s frontend/tests -p 'test_*.py'
```

The frontend tests cover parser round trips, Flow/Arch lowering, callback-only
native expansion, prefetch capacity/lifetimes, and multi-CTA/multi-iteration
prefetch indexing.
