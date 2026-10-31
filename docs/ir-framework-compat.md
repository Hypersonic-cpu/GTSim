# Layered IR support and reproduction

Supported reference profile: Hopper H100, FP16 inputs/output, FP32 accumulation,
HBM placement, one ordered stream, one or more sequential launches.

- GEMM: BM128/BN256/BK64, two stages, 12 warps, original grouped-column order,
  persistent CTA waves. Positive M/N/K must divide the tile sizes exactly.
- FA3: Q128/KV128/D128, two stages, 12 warps, standard original CTA order,
  full online-softmax operation sequence. Positive batch/head counts, integral
  GQA ratio, and Q/KV lengths divisible by 128.
- Unsupported reference options fail: single-token/tail geometry, BF16, D256,
  split-K/split-KV, alternate CTA orders, multiple streams, unknown native
  operations, invalid resource bindings and invalid events/aliases.

This compiler models timing; it does not execute numerical tensor values.
No CUDA, PTX assembler, TVM or external Uniform package is required.
Ampere/Blackwell lowering, concurrent kernels and automatic schedule search
are outside this implementation.

Build from the repository root:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
python3 -m venv .venv
.venv/bin/python -m pip install -e ./frontend
.venv/bin/gtsim-ir run gemm --m 512 --n 512 --k 512 \
  --out-dir validation/results/ir-framework/gemm
.venv/bin/gtsim-ir run fa3 --batch 1 --heads-q 8 --heads-kv 2 \
  --seq-q 512 --seq-kv 1536 \
  --out-dir validation/results/ir-framework/gqa
.venv/bin/gtsim-ir inspect --input validation/results/ir-framework/gemm/kernel.flowir
.venv/bin/gtsim-ir verify --input validation/results/ir-framework/gemm/manifest.json
```

Defaults are `configs/v0.1.0/h100-sxm.yaml`,
`configs/v0.1.0/memory_full_hbm-0F5M.yaml`, and `build/gtsim_program`.
They can be overridden with `--architecture`, `--memory-config`, `--runner`.
`compile` emits every stage without running the program; `run` also records
source/config/binary hashes, simulation logs, kernel timing and metrics.
Multi-launch programs can be supplied as a canonical Program using `--input`.

Verification:

```bash
ctest --test-dir build -E '^workload_profile_boundary$' --output-on-failure
.venv/bin/gtsim-ir validate --oracle legacy --suite h100-v0.1.0 \
  --out-dir validation/results/ir-framework/validation
python3 tests/reference_parity/compare_sync_snapshots.py --before 84b4d81 \
  --out-dir validation/results/ir-framework/sync-comparison
```

`workload_profile_boundary` is skipped at the user's explicit instruction: its
text scanner mistakes read-only `config.memory` expressions for timing overrides.
The checker itself is unchanged. Previous TVM tests are opt-in using
`GTSIM_ENABLE_LEGACY_TVM_TESTS=ON` plus `GTSIM_TVM_PYTHON`. The new frontend and
runner do not import or execute that code. Core/reference executables retain
no Python dependency; the new runner can be disabled with
`GTSIM_BUILD_IR_FRAMEWORK=OFF`.

The 14-case validation command compares every graph field/edge to independent
C++ builders and requires exact same-build cycle parity before calculating
measured-data MAPE, maxAPE and Pearson. Accuracy alarms are relative increases
of 5% in MAPE or 10% in maxAPE, with percentage-point changes also recorded.
The original measured CSVs and stored reference cycles are unchanged.
