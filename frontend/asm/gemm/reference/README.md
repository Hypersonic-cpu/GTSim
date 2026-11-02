# Reference GEMM textual example

`export_textual.py` calls the existing `gemm_single.build` and canonical
`passes.instantiate` generator, then writes the three MLIR-style artifacts shown
by `docs/serving-ir/llm_serving_ir_design_v0_2.md`:

- `gemm.algoir`: GEMM semantics and symbolic `M`, `N`, and `K` domains.
- `gemm.flowir`: the generated H100 Reference tile schedule, its operations,
  dependencies, and mapping.
- `example.progir`: one ordered stream that launches this GEMM and synchronizes.

The workload dimensions and `NUM_SMS` remain symbolic in the emitted files.
The numeric tile dimensions are fixed by the existing Reference profile
(`BM=128`, `BN=256`, `BK=64`, two stages, fp16). This profile requires complete
tiles. Regenerate the files with:

```sh
python3 frontend/asm/gemm/reference/export_textual.py
```

These files use the design document's proposed inspection syntax; the current
frontend does not yet parse this textual form.
