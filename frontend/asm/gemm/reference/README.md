# Reference GEMM textual example

The reusable assembly sources describe a symbolic GEMM program:

- `example.progir`: ordered invocation and tensor bindings.
- `gemm.algoir`: algorithm expressions, typed timing operations, and stages.
- `gemm.flowir`: explicit tile schedule, dependencies, and resource mapping.

The dimensions and `NUM_SMS` remain symbolic. Tile geometry is explicit through
`tile(128x256x64)` and `matrix_tile(64x64x16)`, with two pipeline stages.
`algo.traits` uses enum-backed structural traits.

These files use the design document's proposed inspection syntax.
`frontend/src/parser/` reads them into independent typed AST definitions in
`frontend/src/ast/`: `ProgIR`, `AlgoIR`, and `FlowIR`. Tensor shapes, invocation
references, algorithm expressions, loops, operations, dependencies, and mappings
are parsed into fields. Writing uses these fields rather than the original text.
Assembly round trips preserve AST values; whitespace is normalized.

```sh
PYTHONPATH=frontend/src python3 -m unittest discover -s frontend/tests -v
PYTHONPATH=frontend/src python3 - <<'PYTHON'
from pathlib import Path
from parser import parse_file, parse, serialize

for path in sorted(Path("frontend/asm/gemm/reference").glob("*ir")):
    ir = parse_file(path)
    print(path.name, type(ir).__name__, ir.symbol)
    assert parse(serialize(ir)) == ir
PYTHON
```

The `ast/` source directory is loaded under the private Python namespace
`gtsim_text_ast` to avoid shadowing Python's standard-library `ast` module.
New `.nvir` exports use MLIR-like text; ExecIR DAGs use JSON.

`two_gemms.progir` invokes the same algorithm and schedule twice, with K=192 and
K=256. `two_gemms.flowir` is the generated unified FlowIR: two symbolic scheduled
bodies connected through tensor H, with distinct `static.bind` arguments.
`two_gemms.nvir` retains structured Native loops and conditions;
`two_gemms.dag.json` contains the final 3,718-node ExecIR DAG. Parameter resolution,
odd/even branch selection, and loop expansion occur only at Arch → Exec.
Compilation commands and current scope are in `frontend/README.md`.
