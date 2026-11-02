# Reference GEMM textual example

The three assembly sources describe a single symbolic GEMM program:

- `example.progir`: ordered invocation and tensor bindings.
- `gemm.algoir`: algorithm expressions, typed timing operations, and stages.
- `gemm.flowir`: explicit tile schedule, dependencies, and resource mapping.

The dimensions and `NUM_SMS` remain symbolic. Tile geometry is explicit through
`tile(128x256x64)` and `matrix_tile(64x64x16)`, with two pipeline stages.
`algo.traits` uses enum-backed structural traits. The historical
`export_textual.py` exports the previous syntax from the legacy implementation;
it is not part of the independent AST/parser path and does not regenerate this
revised syntax.

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

`parser` and the copied AST do not import `gtsim_frontend`. The `ast/` source
directory is loaded under the private Python namespace `gtsim_text_ast` to avoid
shadowing Python's standard-library `ast` module. The original IR definitions
are retained in `gtsim_frontend/ir/`. `ArchIR` and `ExecIR` definitions are copied
as well; their structural JSON read/write path is supported, while their
MLIR-like assembly syntax is not implemented in this step.

Symbol binding and lowering to simulator IR are outside this parser step.
