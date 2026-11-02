# Frontend target architecture

This is the target architecture for future frontend work, superseding the
directory layout in `docs/serving-ir/GTSim_Frontend_Refactor.md`.

```text
frontend/
├── asm/                       # MLIR-like program and kernel source files
└── src/
    ├── ast/                   # ProgIR, AlgoIR, FlowIR, ArchIR, ExecIR and shared types
    ├── parser/                # Text IR ↔ typed AST; parsing errors handled here
    └── compile/
        ├── prog_to_flow.py    # IR-01 composition → unified IR-03
        ├── algo_to_flow.py    # IR-02 + supplied schedule → specialized Flow region
        ├── static.py          # Shared parameter/branch resolution
        ├── bind.py            # Typed dimension and tensor bindings
        ├── flow_to_arch.py    # IR-03 → IR-04
        ├── arch_to_exec.py    # IR-04 → IR-05 / DAG
        └── nvidia/
            └── hopper.py     # Shared Hopper lowering rules
```

Each lowering stage has its own Python file and may share common abstractions.
The new implementation is independent of `src/gtsim_frontend/`. Program/Algorithm
compilation specializes explicitly supplied `.flowir` schedules and composes them
into one FlowIR. Automatic schedule synthesis and the Flow → Arch → Exec stages
remain unimplemented; those parts of the layout describe the target.

## AST conventions

All five IR roots use explicit fields with no constructor defaults, schema
versions, or reference-kernel presets. `FlowIR` is an ordered container of
`FlowRegion` code sections. Each region carries operations, a generic
`ScheduleConfig`, and local resource/symbol scopes, without algorithm-family dispatch. Categories use enums;
user-defined names and symbolic expressions remain strings. Validation belongs
to `parser/`, not the AST records.

Algorithm timing operations use `OperationClass` plus a cost subtype when needed:
INT/FLOAT add, subtract, and shift share simple arithmetic; multiply/divide may
have distinct cost classes. Special functions use grouped subtypes. Infinity is
a constant value. The AST stores no numerical execution or hardware latencies.

## Shared compiler macros

`static.*` is a compiler namespace shared by all five IR levels. It is separate
from the algorithm, flow, native, and execution operation dialects:

All five IR roots inherit the same `Header.static: StaticDeclarations` field.
`StaticParameter`, `StaticBinding`, `StaticPredicate`, and their declaration
container are defined once in `src/ast/common.py`. `src/parser/static.py` parses
and prints them before dialect-specific processing; they do not belong to
ProgIR, FlowIR, or any other individual level. The common syntax AST defines
`StaticBranch` once for `static.if`/`static.elif`/`static.else`.

- `static.param` declares a compiler input; the driver binds its value.
- `static.let` defines a derived compile-time symbol, including tile indices.
- `static.pred` defines a reusable boolean expression.
- `static.require` declares a compile-time parameter constraint.
- `static.if`, `static.elif`, and `static.else` select source regions. The first
  matching branch is retained; an `else` branch is optional.

```text
static.param %B : index
static.pred %single_request = %B == 1
static.if <%single_request> {
  prog.launch ...
}
static.elif <%B == 8> {
  prog.launch ...
}
static.else {
  prog.launch ...
}
```

These are compiler macros, not simulated instructions. The compiler binds typed
parameters and resolves expressions in the AST, without rewriting source text.
It must resolve every macro before emitting an executable DAG. Static expressions
may depend on enclosing loop-instance indices: a Flow condition involving `%k`
is resolved when that iteration is expanded, not necessarily at initial parameter
binding. Optional `scope[%cta, %wave]` identifies the iteration scope gated by a
branch. Tensor values and kernel results are not static symbols.

Actual operations remain in their own dialect, for example
`static.if <%has_previous_k> { flow.dependency_rule ... }`. No Flow-specific
`when` or `static_if` attribute is part of the source format.

The common syntax AST/parser recognizes branch blocks across IR levels. The
compiler resolves parameter-dependent `static.if`/`static.elif`/`static.else`
branches on the syntax AST before semantic parsing, then binds typed dimensions,
indices, and tensors. Loop-dependent conditions remain structured until a later
expansion stage; GTSim receives no runtime static branches. The expression
resolver supports the example's integer arithmetic, comparisons, booleans,
`ceildiv`, `floordiv`, `min`, and `max`, without executing Python expressions.

## Program/Algorithm → FlowIR

`prog_to_flow.py` resolves library symbols and ordered invocations;
`algo_to_flow.py` specializes the selected schedule, checks tensor bindings against
the algorithm signature, and attaches typed algorithm operations through explicit
`implements<@operation>` references. The compiler reads the source files directly
and transforms AST objects. It neither invokes legacy workload generators nor
rewrites IR source text.

A `.flowir` output contains the actual scheduled bodies of every invocation,
not calls back to kernel definitions. `flow.region` provides local resources,
loop scopes, and explicit `after[...]` completion ordering. `origin<@algorithm>`
is provenance metadata. Region boundaries retain execution/resource scope; they
do not imply fusion. The shared intermediate is a single global tensor identity.
IR-03 stays structured rather than expanding tile loops into a DAG.

The two-GEMM example reuses `gemm.algoir` and `gemm.flowir`:

1. `[256,192] × [192,256] → H[256,256]`: three K tiles, odd-count branch.
2. `H[256,256] × [256,512] → Y[256,512]`: four K tiles, even-count branch.

```sh
PYTHONPATH=frontend/src python3 -m compile \
  frontend/asm/gemm/reference/two_gemms.progir \
  --algo frontend/asm/gemm/reference/gemm.algoir \
  --flow frontend/asm/gemm/reference/gemm.flowir \
  --param NUM_SMS=1 \
  -o frontend/asm/gemm/reference/two_gemms.flowir

python3 -m unittest discover -s frontend/tests -p 'test_*.py'
```

The generated file round-trips into one typed `FlowIR` with two ordered regions,
12 scheduled operations, concrete shapes and K-loop bounds, and distinct selected
buffer-reuse rules. Loop-instance macros remain for subsequent lowering. This
stage does not emit Native IR/DAG JSON or run the simulator.
