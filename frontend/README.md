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
        ├── expand_native.py   # Final loop expansion and event resolution
        ├── export_runtime.py  # Adapter to the existing C++ ProgramExecutor
        └── nvidia/
            └── hopper.py     # Shared Hopper lowering rules
```

Each lowering stage has its own Python file and may share common abstractions.
The new implementation is independent of `src/gtsim_frontend/`. Program/Algorithm
compilation composes explicitly supplied `.flowir` schedules into one symbolic
FlowIR. Flow → Arch selects structured Hopper instructions. Arch → Exec resolves
static scopes and expands loops into a concrete DAG. Automatic schedule synthesis
and non-GEMM target patterns remain outside the implemented path.

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

- `static.param` declares a compiler input; `static.bind` records its actual
  argument expression without eagerly resolving it.
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

The common syntax AST/parser recognizes branch blocks across IR levels. FlowIR
and ArchIR preserve symbolic shapes, parameter bindings, predicates, conditional
dependency rules, and iteration guards. No early constant folding or loop expansion
is performed by Program/Algorithm → Flow or Flow → Arch. Equivalent branch
alternatives may be represented as guarded statements during semantic parsing.

**Only Arch → Exec resolves `static.*`.** It binds lexical argument scopes,
checks constraints/shapes, selects active regions, evaluates predicates for each
iteration, and emits concrete instruction addresses and Issue/Complete edges.
ExecIR contains no unresolved expressions or conditional branches; its shared
static-declaration container is empty. The expression resolver never executes
Python expressions.

## Program/Algorithm → FlowIR

`prog_to_flow.py` resolves library symbols and ordered invocations;
`algo_to_flow.py` binds logical tensor identities and attaches typed algorithm
operations through explicit `implements<@operation>` references. Concrete shape
and parameter checks are deferred to Arch → Exec. The compiler reads the source files directly
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

The generated FlowIR contains two ordered regions and 12 scheduled operations.
Both invocations preserve the same symbolic schedule with different `static.bind`
arguments. Odd/even K branches are selected only during Arch → Exec.

## FlowIR → ArchIR → ExecIR

`flow_to_arch.py` and `nvidia/hopper.py` lower memory copies, matrix operations,
and stores into typed Native patterns with fragment loops, address expressions,
resource assignments, and phased dependencies. Acquire/release operations become
logical synchronization events; they do not introduce fake zero-duration GPU
instructions. Resource estimates follow the current reference schedule; hardware
timing constants are not changed.

`arch_to_exec.py` and `expand_native.py` consume those Native instructions directly.
They do not reconstruct algorithms from FlowIR or dispatch on workload names.
Only this boundary enumerates CTA/wave/K/fragment loops. It preserves tensor
identities, local resource scopes, ordered-region completion edges, and the
Issue/Complete distinction. Native dependencies, addresses, and loop edits affect
the resulting DAG.

```sh
PYTHONPATH=frontend/src python3 -m compile \
  frontend/asm/gemm/reference/two_gemms.flowir \
  -o frontend/asm/gemm/reference/two_gemms.nvir

PYTHONPATH=frontend/src python3 -m compile \
  frontend/asm/gemm/reference/two_gemms.nvir \
  -o frontend/asm/gemm/reference/two_gemms.dag.json
```

ArchIR `.nvir` and ExecIR DAG files currently use the parser's structural JSON
serialization, including enum/record types; Native IR remains structured rather
than containing expanded instruction instances. Both files deserialize to the
corresponding typed AST. The example expands to 1,062 + 2,656 instruction nodes
across two ordered regions.

Optional `--runtime-dir /tmp/gtsim-text-runtime` on the final command emits the
existing C++ ProgramExecutor wire format. Version discriminators live only in
this adapter, not in the ASTs. It preserves ordered launches, persistent global
tensor identities, and runtime memory-drain boundaries by materializing one DAG
per launch. `--launch-overhead` explicitly sets launch overhead (default zero).
The canonical ExecIR still contains the unified graph and region boundaries.

```sh
./build/gtsim_program configs/v0.1.0/h100-sxm.yaml \
  --memory-config configs/v0.1.0/memory_full_hbm-0F5M.yaml \
  /tmp/gtsim-text-runtime/program.gtsir
```

The two-GEMM example executed successfully through the existing runtime: the
first region completed at cycle 11,899 and the second at 38,948, with zero launch
overhead and persistent GPU/MemorySystem/L2 state. Each region's instructions,
addresses, resources, and dependency edges matched the independent C++ reference
for this example. This is focused GEMM evidence, not a full regression sweep or
support for new attention/vector workloads.
