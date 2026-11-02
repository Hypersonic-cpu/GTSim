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
        ├── flow_to_arch.py    # IR-03 → IR-04
        ├── arch_to_exec.py    # IR-04 → IR-05 / DAG
        └── nvidia/
            └── hopper.py     # Shared Hopper lowering rules
```

Each lowering stage has its own Python file and may share common abstractions.
The new implementation is independent of `src/gtsim_frontend/`. IR-02 → IR-03
scheduling remains unimplemented: compilation consumes explicitly supplied
`.flowir` files. This layout describes the target, not completed functionality.

## AST conventions

All five IR roots use explicit fields with no constructor defaults, schema
versions, or reference-kernel presets. `FlowIR` carries operations and a generic
`ScheduleConfig`, without algorithm-family dispatch. Categories use enums;
user-defined names and symbolic expressions remain strings. Validation belongs
to `parser/`, not the AST records.

Algorithm timing operations use `OperationClass` plus a cost subtype when needed:
INT/FLOAT add, subtract, and shift share simple arithmetic; multiply/divide may
have distinct cost classes. Special functions use grouped subtypes. Infinity is
a constant value. The AST stores no numerical execution or hardware latencies.
