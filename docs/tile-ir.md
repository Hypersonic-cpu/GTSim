# Experimental TVM / Tile IR integration

This frontend is separate from the v0.1.0 reference builders. It never changes
hardware timings, scheduling, memory routing, measured targets, or reference
cycles. TVM is pinned to `4d0ac168c78c1e20bf05a6ec5926760bb1c52b05`.

For computation-only GEMM and native MetaSchedule performance feedback, see
[the automatic scheduling guide](tvm-auto-schedule.md). The Phase 2/3 and CSV
sections below describe the retained manually scheduled reference path.

## Fresh-workspace setup and Phase 1 smoke

Start with a checkout containing this TVM integration implementation. Install
Git, CMake 3.26+, Ninja 1.11+, Python 3.10 with `venv`/pip support, and a C++23
compiler plus its platform SDK. The CMake 3.26 minimum comes from the pinned
TVM-FFI submodule. The commands below use Python 3.10, as in the tested setup.

Run from the repository root. A CUDA device/toolkit and LLVM are unnecessary for
this IR-construction/GTSim workflow. Build the pinned TVM compiler from source;
do not install a different `apache-tvm` wheel or use editable installs.
Keep the Python environment at `.tvm-venv/` in the repository root; deleting
the disposable GTSim `build/` directory then preserves all TVM Python packages.
The separately compiled TVM libraries stay in `build-tvm/`.

```sh
# Initialize the recorded submodule commits, including TVM's nested dependencies.
# The temporary URL rewrite allows GitHub HTTPS access without an SSH key.
git -c 'url.https://github.com/.insteadOf=git@github.com:' \
  submodule update --init --recursive 3rdparty/glaze 3rdparty/tvm

# Create the persistent Python environment and install the pinned-source FFI.
python3.10 -m venv .tvm-venv
.tvm-venv/bin/python -m pip install --upgrade pip
.tvm-venv/bin/python -m pip install \
  ./3rdparty/tvm/3rdparty/tvm-ffi \
  numpy==2.2.6 ml_dtypes==0.6.0 typing_extensions==4.16.0

# Compile the pinned TVM separately; the helper also writes its build manifest.
.tvm-venv/bin/python scripts/tvm/build_tvm.py --jobs 4

# Only after TVM setup, configure/build GTSim and run the interface smoke test.
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGTSIM_TVM_PYTHON="$PWD/.tvm-venv/bin/python"
cmake --build build --parallel 4
.tvm-venv/bin/python scripts/tvm/smoke.py
```

The FFI install builds from TVM's pinned nested submodule; pip installs its
declared build dependencies in an isolated build environment. The TVM build
helper verifies commit `4d0ac168c78c1e20bf05a6ec5926760bb1c52b05`, configures
`USE_LLVM=OFF`, `USE_CUDA=OFF`, `USE_Z3=OFF`, and `USE_RPC=OFF`, and writes
`build-tvm/gtsim-tvm-build.json`. The frontend loads Python code directly from
`3rdparty/tvm/python` and verifies the compiled libraries against that manifest.
No separate system TVM installation or manual `PYTHONPATH` is needed.

After the smoke test, run validation followed by parameter search:

```sh
.tvm-venv/bin/python scripts/tvm/validate_tvm_gemm.py --allow-measurement-error --jobs 4
.tvm-venv/bin/python scripts/experiments/run_tvm_gemm_sweep.py --params scripts/experiments/tvm_gemm_tiles.csv --jobs 4 --repeats 2
```

The smoke constructs an actual TVM TIR program, extracts a single warp's FP16
row-copy accesses, writes `.gtsir`, invokes `gtsim_gemm_ir`, and saves TIR stage
snapshots, command provenance and simulation output under
`validation/results/tvm/smoke/`. This proves the interface, not GEMM lowering.
The smoke also saves a symbolic Hopper representation probe with actual TMA,
WGMMA and mbarrier calls through simplification, buffer flattening and intrinsic
lowering. The probe uses descriptor arguments; it is not an executable GEMM.
The source checkout and hashes of the actual loaded compiler/runtime libraries
are checked against the build manifest at import time. This TVM pin's
`support.libinfo()` is a Python fallback without native commit metadata.

The pinned `python/tvm/s_tir/pipeline.py` lowers blocks, matches buffers, injects
software pipelines, then calls `FlattenBuffer`. It later injects thread sync and
splits host/device functions. Its device finalization lowers warp memory and
intrinsics. `python/tvm/tirx/compilation_pipeline.py` likewise flattens buffers
before finalization. The extraction point is **scheduled TIR before
`FlattenBuffer` and `LowerIntrin`**, after an intrinsic has been explicitly
selected. Rank, shape, access expressions, thread bindings, PTX instructions and
barrier protocol are available there. An ordinary matrix multiplication is not
evidence of a TMA or WGMMA selection. Pinned Hopper examples are in
`tests/python/tirx/codegen/test_codegen_hopper.py` in the TVM submodule.

## Version 1 wire contract

UTF-8 text, one statement per line; `#` begins a comment. The first non-comment
line is `gtsim.tile_ir 1`. Each remaining line is a tensor, CTA, or operation:

```text
tensor id=0 name=A shape=[1,32] strides=[32,1] dtype=fp16 tier=HBM
cta id=0 sm=0 warps=12 shared=0 registers=1536
%0 = ld.dram2reg name=load cta=0 warp=0 tensor=0 dtype=fp16 tile=[0,0,1,32] after_issue=[] after_complete=[]
```

Fields may be reordered; every displayed field is required. Unknown, duplicate
or missing fields fail. Names are nonempty tokens without whitespace or `#`.
Integers are nonnegative signed 32-bit values, with positive dimensions and tile
extents. Lists contain comma-separated integers without spaces. IDs refer to
objects, never GPU addresses. `TensorRegistry` assigns synthetic, nonoverlapping
addresses in tensor declaration order within each device aperture: HBM starts
at `2^32`, HBF at `2^60`. Allocations are 4 KiB aligned; devices never alias
in the shared L2 address space. Reference builders retain their legacy allocator.

Tensors have rank two, `fp16` or `fp32` dtype, element strides `[leading_dim,1]`
with `leading_dim >= columns`, and explicit **HBM** or **HBF** placement. CTAs require a valid SM,
12 warps, nonnegative shared bytes and positive register bytes. Footprints must
fit the supplied hardware limits. Warp identities are local `[0,12)`; their DAG
identity is `cta_id * 12 + local_warp`. All nodes of a CTA use its declared SM.
All existing topologies are accepted when they contain every declared device.
FullHBM rejects HBF placement and FullHBF rejects HBM placement before Simulator
construction. CoLocated and Cascaded preserve both devices; the existing memory
system supplies their controllers, link routing, contention and latency.

| Operation | Pipeline | Region / dtype |
|---|---|---|
| `ld.dram2reg` | LD | FP16/FP32 tensor region |
| `st.reg2dram` | ST | FP16/FP32 tensor region |
| `cp.dram2sram_tma` | LD | Explicitly selected TMA load |
| `cp.sram2dram_tma` | ST | Explicitly selected TMA store |
| `st.reg2sram` | ST | Output staging region |
| `wgmma.m64n64k16` | Tensor | FP32 accumulator fragment, 16x64, FP16 output tensor identity |

`tile=[row_offset,column_offset,rows,columns]` is a row-major access or accumulator
region in the referenced tensor's logical coordinates. Every region must fit
that tensor. Four consecutive compute warps form a WGMMA collective; each has a
16x64 fragment of the same 64x64 output region. Only compute warps 0..7 support
WGMMA. Ordered collective fragments must match across all four warps.
The WGMMA operation fixes reduction width to 16; A/B global traffic is represented
by the explicit TMA regions and completion dependencies. It is not encoded as
extra global traffic on the accumulator node. Shared store conversion matches
the reference's FP32 accumulator to FP16 output staging abstraction.

Operation IDs must be dense and appear in `[0,N)` order; names must be unique.
`after_issue` waits until a predecessor has issued all its transactions;
`after_complete` waits for completion. References may point forward, but the
combined graph, including collapsed four-warp collectives, must be acyclic.
Sibling fragment dependencies fail rather than deadlocking. Edge list order and multiplicity are preserved
(the reference repeats some store dependencies). Unrecognized operations,
dimensions, layouts, control flow, physical addresses, timing overrides, or
unsupported TVM constructs fail explicitly. Version 1 has no implicit edges,
loops, dynamic scheduling or tuning knobs.

```sh
build/gtsim_gemm_ir configs/v0.1.0/h100-sxm.yaml --memory-config configs/memconf/full-hbm.yaml tests/fixtures/copy.gtsir
ctest --test-dir build -R tile_ir --output-on-failure
```

## Phase 2 manual schedule

`scripts/tvm/manual_gemm.py` is the independently authored, parameterized manual
specification. It writes fully expanded version 1 files; it does not call or
dump `build_gemm_h100()`. `tests/fixtures/h100_gemm.gtsir` is the standalone
128x256x128 instance. The parser needs no special GEMM expansion opcode.

The specification fixes BM=128, BN=256, BK=64, two stages, split=1, 12 warps,
and original CTA order. Columns are grouped by eight, then rows, then columns
inside the group. Grid size is `min(SMs, output_tiles)`; CTA b owns tile indices
`b, b+grid, ...` and SM b. Compute warps 0..7 issue two M halves, two N chunks
per warpgroup and four K steps; loader warp 8 issues A plus four B TMA loads.
Input buffer k%2 is reused only after its prior WGMMA consumers complete.
TMA stages use issue ordering. All final WGMMA fragments gate output shared
stores; four TMA output chunks follow, chained by issue edges. The next output
tile's compute waits for prior output transfers, while its loads use the
appropriate prior ping-pong buffer completion edges.

The test oracle includes the **untouched** reference builder only in its test
translation unit. It compares every operation, tensor shape/stride/address,
access region, CTA/SM/warp mapping, resource footprint, and issue/completion
dependency list. The sole expected metadata difference is A/B tier: the
reference tags them HBF and FullHBM routes them to HBM; the original frontend required
explicit HBM tags.

```sh
python3 scripts/tvm/manual_gemm.py 512 512 512 -o validation/results/tvm/manual.gtsir
python3 scripts/tvm/validate_gemm_ir.py --build-dir build --jobs 4
```

The gate runs all eight measured GEMM cases through both executables with the
same architecture and `configs/memconf/full-hbm.yaml`. Results, per-case graph
comparisons, commands, hashes and simulation logs go under
`validation/results/tvm/manual/`. Every case must have exact reference graph and
cycle parity and strictly less than 10% error against measured cycles. A failed
accuracy gate stops the original plan. The user subsequently waived that stop
condition explicitly. The original Phase 2 report remains a failure; the
automatic validation records the waiver separately and never changes targets.

## Phase 3 reference schedule generation

To reproduce validation and the parameter search after deleting `build/`, run
the following from the repository root (the existing `.tvm-venv/` and
`build-tvm/` are reused):

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGTSIM_TVM_PYTHON="$PWD/.tvm-venv/bin/python"
cmake --build build --parallel 4
.tvm-venv/bin/python scripts/tvm/validate_tvm_gemm.py --allow-measurement-error --jobs 4
.tvm-venv/bin/python scripts/experiments/run_tvm_gemm_sweep.py --params scripts/experiments/tvm_gemm_tiles.csv --jobs 4 --repeats 2
```

Validation prints all eight comparisons and saves `report.md` / `results.json`
under `validation/results/tvm/automatic/`. The search prints every candidate and
its best tested configuration, saving reports under
`validation/results/tvm/tile-sweep/`. Validation must precede the search because
the latter verifies its source hashes. `--allow-measurement-error` records the
authorized accuracy waiver; it does not suppress the reported error.

For a single kernel and an independent rerun of the validation:

```sh
.tvm-venv/bin/python scripts/tvm/gemm.py 512 512 512 --run --out-dir validation/results/tvm/generated
.tvm-venv/bin/python scripts/tvm/validate_tvm_gemm.py --allow-measurement-error --jobs 4
```

`h100_schedule.py` constructs scheduled TIR with native tensor-map encoding,
explicit TMA loads/stores, FP32 register accumulators, two shared input stages,
producer warp 8, one or two consumer warpgroups, WGMMA fence/MMA/commit/wait,
full/empty mbarriers, and output CTA joins. It uses the pin's `PTXNamespace`
selection and operand validation. Explicit `setmaxnreg` redistribution gives
compute threads BM+32 registers and producer/idle threads 32; its aggregate
matches the reference resource footprint. The shared footprint follows the
reference's A/B/C payload convention; barrier control storage is not charged
separately by this simulator contract.

TVM `ConvertSSA`, `verify_ssa`, and `verify_well_formed` check the generated IR.
The generation command also executes `StmtSimplify`, `FlattenBuffer` and
`LowerIntrin`, saving a post-lowering snapshot. Extraction uses the earlier
scheduled module, serialized and deserialized with native TVM JSON APIs.

`reference_gemm_lower.py` is a strict, versioned schedule recognizer, not a general TIR
compiler: it matches the **entire** supported scheduled AST, including every
intrinsic, operand, layout, binding and synchronization operation. It then
evaluates the actual native TMA coordinate expressions and tensor-map boxes,
and maps the actual output register/store indices to the four-warp fragments
and canonical 16x16 store regions. Barrier protocol and accumulator data flow
become completion edges; ordered TMA issue becomes issue edges. Removing an
intrinsic or barrier fails recognition. The production generator/lowerer does
not import or invoke the manual fixture or reference builder.

The output directory contains:

- `scheduled.py` and `scheduled.json`: the selected native TVM schedule;
- `lowered-tvm.py`: the real pinned TVM post-extraction pass output;
- `kernel.gtsir`: the expanded graph input;
- `lowering.json`: recovered shapes, bindings, native operations, resources,
  TIR/IR/graph-spec hashes and compiler provenance;
- `simulation.log`: simulator output when `--run` is given.

To lower a saved module independently:

```sh
.tvm-venv/bin/python scripts/tvm/gemm.py --input-tir validation/results/tvm/generated/scheduled.json --run --out-dir validation/results/tvm/reloaded
```

Scope is deliberately bounded: compact rank-2 FP16 A/B/C, two stages, 12 warps,
split=1, original order, explicitly declared HBM/HBF devices on a compatible
topology, and persistent CTAs with one or more output
tiles each. The lowerer expands the native `wave` loop and evaluates its guard
and TMA coordinates for each CTA, including an incomplete final wave. CTA/SM
and warp assignments and per-CTA resources remain fixed across its tiles.

The selected TVM TIR is authoritative for synchronization. It ends each tile
with `cp.async.bulk.commit_group`, `cp.async.bulk.wait_group(0)` without `.read`,
and a CTA barrier. Therefore **every prior output TMA completion gates the next
tile's first input loads**; issue-only dependencies are insufficient. Later
loads and compute inherit that boundary. Input-stage reuse within a tile waits
for consumers of the same ping-pong slot to complete. The boundary also covers
both slots across tiles, including odd K iteration counts. No extra barrier
latency or hardware timing is introduced: these are existing completion edges.
The C++ builder permits different cross-wave overlap and is not the oracle for
this extension. Its existing manual graph path and reference timings are kept.

`semantic_checks.py` independently expands the actual TIR control flow with TVM
symbolic bindings and checks TMA regions, four-warp collective counts, barrier
phases, completion-event reachability, buffer reuse and output joins. An
issue-only edge does not count as predecessor completion. The validation suite
runs seven multi-wave cases twice each on the configured hardware: one/two/three/
four K steps, both compute-warpgroup counts, multiple rows, clamped column
mapping and incomplete final waves. Eight historical single-wave cases retain
exact manual/builder graph and cycle parity. Altered waits/barriers are rejected,
and a deliberately weakened cross-wave graph fails the semantic audit.

Arbitrary scheduled TIR, generic GEMM, alternate layouts/synchronization,
partial tiles and incomplete warpgroups remain unsupported. These checks verify
compiler representation and GTSim behavior; they do not execute a numerical
CUDA kernel or measure physical H100 performance.

Optional native-TVM tests can be enabled without adding TVM to the core build:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGTSIM_TVM_PYTHON="$PWD/.tvm-venv/bin/python"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Phase 4 constrained exploration

```sh
.tvm-venv/bin/python scripts/experiments/run_tvm_gemm_sweep.py --params scripts/experiments/tvm_gemm_tiles.csv --jobs 4 --repeats 2
```

The CSV selects M/N/K and BM/BN/BK. Supported tile axes are BM={64,128},
BN={128,256}, BK={32,64,128}, subject to divisibility, complete intrinsic layout,
hardware capacity. Two stages, 12 warps, split=1, original
order and FullHBM stay fixed. Optional CSV fields for these settings must match
their fixed values. Unknown fields and duplicate schedules fail explicitly.

Hardware capacity and topology come from the existing C++ YAML reader through
the runner's read-only `--describe-config` query, so Python never silently
substitutes a different profile. Invalid candidates are recorded before
simulation. The launcher requires the eight-case Phase 3 report, its explicit
accuracy waiver when needed, successful multi-wave TIR synchronization checks,
and matching frontend/core/configuration hashes.
It generates a native TVM kernel for each distinct candidate and simulates every
accepted schedule at least twice. At least two distinct TIR kernels and graph
specifications must complete reproducibly.

Results are saved under `validation/results/tvm/tile-sweep/`: `results.csv`,
`results.json`, `report.md`, input snapshots, a separate fixed baseline per
problem size, and readable per-candidate directories containing all generated
artifacts/logs. Records include pinned TVM/GTSim revisions, source hashes,
parameters, TIR/IR/graph-spec hashes, resources, cycles, fixed-baseline speedup
and failure reasons. The checked-in workload JSON is also compatible with the
existing batch launcher's one-workload-template / `--params` convention; use
the dedicated sweep launcher for gate checks, repeated runs and comparison
reports.

For the supplied 512x512x512 sweep, 11 candidates complete reproducibly and
128x256x128 is rejected for requiring 262144 shared bytes. The best **tested
simulated** configuration is BM=64, BN=128, BK=128: 6500 cycles versus the fixed
128x256x64 baseline's 12249 cycles, a 1.8845x simulated speedup. This is not an
optimality claim or a physical-H100 result.


For the existing 2048x2048x512 CSV, preserve previous results in a separate output
folder and recompute both validation and the fixed baseline with the new frontend:

```sh
.tvm-venv/bin/python scripts/tvm/validate_tvm_gemm.py \
  --allow-measurement-error --jobs 4 \
  --out-dir validation/results/tvm/automatic-multiwave
.tvm-venv/bin/python scripts/experiments/run_tvm_gemm_sweep.py \
  --params scripts/experiments/tvm_gemm_tiles_2048.csv \
  --validation validation/results/tvm/automatic-multiwave/results.json \
  --out-dir validation/results/tvm/tile-sweep-2048x2048x512-multiwave \
  --jobs 4 --repeats 2
```


## Explicit heterogeneous placement

The native generator accepts `tensor_tiers=("HBM", "HBF", "HBM")` and declares
A/B/C with `global.hbm` / `global.hbf` TVM buffer scopes. The lowerer validates
that these address spaces match `gtsim.tensor_tiers`, including after native
TVM JSON serialization, and exports each `.gtsir` tensor's `tier`. A node's
`matrix_id` references that immutable registry allocation; no duplicated device
label on an operation can diverge from its address. L2 derives the request tier
from the allocated address and the existing MemorySystem routes to HBMModel or
HBFModel. Shared buffers, TMA, WGMMA and barriers are identical across placements.

```sh
.tvm-venv/bin/python scripts/tvm/gemm.py 512 512 512 \
  --placement hybrid --memory-config configs/memconf/colocated-3F2M.yaml \
  --out-dir validation/results/tvm/hybrid --run
.tvm-venv/bin/python scripts/experiments/run_tvm_placement.py
```

Use `--placement baseline` for A/B/C in HBM on the **same** CoLocated profile.
For imported `--input-tir`, the TIR's declared spaces are authoritative.
`gtsim_gemm_ir ... kernel.gtsir --inspect` exports the actual registry bases,
CTA resources and materialized DAG, without simulation. See
[tvm-hbf-placement.md](tvm-hbf-placement.md) for results and assumptions.

## Serving extension: version 2

The experimental serving compiler uses `gtsim.tile_ir 2`, preserving the
version-1 parser/materializer behavior and reference cycles. Version 2 adds BF16,
1..32 warp CTAs, Ampere `mma.sync.m16n8k16`, non-TMA `cp.dram2sram`,
shared/register loads, explicit SIMD/SFU operations, shuffle and zero-setup
completion joins. Hopper WGMMA/TMA are rejected in this version.

Every v2 operation additionally requires `granularity=4|8|16` bytes per thread
and `layout=row|swizzled`. Global accesses retain bounds, dtype and device
validation. Shared/register/compute tiles describe physical local fragments,
which may include padded rows outside the logical output; their materialized
`matrix_id` is -1. They cannot create global-memory traffic through a dummy
tensor identity. Shared accesses must fit the declared CTA allocation.

Global warp IDs are unique and assigned so consecutive CTA warps on each SM
rotate through its four subpartitions. They are no longer `cta*12+warp` in v2;
that v1 mapping is unchanged. Resource limits and all hardware timing remain
in the shared configuration. A100 scalar throughput was independently corrected
in both A100 profiles to 16 FP32 lanes/SP, not overridden by the importer.

See [the serving study](serve-tvm.md) for semantic input, explicit TensorIR
placement, native MetaSchedule feedback and numerical/provenance checks.
