# Layered IR implementation status

The baseline is recorded in `baseline/provenance.json` and the two per-case
CSVs. Raw configure/build/CTest/reference/validation logs remain under
`validation/results/ir-framework/baseline/` (ignored generated artifacts).
The baseline commit is the documentation commit `84b4d81` on `serve-sim`;
implementation proceeds on `feature/layered-serving-ir`.

## P0 baseline

Release build succeeded. 24/25 existing CTests passed; the preexisting
`workload_profile_boundary` check rejects read-only `config.memory` expressions
in both reference CLIs. The standalone profile check reports the same failure.
Both reference checks pass: GEMM 12,249 cycles and FA3 7,212 cycles.
The complete measured suite ran: GEMM 8 cases, MAPE 5.7019825%, Pearson
0.9889885; FA3 6 cases, MAPE 4.9537596%, Pearson 0.9988018.
Both measured CSV SHA256 hashes match the local pinned upstream revision.
Reference graph fingerprints and the focused two-SM buffer-lifetime audit are
recorded in `baseline/graph-sync-audit.json`. The audit finds 30 unguarded TMA
loads each for K=128 and K=256; K=64 and K=192 have none. This proves missing
graph completion paths, not a measured runtime race. The first affected load
and missing consumer identities are retained; full details are in the ignored
`baseline/gemm-sync-audit.json` runtime artifact.

Reproduce the isolated graph audit:

```bash
c++ -std=c++23 -O1 -Iinclude -I3rdparty/glaze/include -I3rdparty/hbfine/include \
  tests/reference_parity/reference_graph_dump.cpp -o /tmp/gtsim-reference-graph-dump
python3 tests/reference_parity/audit_gemm_sync.py /tmp/gtsim-reference-graph-dump \
  --output validation/results/ir-framework/baseline/gemm-sync-audit.json
```

## P1 IR skeleton

Immutable Python dataclasses represent Program, Algorithm, scheduled Flow,
Mapping, Hopper native instructions, and Execution kernel/program records.
The deterministic JSON codec validates required fields and format/version
identities. Parameterized GEMM and FA3 program builders reject unsupported
reference geometry, dtype and GQA bindings. Inspection and validation are
available through `python -m gtsim_frontend.cli` with `PYTHONPATH=frontend/src`.

Run `PYTHONPATH=frontend/src python3 -m unittest discover -s tests/ir_frontend -v`.
Ten tests cover stage round trips, deterministic hashes, streams/events,
tensor bindings, slot ownership, unsupported routes and dependency cycles.
The frontend is not yet an executable compiler: native lowerers,
full resource/collective verification, C++ persistent runtime and new-runner
parity/accuracy validation remain to be implemented. Existing CLI binaries
and legacy parsers are unchanged.

## P2 logical reference schedules

Independent GEMM and FA3 schedules now instantiate typed Flow operations,
SSA tiles, per-CTA executors/buffers and explicit CTA-to-SM placement. GEMM
uses original grouped-column ordering and persistent two-stage slot generations.
FA3 preserves GQA input identities and separate QK, local/shuffle/reduce max,
exponentiation, sum, online scaling, PV and normalization operations. Output
coverage and logical operation multiplicities are tested with hand-computed
expectations. Slot releases require the completion events of all known consumers;
SSA operands require producer readiness/completion dependencies.

These are logical schedules, not native instruction graphs or parity proof.
The native WGMMA collective expansion, resource legalization and full reference
comparison remain P3/P4 work. No candidate simulated-cycle claims are made.
`ir_frontend_contracts` is registered in CTest without a TVM dependency when
`GTSIM_BUILD_IR_FRAMEWORK=ON` and a Python interpreter is available.

## Synchronization audit observations (not a proven fix)

The fork orders prior TMA stages with issue edges; buffer reuse waits on
completion of the consumers two K iterations earlier. Cross-wave buffer 0
reuse selects the prior last or penultimate K-stage according to K parity.
Cross-wave buffer 1 reuse still needs an independent reachability check.
Output staging awaits the final WGMMA round; output TMA awaits staging.
These observations do not prove hazard freedom.

`Simulator::run` drains issued memory using a local `drain_cycle`, leaving
`Simulator::cycle` at DAG completion. A persistent runtime must advance its
global clock through the drain before admitting another launch, while
preserving the legacy API's reported device-cycle semantics. Each constructor
currently creates a new GPU; repeated constructors cannot implement the
required persistent-context program runtime.
