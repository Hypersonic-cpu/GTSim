# Layered IR validation

Status: PASS

Every case uses the same hardware YAML, memory YAML and current fork build for both candidate and reference.

Graph parity includes operation names/types, tensor regions/strides, CTA resources, warp/SM placement and issue/completion edges.

| Workload | Cases | MAPE | maxAPE | Pearson | MAPE change (pp) |
|---|---:|---:|---:|---:|---:|
| gemm | 8 | 5.701983% | 10.796776% | 0.988988531 | 0.000000 |
| fa3 | 6 | 4.953760% | 6.822142% | 0.998801825 | 0.000000 |

Both relative accuracy alarms are evaluated against the corrected same-fork reference. Official v0.1.0 accuracy is a historical anchor, not a same-backend timing oracle.

This is a timing model; no numerical kernel execution or physical GPU validation is claimed.

## Runtime and regression evidence

All 23 active CTests passed. `workload_profile_boundary` was explicitly skipped
at the user's request; five legacy TVM checks are opt-in and were not included
in this final run. Twelve pure-Python contract tests are included in CTest.
The independent native matrix additionally covers two-SM persistent odd/even K
and supported GQA. An actual GQA (1,8,2,512,1536) run matched at 37,568 cycles.

| Experiment | Observed result |
|---|---|
| GEMM → FA3 | GEMM completes at 5,316; FA3 starts at 5,316 and completes at 12,528 |
| Repeated GEMM, shared allocations | HBM reads 98,304 bytes then 0; L2 line count retained |
| Explicit overhead | 17 configured cycles: total 5,316 → 5,333; device cycles unchanged |
| Producer output → next input | Same global binding across distinct local IDs; 131,072 completed write bytes |
| Empty stream | Zero cycles, no launches |
| Invalid execution graphs | Bad target/op/collective/output/cycle/binding rejected before launch |

The producer-consumer experiment initially assumed write allocation. That test
expectation failed: this backend only inserts an L2 line on a read response.
Both dependent and isolated first consumers therefore read 196,608 HBM bytes.
The corrected test checks stable identity and completion instead; the separate
warm-read experiment proves cache persistence. This failed expectation is
retained in `failed-test-expectation.json`; cache policy was not modified.

## GEMM synchronization correction

`pingpong-cross-wave-v2` adds only the missing buffer-1 completion guards in
even-K persistent waves. On two SMs with M=N=512, K=64/128/192/256 old/new cycles
are 13,373/17,469/23,185/25,919 respectively, unchanged by the correction.
K=128 and K=256 each add 3,840 completion edges; odd cases add none. The
independent event-graph audit now finds zero unsafe reuse paths in all four.
TMA issue ordering remains asynchronous; this is not a wholesale restoration
of upstream's more restrictive completion-order graph.

## Provenance and reproduction

`provenance.json` pins the implementation commit, upstream anchor, source/config
identity, compiler, submodules and binary hashes. All 14 run-spec source and
binary hashes were checked against the final implementation. Full generated
stages and logs are in `validation/results/ir-framework/final-validation/`;
small summaries and runtime evidence are tracked here. Every candidate graph
matches the separately compiled C++ oracle, including instruction defaults.
The original measured CSVs and stored cycle references are unchanged.

Reproduce using the commands in `docs/ir-framework-compat.md`. Optional official
historical accuracy anchors (GEMM 2.65%, FA3 9.79%) belong to upstream's original
backend and were not remeasured here. The acceptance comparison is against the
corrected current fork on identical YAMLs and memory backend.
