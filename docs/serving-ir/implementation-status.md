# Layered IR implementation status

The supplied P0–P6 plan is implemented for its H100 reference scope.
See [implementation](../ir-framework-implementation.md),
[wire contracts](../ir-framework-spec.md),
[commands and supported inputs](../ir-framework-compat.md), and
[the final validation report](results/report.md).

| Acceptance item                                                     | Result                                                    |
| ------------------------------------------------------------------- | --------------------------------------------------------- |
| Existing baseline captured before code changes                      | PASS; `baseline/`                                         |
| Typed IR-01/02/03/04/05, deterministic codecs and inspection        | PASS                                                      |
| Independent fixed GEMM/FA3 schedules and native lowering            | PASS                                                      |
| Generic JSON execution graph materializer, actual FA3 op set        | PASS                                                      |
| One persistent GPU/L2/MemorySystem, sequential launch/event/sync    | PASS                                                      |
| Mixed GEMM→FA3 and shared-allocation/warm-cache tests               | PASS                                                      |
| Explicit launch state transitions, overhead and memory-drain clocks | PASS                                                      |
| GEMM odd/even multiwave dependency audit and minimal correction     | PASS                                                      |
| All 8 GEMM + 6 FA3 graph and cycle comparisons                      | PASS; zero unapproved differences                         |
| Same-fork accuracy comparison and relative alarms                   | PASS; zero changes                                        |
| Active regression tests                                             | 23/23 PASS                                                |
| Workload profile text scanner                                       | Explicitly skipped at user request; false positive        |
| Previous TVM compatibility tests                                    | Opt-in; excluded from final acceptance per user direction |

GEMM MAPE is 5.7019825%, maxAPE 10.7967763%; FA3 MAPE is 4.9537596%,
maxAPE 6.8221418%. These are modeled timing comparisons with supplied measured
CSV targets, not numerical execution or new physical GPU measurements.

`Simulator::bind_kernel` separates launch-local scheduler/scoreboard lifetime
from the persistent GPU/memory lifetime. Program-global tensor allocations
remain stable; output completion includes issued memory drain without an L2
flush. Native output coverage, four-warp collectives and supported operations
are checked before launch. Unsupported reference shapes/options fail closed.

The corrected even-K GEMM dependency adds the missing prior buffer-1 consumer
completion edges. It does not change timing constants, arithmetic or traffic.
See [the dedicated sync report](gemm-sync-fix.md) for edge and cycle evidence.
The earlier baseline's passing cycle check was not treated as correctness proof.

No external Uniform code was copied and production compilation does not import
TVM, reference C++ builders or precomputed DAG fixtures. Generated multi-MB
artifacts stay under ignored `validation/results/ir-framework/`; small final
CSV/JSON reports and reproducibility utilities are tracked.

## Structured refactor follow-up

See `refactor-results/report.md` for the subsequent architecture correction.
Flow and native IR are now compact typed loop programs; full DAG expansion occurs
only in IR-05. The persistent runtime uses an extracted `KernelExecutionSession`
for launch-local state. Earlier size and wire descriptions in the original result
snapshot are historical; current contracts are in `docs/ir-framework-spec.md`.
