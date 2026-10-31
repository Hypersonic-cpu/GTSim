# GEMM cross-wave buffer-1 dependency fix

Synchronization semantics: `pingpong-cross-wave-v2`.

For an even number of K stages, buffer 0 was guarded by the previous tile's
penultimate WGMMA stage, while the next tile's k=1 loads could issue without
waiting for the previous final stage consuming buffer 1. Add completion edges
from that final stage to the five k=1 A/B load nodes. Odd K and within-tile
reuse rules are unchanged; arithmetic, regions, resources and timing parameters
are unchanged. Prior-stage TMA ordering remains issue ordering.

The independent event-graph audit found 30 unguarded loads at each of K=128
and 256 for 512x512 GEMM on two SMs. After the fix it finds zero at K=64,
128, 192 and 256. This is graph-contract evidence, not a numerical GPU test.

All 26 CTests other than the user-authorized skipped `workload_profile_boundary`
passed, including the existing optional TVM checks in this run. The complete
8 GEMM + 6 FA3 validation matrix is unchanged versus the recorded pre-fix
baseline. Those measured cases do not force persistent waves at 132 SMs.
No stored reference cycles were updated: 512-cubed remains 12249; FA3 smoke
remains 7212. Raw test/validation logs are `/tmp/gtsim-sync-ctest.log` and
`/tmp/gtsim-sync-validation.log`; corrected results are under the ignored
`validation/results/ir-framework/corrected/` directory.

## Same-backend two-SM old/new experiment

Both builders ran against the same current core and unchanged YAMLs. The
pre-fix builder was loaded from commit `84b4d81`. Each case is M=N=512.

| K | Before cycles | Corrected cycles | Added completion edges |
|---:|---:|---:|---:|
| 64 | 13373 | 13373 | 0 |
| 128 | 17469 | 17469 | 3840 |
| 192 | 23185 | 23185 | 0 |
| 256 | 25919 | 25919 | 3840 |

The dependency gap is closed without a cycle change in these observed runs.
The previous timing equality was therefore insufficient to prove the graph's
buffer-lifetime contract. No claim of an observed numerical/runtime race is made.
The correction retains legal asynchronous TMA issue ordering; it does not
restore upstream's broader TMA completion serialization.

Reproduce using `tests/reference_parity/compare_sync_snapshots.py` as documented
in `docs/ir-framework-compat.md`. The small comparison JSON is tracked under
`docs/serving-ir/results/`. Original graph fingerprints predate the expanded
metadata dump; compare named nodes/edges rather than attributing raw dump-hash
changes solely to synchronization.
