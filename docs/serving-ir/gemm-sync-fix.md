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
