#!/usr/bin/env python3
"""Compare optimized Full-HBM and 4F1M on the Experiment 002 Q projection."""

import argparse
import asyncio
from collections import Counter
import csv
from dataclasses import replace
import os
from pathlib import Path
import signal

from common import sweep_scheduler as queue
from report_q_projection import best_row, metrics, table
from sweep_q_projection import (
    FRONTEND, OUTPUT, ROOT, SPLITS, Candidate, QSchedule, compile_case,
    input_manifest, phase_a, result_present, run_phase, same_simulation_inputs, write_ir,
)


RESULTS = OUTPUT / "full-hbm"


def compare(rows, original, reference):
    successful = [row for row in rows if row["status"] == "SUCCESS"]
    hbm = best_row(successful)
    hbf = original["best_4f1m"]
    split_best = [best_row([row for row in successful if row["split_k"] == split])
                  for split in SPLITS]
    counts = Counter(row["status"] for row in rows)
    with (RESULTS / "case_results.csv").open("w") as stream:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)
    if not hbm or not hbf:
        raise RuntimeError("no valid timing result for one or both hardware configurations")
    hbm_cycles, hbf_cycles = hbm["total_cycles"], hbf["total_cycles"]
    matched_hbm = best_row([row for row in successful if all(
        row[key] == hbf[key] for key in ("bn", "bk", "split_k", "ordering"))])
    summary = dict(best_full_hbm=hbm, best_4f1m=hbf, fixed_full_hbm_reference=reference,
                   hbm_at_best_4f1m_compute=matched_hbm, full_hbm_by_split=split_best,
                   full_hbm_statuses=counts, complete_grid=len(successful) == 114,
                   cycles_ratio_4f1m_over_full_hbm=hbf_cycles / hbm_cycles)
    queue.write_json(RESULTS / "comparison.json", summary)

    def line(label, row):
        return (label, row["bn"], row["bk"], row["split_k"], row["ordering"],
                f"{row['pf_bn']} × {row['pf_bk']}, ahead {row['ahead']}" if row["ahead"] else "OFF",
                f"{row['total_cycles']:,}")

    fastest = "Full-HBM" if hbm_cycles < hbf_cycles else "4F1M"
    gap = 1 - min(hbm_cycles, hbf_cycles) / max(hbm_cycles, hbf_cycles)
    coverage = "All 114 Full-HBM configurations succeeded." if len(successful) == 114 else (
        f"{len(successful)}/114 Full-HBM configurations succeeded. The comparison uses measured successes.")
    lines = [
        "# Experiment 002 optimized hardware comparison", "",
        coverage,
        "Full-HBM used the same compute geometry and ordering grid as 4F1M Phase A. "
        "Split-K=1 ordering aliases were deduplicated. 4F1M also uses its Phase B prefetch search.",
        "",
        *table(("Configuration", "BN", "BK", "Split-K", "Ordering", "Prefetch", "Cycles"), [
            line("Best Full-HBM", hbm), line("Best 4F1M", hbf),
            line("Original Full-HBM reference", reference),
            *([line("Full-HBM at best 4F1M compute", matched_hbm)] if matched_hbm else []),
        ]), "",
        f"The best results tie at {hbm_cycles:,} cycles." if hbm_cycles == hbf_cycles else
        f"{fastest} used **{gap:.1%} fewer cycles** than the other hardware's fastest tested case.",
        "",
        *table(("Split-K", "Best Full-HBM cycles", "Best 4F1M cycles"),
               ((split, f"{left['total_cycles']:,}" if left else "Unavailable",
                 f"{right['total_cycles']:,}" if right else "Unavailable")
                for split, left, right in zip(SPLITS, split_best, original["best_by_split"]))),
        "",
    ]
    (RESULTS / "comparison.md").write_text("\n".join(lines))
    return hbm


async def run(runner):
    os.chdir(ROOT)
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(FRONTEND / "src"), os.environ.get("PYTHONPATH"))))
    original_manifest = queue.read_json(OUTPUT / "candidate_manifest.json")
    original = queue.read_json(OUTPUT / "best_candidates.json")
    current_inputs = input_manifest(runner)
    provenance = original_manifest.get("provenance", {})
    if not same_simulation_inputs(current_inputs, provenance):
        raise RuntimeError("current compiler, simulator or configuration differs from the 4F1M sweep")
    pilot = Candidate("pilot", QSchedule(256, 64, 1, "n_major"), "fullhbm")
    pilot_job = pilot.job(runner, provenance)
    pilot_status = queue.read_json(pilot.directory / "case_status.json")
    if (pilot_status.get("status") != "SUCCESS" or
            pilot_status.get("fingerprint") != pilot_job.fingerprint or not result_present(pilot_job)):
        raise RuntimeError("the existing Full-HBM pilot could not be validated")
    hbf = original["best_4f1m"]
    hbf_case = next(entry for entry in original_manifest[hbf["phase"].replace("-", "_")]
                    if entry["id"] == hbf["case"])
    hbf_schedule = QSchedule(**{key: value for key, value in hbf_case["parameters"].items()
                                if key != "bm"})
    hbf_job = Candidate(hbf_case["phase"], hbf_schedule).job(runner, provenance)
    hbf_status = queue.read_json(hbf_job.output / "case_status.json")
    if (hbf_status.get("status") != "SUCCESS" or
            hbf_status.get("fingerprint") != hbf_job.fingerprint or not result_present(hbf_job) or
            queue.read_json(hbf_job.output / "result.json")["timing"]["total_cycles"] != hbf["total_cycles"]):
        raise RuntimeError("the existing best 4F1M result could not be validated")
    original_cases, _ = phase_a()
    others = [replace(case, phase="full-hbm", hardware="fullhbm") for case in original_cases
              if case.schedule.ahead == 0 and case.schedule != pilot.schedule]
    assert len(others) == len({case.id for case in others}) == 113
    scheduler = queue.SweepScheduler(RESULTS, ROOT, job_ram=(
        queue.PILOT_RAM_MULTIPLIER * pilot_status["resources"]["peak_ram_bytes"]))
    try:
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signum, task.cancel)
        queue.write_json(RESULTS / "candidate_manifest.json", dict(
            provenance=provenance, reused_pilot=pilot.entry(provenance),
            scheduler_sha256=current_inputs["sha256"]["scripts/experiments/common/sweep_scheduler.py"],
            existing_4f1m_winner=hbf_case, cases=[case.entry(provenance) for case in others]))
        scheduler.event("PILOT_REUSED", case=pilot.id, reservation_bytes=scheduler.job_ram)
        results = await run_phase(scheduler, others, runner, provenance)
        baseline = metrics(pilot, pilot_status)
        winner = compare([baseline, *(metrics(case, record) for case, record in results)],
                         original, baseline)
        winner_case = next(case for case in (pilot, *others) if case.id == winner["case"])
        _, _, flow, arch, _, _, _ = compile_case(winner_case.schedule, winner_case.hardware)
        write_ir(winner_case.directory, flow, arch)
        scheduler.event("SWEEP_COMPLETE", peak_ram_utilization=scheduler.peak_utilization,
                        final_reservation_bytes=scheduler.job_ram)
        print(f"Comparison complete. Results in {RESULTS}", flush=True)
    finally:
        scheduler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=ROOT / "build/gtsim_program")
    args = parser.parse_args()
    queue.use_pidfd_child_watcher()
    asyncio.run(run(args.runner.resolve()))


if __name__ == "__main__":
    main()
