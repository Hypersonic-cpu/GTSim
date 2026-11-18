#!/usr/bin/env python3
"""Run the bounded two-phase Experiment 002 sweep."""

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
from itertools import product
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys

if __package__:
    from .common import sweep_scheduler as queue
    from .report_q_projection import write_reports
else:
    from common import sweep_scheduler as queue
    from report_q_projection import write_reports


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "validation/results/002-tiling-order-Qmat"
FRONTEND = ROOT / "frontend"
sys.path[:0] = [str(FRONTEND / folder) for folder in ("src", "tools")]
from run_q_projection import ARCHITECTURE, MEMORIES, SOURCE, compile_case, write_ir
from compile.opt.q_projection import BM, ORDERINGS, QSchedule


DRIVER = Path("frontend/tools/run_q_projection.py")
SPLITS = (1, 2, 4, 8)


@dataclass(frozen=True)
class Candidate:
    phase: str
    schedule: QSchedule
    hardware: str = "4f1m"

    @property
    def id(self):
        p = self.schedule
        return (f"bn{p.bn}-bk{p.bk}-sk{p.split_k}-{p.ordering}"
                f"-pf{p.pf_bn}x{p.pf_bk}-a{p.ahead}")

    @property
    def directory(self):
        return OUTPUT / self.phase if self.phase == "pilot" else OUTPUT / self.phase / self.id

    def signature(self, provenance):
        return queue.fingerprint(dict(parameters=asdict(self.schedule), hardware=self.hardware,
                                      inputs=provenance))

    def job(self, runner, provenance, *, preflight=False):
        p = self.schedule
        identity = self.signature(provenance)
        command = (
            sys.executable, str(DRIVER), "--runner", str(runner),
            "--bn", str(p.bn), "--bk", str(p.bk), "--split-k", str(p.split_k),
            "--ordering", p.ordering, "--pf-bn", str(p.pf_bn), "--pf-bk", str(p.pf_bk),
            "--ahead", str(p.ahead), "--hardware", self.hardware,
            "--fingerprint", identity,
        )
        output = self.directory
        if preflight:
            output = OUTPUT / "preflight" / self.phase / self.id
            command += ("--preflight",)
            identity = queue.fingerprint(dict(compile_only=True, fingerprint=identity))
        elif self.phase == "pilot":
            command += ("--save-ir",)
        return queue.Job(self.id, identity, (*command, "--out", str(output)), output)

    def entry(self, provenance):
        return dict(id=self.id, phase=self.phase, hardware=self.hardware,
                    parameters=dict(bm=BM, **asdict(self.schedule)),
                    fingerprint=self.signature(provenance),
                    directory=str(self.directory.relative_to(OUTPUT)))


def phase_a():
    cases, duplicates = [], []
    for bn, bk, split_k in product((128, 256), (32, 64, 128), SPLITS):
        for ordering in ORDERINGS:
            p = QSchedule(bn, bk, split_k, ordering)
            for schedule in (p, replace(p, pf_bn=1024, pf_bk=1024, ahead=2)):
                candidate = Candidate("phase-a", schedule)
                if split_k == 1 and ordering != "n_major":
                    parent = Candidate("phase-a", replace(schedule, ordering="n_major"))
                    duplicates.append(dict(case=candidate.id, reuse=parent.id,
                                           reason="Split-K=1 ordering equivalence"))
                else:
                    cases.append(candidate)
    assert len(cases) == 228 and len({case.id for case in cases}) == 228
    return cases, duplicates


def phase_b(results):
    scores = {}
    for case, record in results:
        if record["status"] != "SUCCESS":
            continue
        compute = replace(case.schedule, pf_bn=0, pf_bk=0, ahead=0)
        cycle_count = queue.read_json(case.directory / "result.json")["timing"]["total_cycles"]
        scores[compute] = min(cycle_count, scores.get(compute, cycle_count))
    selected, shortfall = [], {}
    for split in SPLITS:
        ordered = sorted((p for p in scores if p.split_k == split),
                         key=lambda p: (scores[p], p.bn, p.bk, p.ordering))
        if ordered:
            selected.append(ordered[0])
        if len(ordered) > 1:
            selected.append(next((p for p in ordered[1:] if p.ordering != ordered[0].ordering),
                                 ordered[1]))
        if len(ordered) < 2:
            shortfall[str(split)] = 2 - len(ordered)
    cases, duplicates = [], []
    for p in selected:
        for (bn, bk), ahead in product(((256, 128), (1024, 1024), (8192, 4096)), (1, 2, 4)):
            schedule = replace(p, pf_bn=bn, pf_bk=bk, ahead=ahead)
            candidate = Candidate("phase-b", schedule)
            if bn == 1024 and ahead == 2:
                duplicates.append(dict(case=candidate.id, reuse=f"phase-a/{candidate.id}",
                                       reason="already measured in Phase A"))
            elif bn == 8192 and ahead != 1:
                winner = Candidate("phase-b", replace(schedule, ahead=1))
                duplicates.append(dict(case=candidate.id, reuse=f"phase-b/{winner.id}",
                                       reason="whole weight tensor is one region"))
            else:
                cases.append(candidate)
    assert len(cases) == 6 * len(selected) <= 48
    return cases, duplicates, selected, shortfall


def input_manifest(runner):
    tracked = [Path(__file__).resolve(), ROOT / DRIVER, ROOT / ARCHITECTURE, ROOT / SOURCE,
               *(ROOT / path for path in MEMORIES.values()), runner,
               *(FRONTEND / "src").rglob("*.py"), *(FRONTEND / "lib").glob("q_*ir"),
               FRONTEND / "tools/run_qwen_simulations.py",
               *Path(queue.__file__).parent.glob("*.py")]
    cache = runner.parent / "CMakeCache.txt"
    config = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.startswith(("CMAKE_BUILD_TYPE:", "CMAKE_CXX_COMPILER:",
                                "CMAKE_CXX_FLAGS", "GTSIM_", "BUILD_TESTING:")):
                name, _, value = line.partition("=")
                config[name] = value
    hashes = {}
    for path in sorted(set(tracked)):
        name = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
        with path.open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "source_commit": subprocess.check_output(("git", "rev-parse", "HEAD"),
                                                  cwd=ROOT, text=True).strip(),
        "build_configuration": config,
        "python_version": sys.version,
        "sha256": hashes,
    }


def same_simulation_inputs(current, recorded):
    """Scheduling changes preserve completed simulations and their fingerprints."""
    orchestration = {
        "scripts/experiments/common/sweep_scheduler.py",
        "scripts/experiments/sweep_q_projection.py",
    }

    def execution_inputs(manifest):
        return dict(manifest, sha256={path: value for path, value in
                    manifest.get("sha256", {}).items() if path not in orchestration})

    return execution_inputs(current) == execution_inputs(recorded)


def result_present(job):
    payload = queue.read_json(job.output / "result.json")
    provenance = queue.read_json(job.output / "provenance.json")
    return (payload.get("timing", {}).get("total_cycles", 0) > 0
            and provenance.get("candidate_fingerprint") == job.fingerprint)


async def run_phase(scheduler, cases, runner, provenance):
    if not cases:
        return []
    checks = [case.job(runner, provenance, preflight=True) for case in cases]
    check_results = await scheduler.run(
        f"{cases[0].phase} preflight", checks, ramp=False,
        validate_resume=lambda job: queue.read_json(
            job.output / "compile_summary.json").get("status") == "SUCCESS")
    ready, results = [], {}
    for case, check in zip(cases, checks):
        record = check_results[check.id]
        if record["status"] == "SUCCESS":
            ready.append(case.job(runner, provenance))
            continue
        compile_summary = queue.read_json(check.output / "compile_summary.json")
        case.directory.mkdir(parents=True, exist_ok=True)
        if compile_summary:
            queue.write_json(case.directory / "compile_summary.json", compile_summary)
        if (check.output / "driver.log").is_file():
            shutil.copyfile(check.output / "driver.log", case.directory / "driver.log")
        results[case.id] = dict(record, stage="preflight", fingerprint=case.signature(provenance),
                                error=compile_summary.get("reason", record["error"]))
    results.update(await scheduler.run(f"{cases[0].phase} simulation", ready, ramp=True,
                                       validate_resume=result_present))
    return [(case, results[case.id]) for case in cases]


async def run(runner):
    os.chdir(ROOT)
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(FRONTEND / "src"), os.environ.get("PYTHONPATH"))))
    scheduler = queue.SweepScheduler(OUTPUT, ROOT)
    try:
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signum, task.cancel)
        current_inputs = input_manifest(runner)
        saved_inputs = queue.read_json(OUTPUT / "candidate_manifest.json").get("provenance", {})
        provenance = saved_inputs if same_simulation_inputs(current_inputs, saved_inputs) else current_inputs
        pilot = Candidate("pilot", QSchedule(256, 64, 1, "n_major"), "fullhbm")
        a_cases, duplicates = phase_a()
        manifest = dict(provenance=provenance, pilot=pilot.entry(provenance),
                        scheduler_sha256=current_inputs["sha256"][
                            "scripts/experiments/common/sweep_scheduler.py"],
                        phase_a=[c.entry(provenance) for c in a_cases],
                        phase_a_deduplicated=duplicates, phase_b_maximum=48,
                        total_simulations_maximum=277,
                        hardware_configs={key: str(path) for key, path in MEMORIES.items()})
        queue.write_json(OUTPUT / "candidate_manifest.json", manifest)
        print("Full-HBM pilot/reference scheduled.", flush=True)
        reference = await scheduler.pilot(pilot.job(runner, provenance), validate_resume=result_present)
        initial_reservation = scheduler.job_ram
        capacity = scheduler.memory()
        estimated_memory_slots = max(0, int(min(
            (limit.capacity * queue.RAM_ADMISSION_TARGET - limit.used) / initial_reservation
            for limit in capacity)))
        effective_limit = min(scheduler.limit // scheduler.default_cpu_slots, estimated_memory_slots)
        scheduler.event("SWEEP_ADMISSION", initial_concurrency_limit=effective_limit,
                        memory_slots=estimated_memory_slots, cpu_budget=scheduler.cpu_budget)
        phase_a_results = await run_phase(scheduler, a_cases, runner, provenance)
        b_cases, duplicates, selected, shortfall = phase_b(phase_a_results)
        manifest.update(phase_b=[c.entry(provenance) for c in b_cases],
                        phase_b_deduplicated=duplicates, phase_b_selected=[asdict(p) for p in selected],
                        phase_b_shortfall=shortfall)
        queue.write_json(OUTPUT / "candidate_manifest.json", manifest)
        phase_b_results = await run_phase(scheduler, b_cases, runner, provenance)
        cases = phase_a_results + phase_b_results
        resource_summary = {
            "pilot": reference["resources"], "initial_job_ram_bytes": initial_reservation,
            "final_job_ram_bytes": scheduler.job_ram, "effective_cpu_budget": scheduler.cpu_budget,
            "cpu_slots_per_simulation": scheduler.default_cpu_slots,
            "initial_memory_limited_concurrency": effective_limit,
            "ram_hard_limit": queue.RAM_HARD_LIMIT, "admission_target": queue.RAM_ADMISSION_TARGET,
            "recovery_target": queue.RAM_RECOVERY_TARGET,
            "memory_constraints_after_pilot": [asdict(limit) for limit in capacity],
            "peak_system_ram_utilization": scheduler.peak_utilization,
            "kernel_ceiling_enabled": scheduler.cgroup is not None,
            "final_concurrency_limit": scheduler.limit,
            "statuses": Counter(record["status"] for _, record in cases),
        }
        queue.write_json(OUTPUT / "resource_summary.json", resource_summary)
        best = write_reports(OUTPUT, pilot, reference, cases, shortfall)
        for case in best:
            _, _, flow, arch, _, _, _ = compile_case(case.schedule, case.hardware)
            write_ir(case.directory, flow, arch)
        scheduler.event("SWEEP_COMPLETE", statuses=resource_summary["statuses"])
        print(f"Experiment complete. Results in {OUTPUT}", flush=True)
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
