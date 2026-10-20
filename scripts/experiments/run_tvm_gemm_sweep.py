#!/usr/bin/env python3
"""Constrained native-TVM H100 GEMM tile sweep through .gtsir and GTSim.

The Phase 3 graph/cycle gate must pass before this launcher accepts candidates.
The original measurement-accuracy failure remains recorded when explicitly
waived. Results are simulated performance, not physical-H100 optimization.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tvm"))
from gemm import ARCH, MEMORY, cycles, generate, hardware_config, run_simulation, source_hashes
from h100_schedule import GemmSchedule
from pinned_tvm import TVM_COMMIT, load_tvm

FIELDS = ("m", "n", "k", "bm", "bn", "bk")
FIXED = {"stages": "2", "warps": "12", "split": "1", "order": "original", "memory": "FullHBM"}


def parse_parameters(raw):
    unknown = set(raw) - set(FIELDS) - set(FIXED)
    if unknown:
        raise ValueError(f"unsupported parameter fields: {sorted(map(str, unknown))}")
    for key, expected in FIXED.items():
        if key in raw and raw[key] != expected:
            raise ValueError(f"{key} is fixed at {expected}")
    try:
        result = {key: int(raw[key]) for key in FIELDS}
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("each row requires integer m,n,k,bm,bn,bk") from error
    return result


def check_gate(path):
    report = json.loads(path.read_text())
    if report.get("tvm_commit") != TVM_COMMIT or not report.get("translation_pass") or len(report.get("rows", [])) != 8:
        raise ValueError("Phase 3 validation report does not prove all eight translations")
    if not all(row.get("exact_graph") and row.get("exact_cycles") for row in report["rows"]):
        raise ValueError("Phase 3 graph/cycle gate has not passed")
    if not report.get("accuracy_pass") and not report.get("measurement_accuracy_waived"):
        raise ValueError("measurement error needs an explicit --allow-measurement-error validation waiver")
    persistent = report.get("persistent_cases", [])
    if not report.get("multiwave_semantics_pass") or not persistent or not all(
            row.get("semantics_pass") and len(row.get("cycle_samples", [])) >= 2
            and len(set(row["cycle_samples"])) == 1 for row in persistent):
        raise ValueError("multi-wave TIR synchronization validation must pass; rerun validate_tvm_gemm.py")
    if report.get("source_sha256") != source_hashes():
        raise ValueError("frontend/core/configuration changed after Phase 3 validation; rerun validate_tvm_gemm.py")
    return report


def simulate(task):
    row, path, evidence, directory, build, repeats = task
    try:
        samples, commands = [], []
        for repeat in range(repeats):
            log = directory / f"simulation-{repeat + 1}.log"
            commands.append(run_simulation(build, ARCH, MEMORY, path, log))
            samples.append(cycles(log.read_text()))
        if len(set(samples)) != 1:
            raise RuntimeError(f"non-reproducible cycles: {samples}")
        return {**row, "status": "complete", "failure_reason": "", "cycles": samples[0], "cycle_samples": samples,
                "nodes": evidence["nodes"], "ir_path": str(path), "ir_sha256": evidence["ir_sha256"],
                "dag_spec_sha256": evidence["dag_spec_sha256"], "tir_sha256": evidence["tir_sha256"],
                "shared_bytes": evidence["shared_bytes"], "register_bytes": evidence["register_bytes"], "waves": evidence["waves"],
                "commands": commands}
    except (OSError, RuntimeError) as error:
        return {**row, "status": "failed", "failure_reason": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=ROOT / "scripts/experiments/tvm_gemm_tiles.csv")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--validation", type=Path, default=ROOT / "validation/results/tvm/automatic/results.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "validation/results/tvm/tile-sweep")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1 or args.repeats < 2:
        parser.error("--jobs must be positive and --repeats must be at least 2")
    gate = check_gate(args.validation)
    build, out = args.build_dir.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = out / "inputs"
    inputs.mkdir(exist_ok=True)
    shutil.copyfile(args.params, inputs / "parameters.csv")
    shutil.copyfile(args.validation, inputs / "phase3-validation.json")
    with args.params.open(newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV requires unique column names")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("parameter CSV is empty")
    hardware = hardware_config(build)
    tvm = load_tvm()
    rows, tasks, baseline_tasks, baseline_errors, seen = [], [], {}, {}, set()
    for index, raw in enumerate(raw_rows, start=1):
        row = {"case": index, "parameters": raw}
        try:
            values = parse_parameters(raw)
            row.update(values)
            dimensions = tuple(values[key] for key in ("m", "n", "k"))
            schedule = GemmSchedule(*(values[key] for key in ("bm", "bn", "bk")))
            schedule.validate(*dimensions, hardware["num_sms"], hardware)
            signature = tuple(values[key] for key in FIELDS)
            if signature in seen:
                raise ValueError("duplicate schedule row")
            seen.add(signature)
            if dimensions not in baseline_tasks and dimensions not in baseline_errors:
                try:
                    directory = out / "baselines" / "x".join(map(str, dimensions))
                    path, evidence = generate(tvm, dimensions, GemmSchedule(), hardware, directory)
                    baseline_tasks[dimensions] = ({"dimensions": dimensions}, path, evidence, directory, build, args.repeats)
                except ValueError as error:
                    baseline_errors[dimensions] = str(error)
            if dimensions in baseline_errors:
                raise ValueError("fixed baseline is infeasible: " + baseline_errors[dimensions])
            directory = out / (f"case-{index:03d}_" + "x".join(map(str, dimensions)) +
                               f"_b{schedule.bm}-{schedule.bn}-{schedule.bk}")
            path, evidence = generate(tvm, dimensions, schedule, hardware, directory)
            tasks.append((row, path, evidence, directory, build, args.repeats))
        except (ValueError, TypeError, OSError) as error:
            rows.append({**row, "status": "rejected", "failure_reason": str(error)})
    # Generation/recognition runs on one thread; independent Simulator processes
    # use the established four-job sweep convention.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        baselines = dict(zip(baseline_tasks, pool.map(simulate, baseline_tasks.values())))
        completed = list(pool.map(simulate, tasks))
    for row in completed:
        dimensions = tuple(row[key] for key in ("m", "n", "k"))
        baseline = baselines[dimensions]
        if row["status"] == "complete" and baseline["status"] == "complete":
            row["baseline_cycles"] = baseline["cycles"]
            row["speedup"] = baseline["cycles"] / row["cycles"]
        elif row["status"] == "complete":
            row.update(status="failed", failure_reason="baseline simulation failed: " + baseline["failure_reason"])
        rows.append(row)
    rows.sort(key=lambda row: row["case"])
    valid = [row for row in rows if row["status"] == "complete"]
    distinct = len({row["dag_spec_sha256"] for row in valid})
    distinct_tir = len({row["tir_sha256"] for row in valid})
    best = {}
    for row in valid:
        dimensions = "x".join(str(row[key]) for key in ("m", "n", "k"))
        if dimensions not in best or row["cycles"] < best[dimensions]["cycles"]:
            best[dimensions] = row
    report = {"tvm_commit": TVM_COMMIT,
        "gtsim_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": source_hashes(), "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameter_sha256": hashlib.sha256(args.params.read_bytes()).hexdigest(),
        "measurement_accuracy_waived": gate["measurement_accuracy_waived"], "measurement_accuracy_pass": gate["accuracy_pass"],
        "semantics_authority": "TVM TIR; bulk.wait_group(0) followed by CTA barrier between output tiles",
        "multiwave_semantics_pass": gate["multiwave_semantics_pass"],
        "fixed": FIXED, "repeats": args.repeats, "distinct_valid_dags": distinct, "distinct_tir": distinct_tir,
        "baselines": {"x".join(map(str, key)): value for key, value in baselines.items()}, "rows": rows, "best": best}
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = ["case", *FIELDS, "status", "cycles", "baseline_cycles", "speedup", "waves", "nodes",
               "ir_path", "ir_sha256", "dag_spec_sha256", "tir_sha256", "failure_reason"]
    with (out / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Case | BM x BN x BK | Cycles | Baseline speedup | Status / reason |", "|---|---|---|---|---|"]
    for row in rows:
        tile = " x ".join(str(row.get(key, "?")) for key in ("bm", "bn", "bk"))
        speedup = f"{row['speedup']:.4f}x" if "speedup" in row else "-"
        lines.append(f"| {row['case']} | {tile} | {row.get('cycles', '-')} | {speedup} | "
                     f"{row['status']}: {row['failure_reason']} |")
    lines += ["", f"Distinct valid DAGs: {distinct}; distinct native TIR kernels: {distinct_tir}; repeats: {args.repeats}."]
    for dimensions, winner in best.items():
        lines.append(f"Best tested simulated configuration for {dimensions}: "
                     f"{winner['bm']}x{winner['bn']}x{winner['bk']}, {winner['cycles']} cycles, {winner['speedup']:.4f}x baseline.")
    lines += ["", "These are reproducible GTSim results, not a claim of optimality or numerical performance on physical H100.",
              "The recorded baseline measurement-accuracy failure was explicitly waived; timing and targets were not changed."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if distinct >= 2 and distinct_tir >= 2 and not any(row["status"] == "failed" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
