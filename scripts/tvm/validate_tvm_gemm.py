#!/usr/bin/env python3
"""Phase 3 translation gate; measurement accuracy is reported independently."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from gemm import ARCH, MEMORY, cycles, generate, hardware_config, run_simulation, source_hashes
from h100_schedule import GemmSchedule
from manual_gemm import manual_gemm  # Test oracle only; production lowerer never imports this.
from pinned_tvm import ROOT, TVM_COMMIT, load_tvm


def validate_case(case, build, directory, generated_path, evidence):
    dimensions = [int(case[key]) for key in ("M", "N", "K")]
    manual_path = directory / "manual.gtsir"
    manual_gemm(*dimensions).write(manual_path)
    if generated_path.read_bytes() != manual_path.read_bytes():
        raise RuntimeError(f"TVM/manual IR differs for {dimensions}")
    oracle_command = [str(build / "tile_ir_parity_tests"), str(generated_path), *map(str, dimensions)]
    with (directory / "graph-parity.log").open("w") as log:
        subprocess.run(oracle_command, stdout=log, stderr=subprocess.STDOUT, check=True)
    auto_command = run_simulation(build, ARCH, MEMORY, generated_path, directory / "tvm.log")
    manual_command = run_simulation(build, ARCH, MEMORY, manual_path, directory / "manual.log")
    reference_command = [str(build / "gtsim_gemm"), str(ARCH), "--memory-config", str(MEMORY), *map(str, dimensions)]
    with (directory / "reference.log").open("w") as log:
        subprocess.run(reference_command, stdout=log, stderr=subprocess.STDOUT, check=True)
    counts = {name: cycles((directory / f"{name}.log").read_text()) for name in ("tvm", "manual", "reference")}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"cycle mismatch for {dimensions}: {counts}")
    measured = int(case["measured_cycles"])
    error = abs(counts["tvm"] - measured) / measured * 100
    return {"dimensions": dimensions, "measured_cycles": measured, "cycles": counts,
            "ape_percent": error, "accuracy_pass": error < 10, "exact_graph": True, "exact_cycles": True,
            "ir_sha256": evidence["ir_sha256"], "ir_path": str(generated_path),
            "commands": [oracle_command, auto_command, manual_command, reference_command]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "validation/results/tvm/automatic")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--allow-measurement-error", action="store_true",
                        help="record an explicit waiver of the Phase 2 measurement accuracy gate")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    build, out = args.build_dir.resolve(), args.out_dir.resolve()
    hardware = hardware_config(build)
    tvm = load_tvm()
    with (ROOT / "validation/data/h100_gemm.csv").open() as file:
        cases = list(csv.DictReader(file))
    tasks = []
    for case in cases:
        dimensions = [int(case[key]) for key in ("M", "N", "K")]
        directory = out / "x".join(map(str, dimensions))
        path, evidence = generate(tvm, dimensions, GemmSchedule(), hardware, directory)
        tasks.append((case, build, directory, path, evidence))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(lambda task: validate_case(*task), tasks))
    accuracy_pass = all(row["accuracy_pass"] for row in rows)
    report = {"phase": 3, "translation_pass": True, "accuracy_pass": accuracy_pass,
              "measurement_accuracy_waived": args.allow_measurement_error, "tvm_commit": TVM_COMMIT,
              "gtsim_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": source_hashes(), "rows": rows,
              "scope": "one persistent CTA wave; two stages; 12 warps; split=1; original order; FullHBM"}
    (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["| M x N x K | TVM / manual / builder cycles | Measured | Error |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {' x '.join(map(str, row['dimensions']))} | {row['cycles']['tvm']} / "
                     f"{row['cycles']['manual']} / {row['cycles']['reference']} | {row['measured_cycles']} | {row['ape_percent']:.3f}% |")
    lines += ["", "Exact graph and cycle parity: PASS for every case.",
              f"Measurement accuracy: {'PASS' if accuracy_pass else 'FAIL'}; waiver: {args.allow_measurement_error}."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if accuracy_pass or args.allow_measurement_error else 2


if __name__ == "__main__":
    raise SystemExit(main())
