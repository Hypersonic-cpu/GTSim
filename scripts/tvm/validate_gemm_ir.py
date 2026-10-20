#!/usr/bin/env python3
"""Phase 2 gate: exact reference graph/cycle parity and every measured APE <10%."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

from manual_gemm import manual_gemm
from pinned_tvm import ROOT, TVM_COMMIT


def run(command, log):
    with log.open("w") as output:
        subprocess.run([str(x) for x in command], stdout=output, stderr=subprocess.STDOUT, check=True)
    return log.read_text()


def cycles(text):
    result = re.findall(r"^Total cycles:\s*(\d+)\s*$", text, re.MULTILINE)
    if len(result) != 1:
        raise RuntimeError("missing or ambiguous simulator cycle result")
    return int(result[0])


def validate_case(case, build, out):
    dims = [int(case[x]) for x in ("M", "N", "K")]
    directory = out / ("x".join(str(x) for x in dims))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manual.gtsir"
    manual_gemm(*dims).write(path)
    prefix = [ROOT / "configs/v0.1.0/h100-sxm.yaml", "--memory-config", ROOT / "configs/memconf/full-hbm.yaml"]
    commands = [[build / "tile_ir_parity_tests", path, *dims],
                [build / "gtsim_gemm", *prefix, *dims],
                [build / "gtsim_gemm_ir", *prefix, path]]
    parity = run(commands[0], directory / "parity.log")
    reference = cycles(run(commands[1], directory / "reference.log"))
    simulated = cycles(run(commands[2], directory / "ir.log"))
    measured = int(case["measured_cycles"])
    error = abs(simulated - measured) / measured
    return {**{key: int(value) for key, value in case.items()}, "reference_cycles": reference,
            "ir_cycles": simulated, "ape_percent": error * 100,
            "exact_cycles": reference == simulated, "accuracy_pass": error < 0.10,
            "graph_parity": True, "graph_summary": parity.strip(),
            "ir_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "ir_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "commands": [[str(x) for x in command] for command in commands]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "validation/results/tvm/manual")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (ROOT / "validation/data/h100_gemm.csv").open() as file:
        cases = list(csv.DictReader(file))
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(lambda case: validate_case(case, args.build_dir.resolve(), args.out_dir.resolve()), cases))
    passed = all(row["exact_cycles"] and row["accuracy_pass"] for row in rows)
    report = {"phase": 2, "passed": passed, "tvm_commit": TVM_COMMIT,
              "gtsim_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "rows": rows}
    # HEAD alone cannot reproduce an uncommitted implementation. Preserve the
    # exact frontend, reference, core and configuration hashes with the results.
    patterns = ("CMakeLists.txt", "include/*.h", "src/*.cpp", "workloads/gemm/*.cpp", "workloads/common/*.h",
                "scripts/tvm/*.py", "tests/tile_ir*.cpp", "configs/v0.1.0/h100-sxm.yaml", "configs/memconf/full-hbm.yaml")
    files = sorted({path for pattern in patterns for path in ROOT.glob(pattern)})
    report["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in files}
    (args.out_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["| M | N | K | Reference | IR | Measured | Error |", "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['M']} | {row['N']} | {row['K']} | {row['reference_cycles']} | {row['ir_cycles']} | "
                     f"{row['measured_cycles']} | {row['ape_percent']:.3f}% |")
    lines += ["", f"Phase 2 gate: {'PASS' if passed else 'FAIL; stop before Phase 3'}. "
              "Graph comparison includes per-node operations, access regions, resources and both edge kinds."]
    text = "\n".join(lines) + "\n"
    (args.out_dir / "report.md").write_text(text)
    print(text)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
