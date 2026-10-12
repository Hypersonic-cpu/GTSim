#!/usr/bin/env python3
"""Validate the public H100 GEMM and FA3 builders against measured cycles."""

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from collections import deque
from pathlib import Path


def run_simulator(command: list[str]) -> int:
    """Stream simulator output so large graphs do not retain full logs in memory."""
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    total_cycles = None
    tail: deque[str] = deque(maxlen=20)
    assert process.stdout is not None
    for line in process.stdout:
        tail.append(line)
        match = re.fullmatch(r"Total cycles:\s*(\d+)\s*", line.strip())
        if match:
            total_cycles = int(match.group(1))
    if process.wait() != 0:
        raise RuntimeError(f"simulator exited with status {process.returncode}\n{''.join(tail)}")
    if total_cycles is None:
        raise RuntimeError("simulator output does not contain a total-cycle line")
    return total_cycles


def pearson(xs: list[int], ys: list[int]) -> float:
    if len(xs) < 2:
        return float("nan")
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return numerator / (x_scale * y_scale) if x_scale and y_scale else float("nan")


def load_cases(path: Path, start: int, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        cases = list(csv.DictReader(file))
    cases = cases[start:]
    return cases if limit is None else cases[:limit]


def validate(root: Path, build: Path, name: str, fields: list[str], start: int, limit: int | None,
             verbose: bool) -> dict:
    cases = load_cases(root / "validation" / "data" / f"h100_{name}.csv", start, limit)
    executable = build / f"gtsim_{name}"
    if not executable.is_file():
        raise FileNotFoundError(f"missing executable: {executable}")
    config_file = root / "configs" / "v0.1.0" / "h100-sxm.yaml"

    rows = []
    for index, case in enumerate(cases, start=1):
        command = [str(executable), str(config_file), *(case[field] for field in fields)]
        simulated = run_simulator(command)
        measured = int(case["measured_cycles"])
        error = abs(simulated - measured) / measured * 100.0
        rows.append({**case, "simulated_cycles": simulated, "ape_percent": error})
        if verbose:
            print(f"{name} {index}/{len(cases)}: measured={measured}, simulated={simulated}, APE={error:.2f}%")

    measured_values = [int(row["measured_cycles"]) for row in rows]
    simulated_values = [row["simulated_cycles"] for row in rows]
    return {
        "workload": name,
        "cases": len(rows),
        "mape_percent": sum(row["ape_percent"] for row in rows) / len(rows),
        "pearson_r": pearson(measured_values, simulated_values),
        "rows": rows,
    }


def write_report(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for workload in ("gemm", "fa3"):
        if workload not in report:
            continue
        rows = report[workload].pop("rows")
        if rows:
            with (out_dir / f"h100_{workload}_results.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--workload", choices=("all", "gemm", "fa3"), default="all")
    parser.add_argument("--start-case", type=int, default=0,
                        help="zero-based case offset, useful for split runs")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--out-dir", default="validation/results")
    parser.add_argument("--verbose", action="store_true", help="print each case as it completes")
    args = parser.parse_args()
    if args.start_case < 0:
        parser.error("--start-case must not be negative")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")

    root = Path(__file__).resolve().parents[1]
    build = (root / args.build_dir).resolve()
    selected = ("gemm", "fa3") if args.workload == "all" else (args.workload,)
    specifications = {
        "gemm": ["M", "N", "K"],
        "fa3": ["batch", "heads_q", "heads_kv", "seq_q", "seq_kv"],
    }
    report = {name: validate(root, build, name, specifications[name], args.start_case,
                             args.max_cases, args.verbose)
              for name in selected}
    write_report((root / args.out_dir).resolve(), report)
    for name in selected:
        summary = report[name]
        print(f"{name}: {summary['cases']} cases, MAPE={summary['mape_percent']:.2f}%, "
              f"Pearson r={summary['pearson_r']:.6f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
