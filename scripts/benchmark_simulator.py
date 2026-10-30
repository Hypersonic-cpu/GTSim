#!/usr/bin/env python3
"""Compare host runtime while requiring byte-identical simulator output."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("gemm_1024", "gtsim_gemm", [1024, 1024, 1024], "full_hbm-0F5M"),
    ("gemm_2048", "gtsim_gemm", [2048, 2048, 512], "full_hbm-0F5M"),
    ("fa3_8h_1024", "gtsim_fa3", [1, 8, 8, 1024, 1024], "full_hbm-0F5M"),
    ("fa3_16h_1536", "gtsim_fa3", [1, 16, 16, 1536, 1536], "full_hbm-0F5M"),
    ("gemm_colocated", "gtsim_gemm", [1024, 1024, 1024], "co_located-3F2M"),
    ("fa3_colocated", "gtsim_fa3", [1, 8, 8, 1024, 1024], "co_located-3F2M"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-build-dir", type=Path, required=True)
    parser.add_argument("--optimized-build-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    builds = {"baseline": args.baseline_build_dir.resolve(),
              "optimized": args.optimized_build_dir.resolve()}
    records = []
    for name, executable, dimensions, topology in CASES:
        directory = out / name
        directory.mkdir(exist_ok=True)
        commands = {variant: [str(build / executable),
                              str(ROOT / "configs/v0.1.0/h100-sxm.yaml"),
                              "--memory-config", str(ROOT / f"configs/v0.1.0/memory_{topology}.yaml"),
                              *map(str, dimensions)] for variant, build in builds.items()}
        durations = {variant: [] for variant in builds}
        reference = None
        cycles = None
        # One warm-up per executable, then alternate the timed run order.
        for repeat in range(-1, args.repeats):
            order = ["baseline", "optimized"] if repeat % 2 else ["optimized", "baseline"]
            for variant in order:
                started = time.perf_counter()
                result = subprocess.run(commands[variant], capture_output=True, check=True)
                elapsed = time.perf_counter() - started
                text = result.stdout.decode()
                if "✅ Simulation completed" not in text:
                    raise RuntimeError(f"{name}/{variant} did not complete")
                matches = re.findall(r"^Total cycles:\s*(\d+)\s*$", text, re.MULTILINE)
                if len(matches) != 1:
                    raise RuntimeError(f"{name}/{variant} missing cycle count")
                if reference is None:
                    reference = (result.stdout, result.stderr)
                    cycles = int(matches[0])
                elif reference != (result.stdout, result.stderr):
                    diff = "".join(difflib.unified_diff(reference[0].decode().splitlines(True),
                                                       text.splitlines(True)))
                    (directory / "output-diff.txt").write_text(diff)
                    raise RuntimeError(f"{name}/{variant} output changed; see {directory}")
                label = "warmup" if repeat < 0 else str(repeat + 1)
                (directory / f"{variant}-{label}.log").write_bytes(result.stdout)
                (directory / f"{variant}-{label}.err").write_bytes(result.stderr)
                if repeat >= 0:
                    durations[variant].append(elapsed)
        medians = {variant: statistics.median(samples) for variant, samples in durations.items()}
        record = {"case": name, "executable": executable, "dimensions": dimensions,
                  "memory_profile": topology, "cycles": cycles, "identical_stdout_stderr": True,
                  "stdout_sha256": hashlib.sha256(reference[0]).hexdigest(),
                  "seconds": durations, "median_seconds": medians,
                  "host_speedup": medians["baseline"] / medians["optimized"], "commands": commands}
        records.append(record)
        (directory / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{name}: {cycles} cycles unchanged; {medians['baseline']:.3f}s -> "
              f"{medians['optimized']:.3f}s ({record['host_speedup']:.2f}x)", flush=True)
    binaries = {variant: {name: hashlib.sha256((build / name).read_bytes()).hexdigest()
                           for name in ("gtsim_gemm", "gtsim_fa3")} for variant, build in builds.items()}
    result = {"branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "method": "Whole-process wall time; sequential runs; warmup; alternating order; median",
              "repeats": args.repeats, "binary_sha256": binaries, "cases": records}
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["| Case | Simulated cycles (both) | Before seconds | After seconds | Host speedup |",
             "|---|---:|---:|---:|---:|"]
    for row in records:
        lines.append(f"| {row['case']} | {row['cycles']} | {row['median_seconds']['baseline']:.4f} | "
                     f"{row['median_seconds']['optimized']:.4f} | {row['host_speedup']:.2f}x |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
