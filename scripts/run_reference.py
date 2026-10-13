#!/usr/bin/env python3
"""Run and verify deterministic GTSim v0.1.0 reference cases."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def total_cycles(output: str) -> int:
    matches = re.findall(r"^Total cycles:\s*(\d+)\s*$", output, re.MULTILINE)
    if not matches:
        raise RuntimeError("simulator output does not contain a total-cycle line")
    return int(matches[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "configs/v0.1.0/references.json").read_text())
    config_file = root / "configs/v0.1.0/h100-sxm.yaml"
    memory_config_file = root / "configs/v0.1.0/memory_full_hbm-0F5M.yaml"
    build = (root / args.build_dir).resolve()
    failures = []

    for item in manifest["references"]:
        if item["status"] != "release_reproduced":
            continue
        executable = build / ("gtsim_gemm" if "gemm" in item["id"] else "gtsim_fa3")
        run = subprocess.run([str(executable), str(config_file),
                              "--memory-config", str(memory_config_file),
                              *item["arguments"]], text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        if run.returncode != 0:
            failures.append(f"{item['id']}: exit status {run.returncode}")
            continue
        actual = total_cycles(run.stdout)
        expected = item["expected_total_cycles"]
        print(f"{item['id']}: {actual} cycles (expected {expected})")
        if args.verify and actual != expected:
            failures.append(f"{item['id']}: expected {expected}, got {actual}")

    if failures:
        print("Reference verification failed:", *failures, sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
