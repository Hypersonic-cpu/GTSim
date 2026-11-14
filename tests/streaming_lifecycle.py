#!/usr/bin/env python3
"""Exercise persistent non-prefetch kernel and timeout lifetimes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def command(root: Path, runner: Path, out: Path, max_cycles=None):
    result = [
        sys.executable,
        str(root / "frontend/tools/run_qwen_simulations.py"),
        "--source",
        str(root / "frontend/asm/gemm/reference/two_gemms.progir"),
        "--lib",
        str(root / "frontend/asm/gemm/reference"),
        "--params",
        str(root / "frontend/asm/gemm/reference/params_one_sms.json"),
        "--runner",
        str(runner),
        "--architecture",
        str(root / "configs/v0.1.0/h100-sxm.yaml"),
        "--memory",
        str(root / "configs/memconf/full-hbm.yaml"),
        "--out",
        str(out),
    ]
    if max_cycles is not None:
        result.extend(("--max-cycles", str(max_cycles)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runner = args.build_dir / "gtsim_program"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "frontend/src")

    with tempfile.TemporaryDirectory(prefix="gtsim-stream-lifetime-") as name:
        directory = Path(name)
        normal = subprocess.run(
            command(root, runner, directory / "normal"),
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            timeout=120,
        )
        if normal.returncode:
            raise AssertionError(normal.stdout + normal.stderr)
        timing = json.loads(
            (directory / "normal/result.json").read_text()
        )["timing"]
        if len(timing["per_kernel"]) != 2 or any(
            kernel["state"] != "COMPLETED" for kernel in timing["per_kernel"]
        ):
            raise AssertionError("multi-kernel retirement state is incomplete")

        timeout = subprocess.run(
            command(root, runner, directory / "timeout", max_cycles=1),
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
        )
        output = timeout.stdout + timeout.stderr
        if timeout.returncode == 0 or "kernel execution timeout" not in output:
            raise AssertionError("timeout path did not report kernel timeout")
        if "segmentation fault" in output.lower():
            raise AssertionError("timeout path crashed while cleaning up")


if __name__ == "__main__":
    main()
