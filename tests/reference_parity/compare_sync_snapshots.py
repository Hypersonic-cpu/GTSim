#!/usr/bin/env python3
"""Build old/new GEMM builders against the same current simulator backend."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", default="84b4d81")
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(
        ["git", "show", f"{args.before}:workloads/gemm/gemm_h100.cpp"],
        cwd=root,
        text=True,
    )
    before_sha = hashlib.sha256(source.encode()).hexdigest()
    source = source.replace('"../common/', f'"{root}/workloads/common/').replace(
        "build_gemm_h100", "build_gemm_before"
    )
    with tempfile.TemporaryDirectory() as directory:
        old = Path(directory) / "old.cpp"
        old.write_text(source)
        executable = Path(directory) / "compare"
        build = args.build_dir.resolve()
        command = [
            "c++",
            "-std=c++23",
            "-O2",
            f"-I{root}/include",
            f"-I{root}/3rdparty/glaze/include",
            f"-I{root}/3rdparty/hbfine/include",
            f'-DGTSIM_OLD_SOURCE="{old}"',
            str(root / "tests/reference_parity/compare_sync_snapshots.cpp"),
            str(build / "libgtsim_core.a"),
            str(build / "3rdparty/hbfine/libhbfine.a"),
            "-o",
            str(executable),
        ]
        with (args.out_dir / "build.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        text = subprocess.check_output(
            [
                str(executable),
                str(root / "configs/v0.1.0/h100-sxm.yaml"),
                str(root / "configs/v0.1.0/memory_full_hbm-0F5M.yaml"),
            ],
            text=True,
        )
    (args.out_dir / "simulation.log").write_text(text)
    cases = [
        dict(
            zip(
                ("K", "before_cycles", "after_cycles", "added_completion_edges"),
                map(int, m),
            )
        )
        for m in re.findall(
            r"K=(\d+) old=(\d+) new=(\d+) added_completion_edges=(\d+)", text
        )
    ]
    if len(cases) != 4:
        raise RuntimeError("missing comparison cases")
    report = {
        "before_ref": args.before,
        "before_source_sha256": before_sha,
        "after_source_sha256": hashlib.sha256(
            (root / "workloads/gemm/gemm_h100.cpp").read_bytes()
        ).hexdigest(),
        "shape_M_N": [512, 512],
        "sms": 2,
        "cases": cases,
    }
    (args.out_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
