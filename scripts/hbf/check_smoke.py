#!/usr/bin/env python3
"""Exercise both public builders through the real HBFine routing and drain path."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    for topology in ("3F2M", "1F4M"):
        for workload, placement, shape in (("gemm", "gemm_b_hbf", [128, 256, 64]),
                                           ("fa3", "fa3_q_hbf", [1, 1, 1, 128, 128])):
            command = [str(args.build_dir / f"gtsim_{workload}"), str(ROOT / "configs/v0.1.0/h100-sxm.yaml"),
                       "--memory-config", str(ROOT / f"configs/memconf/{topology}_bank_plane.yaml"),
                       "--memory-config", str(ROOT / f"configs/memconf/placement_{placement}.yaml"),
                       *map(str, shape)]
            text = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
            assert "Simulation completed at cycle" in text and "Memory drained at cycle:" in text
            stats = json.loads(re.search(r"^HBF detailed JSON: (.*)$", text, re.M)[1])
            assert stats["host_read_bytes"] > 0 and stats["senses"] > 0
            assert stats["accepted_requests"] == stats["completed_requests"]
            assert stats["host_write_bytes"] == stats["failed_requests"] == stats["programs"] == stats["erases"] == 0
            assert stats["host_read_bytes"] == sum(stats["channel_read_bytes"])
            cycles = re.search(r"^Total cycles: (\d+)$", text, re.M)[1]
            print(f"{topology} {workload}: {cycles} cycles, {stats['host_read_bytes']} HBF bytes; drained and conserved")


if __name__ == "__main__":
    main()
