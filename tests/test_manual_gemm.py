#!/usr/bin/env python3
"""Exercise persistent waves and ping-pong boundary cases against the oracle."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/tvm"))
from manual_gemm import manual_gemm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    args = parser.parse_args()
    fixture = manual_gemm(128, 256, 128).text()
    if fixture != (ROOT / "tests/fixtures/h100_gemm.gtsir").read_text():
        raise RuntimeError("standalone manual fixture drifted from its specification")
    # Two SMs force persistent waves; odd/even K stages exercise cross-wave
    # ping-pong reuse. Ten N tiles exercise the clamped final column group.
    cases = [(256, 768, 64, 2), (256, 768, 128, 2), (256, 768, 192, 2), (128, 2560, 64, 2)]
    with tempfile.TemporaryDirectory() as directory:
        for m, n, k, sms in cases:
            path = Path(directory) / "manual.gtsir"
            manual_gemm(m, n, k, sms).write(path)
            subprocess.run([str(args.oracle.resolve()), str(path), str(m), str(n), str(k), str(sms)], check=True)
    for dimensions in ((0, 256, 64), (128, 128, 64), (128, 256, 63)):
        try:
            manual_gemm(*dimensions)
        except ValueError:
            continue
        raise RuntimeError(f"accepted unsupported dimensions: {dimensions}")


if __name__ == "__main__":
    main()
