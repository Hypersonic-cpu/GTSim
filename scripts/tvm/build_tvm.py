#!/usr/bin/env python3
"""Build the pinned compiler separately and record source/binary provenance."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from pinned_tvm import ROOT, TVM_COMMIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-tvm")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    source = ROOT / "3rdparty/tvm"
    build = args.build_dir.resolve()
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != TVM_COMMIT:
        raise RuntimeError(f"expected pinned TVM {TVM_COMMIT}, got {commit}")
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                    "-DUSE_LLVM=OFF", "-DUSE_CUDA=OFF", "-DUSE_Z3=OFF", "-DUSE_RPC=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", str(args.jobs)], check=True)
    libraries = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (build / "lib").iterdir()
                 if p.name.startswith(("libtvm_compiler.", "libtvm_runtime.")) and p.is_file()}
    if len(libraries) != 2:
        raise RuntimeError("expected compiler and runtime shared libraries")
    (build / "gtsim-tvm-build.json").write_text(json.dumps({"tvm_commit": commit,
        "source": str(source), "libraries": libraries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
