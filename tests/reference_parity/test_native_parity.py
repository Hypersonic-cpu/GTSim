#!/usr/bin/env python3
"""Compare independently lowered graphs to separately compiled C++ oracles."""

import argparse
import csv
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "frontend/src"))
from gtsim_frontend.programs.gemm_single import build as gemm
from gtsim_frontend.programs.fa3_single import build as fa3
from gtsim_frontend.passes.instantiate import instantiate
from gtsim_frontend.passes.lower_execution import lower as expand_execution
from gtsim_frontend.passes.lower_hopper_gemm import lower as lower_gemm
from gtsim_frontend.passes.lower_hopper_fa3 import lower as lower_fa3


def compare(executable, kind, dimensions, sms):
    p = gemm(*dimensions) if kind == "gemm" else fa3(*dimensions)
    flow = instantiate(p.launches[0].algorithm, sms)
    kernel = expand_execution((lower_gemm if kind == "gemm" else lower_fa3)(flow))
    raw = subprocess.check_output(
        [str(executable), kind, *map(str, dimensions), str(sms)], text=True
    )
    reference = []
    resources = {}
    tensors = {}
    for line in raw.splitlines():
        f = line.split()
        if f[0] == "cta":
            resources[int(f[1])] = tuple(map(int, f[2:]))
        elif f[0] == "tensor":
            si = f.index("strides")
            tensors[int(f[1])] = (
                f[2],
                int(f[3]),
                int(f[4]),
                tuple(map(int, f[5:si])),
                tuple(map(int, f[si + 1 :])),
            )
        elif f[0] == "op":
            ci, ii = f.index("complete"), f.index("issue")
            mi = f.index("metadata") if "metadata" in f else ci
            if mi < ci:
                assert tuple(map(int, f[mi + 1 : ci])) == (
                    0,
                    0,
                    0,
                    -1,
                    -1,
                    -1,
                    128,
                    4,
                    0,
                    0,
                    -1,
                    -1,
                ), f"instruction metadata differs at {f[2]}"
            reference.append(
                (
                    int(f[1]),
                    f[2],
                    f[3],
                    f[4],
                    *map(int, f[5:mi]),
                    tuple(map(int, f[ci + 1 : ii])),
                    tuple(map(int, f[ii + 1 :])),
                )
            )
    assert resources == {
        c.id: (c.warps, c.shared_bytes, c.register_bytes) for c in kernel.ctas
    }, "CTA resources differ"
    assert tensors == {
        t.id: (t.name, 0 if t.dtype == "fp16" else 3, 0, t.shape, t.strides)
        for t in kernel.tensors
    }, "tensor metadata differs"
    assert len(reference) == len(kernel.ops), (
        kind,
        dimensions,
        "node count",
        len(reference),
        len(kernel.ops),
    )
    sm = {c.id: c.sm for c in kernel.ctas}
    for r, o in zip(reference, kernel.ops):
        candidate = (
            o.id,
            o.name,
            o.pipeline,
            o.op,
            sm[o.cta],
            o.cta,
            o.warp,
            o.tensor,
            0 if o.dtype == "fp16" else 3,
            0,
            o.leading_dim,
            *o.tile,
            o.after_complete,
            o.after_issue,
        )
        assert r == candidate, (
            f"{kind} {dimensions}: first divergence at {o.id}\nreference={r}\ncandidate={candidate}"
        )
    print(kind, dimensions, sms, len(kernel.ops), "nodes: exact graph parity")
    return kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--full-validation-matrix", action="store_true")
    args = parser.parse_args()
    if args.full_validation_matrix:
        root = Path(__file__).resolve().parents[2]
        for kind, fields in (
            ("gemm", ("M", "N", "K")),
            ("fa3", ("batch", "heads_q", "heads_kv", "seq_q", "seq_kv")),
        ):
            with (root / "validation/data" / f"h100_{kind}.csv").open() as source:
                for row in csv.DictReader(source):
                    compare(args.oracle, kind, tuple(int(row[f]) for f in fields), 132)
        return
    for k in (64, 128, 192, 256, 512):
        compare(args.oracle, "gemm", (512, 512, k), 2)
    compare(args.oracle, "gemm", (512, 512, 512), 132)
    for dims in (
        (1, 1, 1, 128, 128),
        (1, 8, 8, 512, 512),
        (1, 16, 16, 1024, 1024),
        (1, 8, 2, 512, 1536),
    ):
        compare(args.oracle, "fa3", dims, 132)


if __name__ == "__main__":
    main()
