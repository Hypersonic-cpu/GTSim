#!/usr/bin/env python3
"""Inspect actual pinned Hopper primitives through TVM's lowering stages.

The symbolic descriptor inputs make this a representation probe, not an
executable GEMM kernel or a claim of numerical/CUDA correctness.
"""

import argparse
import json
from pathlib import Path

from pinned_tvm import ROOT, load_tvm


def inspect_hopper(tvm, out):
    from tvm.backend.cuda.op import cuda_mbarrier_wait
    from tvm.backend.cuda.ptx import PTXNamespace

    t = tvm.tirx
    ptx = PTXNamespace()
    smem = t.decl_buffer((64, 64), "float16", scope="shared", name="S")
    bar = t.decl_buffer((1,), "uint64", scope="shared", name="bar")
    acc = t.decl_buffer((32,), "float32", scope="local", name="acc")
    descriptor = t.Var("tensor_map", "handle")
    da, db = t.Var("desc_a", "uint64"), t.Var("desc_b", "uint64")
    bx, tx = t.Var("bx", "int32"), t.Var("tx", "int32")
    load = ptx["cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"](
        smem.data, descriptor, 0, 0, t.address_of(bar[0]))
    mma = ptx["wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16"](
        *[acc[i] for i in range(32)], da, db, False, 1, 1, 0, 0)
    wait = cuda_mbarrier_wait(t.address_of(bar[0]), 0)
    statements = [
        t.IfThenElse(tx == 256, t.SeqStmt([
            t.Evaluate(load),
            t.Evaluate(ptx.mbarrier.arrive.expect_tx.shared.b64(t.address_of(bar[0]), t.const(8192, "uint32"))),
        ]), None),
        t.IfThenElse(tx < 128, t.SeqStmt([
            t.Evaluate(wait), t.Evaluate(ptx.wgmma.fence.sync.aligned()), t.Evaluate(mma),
            t.Evaluate(ptx.wgmma.commit_group.sync.aligned()),
            t.Evaluate(ptx.wgmma.wait_group.sync.aligned(0)),
        ]), None),
    ]
    body = t.For(tx, 0, 384, t.ForKind.THREAD_BINDING, t.SeqStmt(statements),
                 thread_binding=t.IterVar((0, 384), tx, t.IterVar.ThreadIndex, "threadIdx.x"))
    body = t.For(bx, 0, 1, t.ForKind.THREAD_BINDING, body,
                 thread_binding=t.IterVar((0, 1), bx, t.IterVar.ThreadIndex, "blockIdx.x"))
    body = t.SeqStmt([t.AllocBuffer(smem), t.AllocBuffer(bar), t.AllocBuffer(acc), body])
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a", "max_num_threads": 1024,
                                "thread_warp_size": 32})
    func = t.PrimFunc([descriptor, da, db], body).with_attr("target", target)
    mod = tvm.IRModule({"probe": func})
    out.mkdir(parents=True, exist_ok=True)
    stages = {"scheduled": mod}
    stages["simplified"] = t.transform.StmtSimplify()(mod)
    stages["flattened"] = t.transform.FlattenBuffer()(stages["simplified"])
    stages["intrinsics"] = t.transform.LowerIntrin()(stages["flattened"])
    report = {"extraction": "scheduled TIR before FlattenBuffer/LowerIntrin",
              "explicit_ops": [load.op.name, mma.op.name, wait.op.name], "stages": {}}
    for name, stage in stages.items():
        text = stage.script()
        (out / f"{name}.py").write_text(text + "\n")
        report["stages"][name] = {"thread_binding": "thread_binding" in text,
                                  "ptx": "T.ptx" in text, "mbarrier_wait": "mbarrier_wait" in text}
    (out / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "validation/results/tvm/inspection")
    args = parser.parse_args()
    print(json.dumps(inspect_hopper(load_tvm(), args.out_dir), indent=2))


if __name__ == "__main__":
    main()
