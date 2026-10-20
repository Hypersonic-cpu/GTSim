#!/usr/bin/env python3
"""Prove TVM TIR -> Tile IR -> existing Simulator, without claiming GEMM lowering."""

import argparse
import json
from pathlib import Path
import subprocess

from pinned_tvm import ROOT, TVM_COMMIT, load_tvm
from inspect_ir import inspect_hopper
from tile_ir import Operation, TileIR


def copy_program(tvm):
    t = tvm.tirx
    a = t.decl_buffer((1, 32), "float16", name="A")
    b = t.decl_buffer((1, 32), "float16", name="B")
    bx, tx = t.Var("bx", "int32"), t.Var("tx", "int32")
    body = t.BufferStore(b, t.BufferLoad(a, [0, tx]), [0, tx])
    body = t.For(tx, 0, 32, t.ForKind.THREAD_BINDING, body,
                 thread_binding=t.IterVar((0, 32), tx, t.IterVar.ThreadIndex, "threadIdx.x"))
    body = t.For(bx, 0, 1, t.ForKind.THREAD_BINDING, body,
                 thread_binding=t.IterVar((0, 1), bx, t.IterVar.ThreadIndex, "blockIdx.x"))
    return t.PrimFunc([a, b], body).with_attr("global_symbol", "copy")


def lower_copy(tvm, func):
    """Recognize exactly one FP16 row copy; reject all other statements/values.

    Consecutive lanes form a warp access region. A BufferLoad remains an ordinary
    register load. Only the store's explicit data flow creates a completion edge.
    """
    t = tvm.tirx
    bx = func.body
    if not isinstance(bx, t.For) or bx.thread_binding is None or str(bx.thread_binding.thread_tag) != "blockIdx.x":
        raise ValueError("copy frontend expects a blockIdx.x loop")
    tx = bx.body
    if not isinstance(tx, t.For) or tx.thread_binding is None or str(tx.thread_binding.thread_tag) != "threadIdx.x":
        raise ValueError("copy frontend expects a threadIdx.x loop")
    if int(bx.min) != 0 or int(bx.extent) != 1 or int(tx.min) != 0 or int(tx.extent) != 32:
        raise ValueError("copy frontend supports exactly one block and one active warp")
    if len(func.params) != 2 or not all(t.is_buffer_var(p) for p in func.params):
        raise ValueError("copy frontend requires two explicit buffers")
    a, b = func.params
    for buffer in (a, b):
        if tuple(int(x) for x in buffer.shape) != (1, 32) or str(buffer.dtype) != "float16":
            raise ValueError("copy frontend requires 1x32 FP16 buffers")
        if buffer.scope() != "global" or buffer.strides or int(buffer.elem_offset) != 0:
            raise ValueError("copy frontend requires compact global buffers at offset zero")
    expected = t.BufferStore(b, t.BufferLoad(a, [0, tx.loop_var]), [0, tx.loop_var])
    try:
        tvm.ir.assert_structural_equal(tx.body, expected)
    except ValueError as error:
        raise ValueError("unsupported TIR: expected B[0,tx] = A[0,tx] without arithmetic or side effects") from error
    ir = TileIR()
    ir.tensor(0, "A", (1, 32), (32, 1))
    ir.tensor(1, "B", (1, 32), (32, 1))
    ir.cta(0, 0, 0, 12 * 32 * 4)
    load = ir.add(Operation("load", "ld.dram2reg", 0, 0, 0, (0, 0, 1, 32)))
    ir.add(Operation("store", "st.reg2dram", 0, 0, 1, (0, 0, 1, 32), after_complete=[load]))
    return ir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "validation/results/tvm/smoke")
    args = parser.parse_args()
    tvm = load_tvm()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inspect_hopper(tvm, out / "hopper-inspection")
    func = copy_program(tvm)
    mod = tvm.IRModule({"copy": func})
    (out / "scheduled.py").write_text(mod.script())
    # Inspect the real pinned passes, recording where rank/thread information
    # changes. Extraction uses the scheduled program before these lossy passes.
    simplified = tvm.tirx.transform.StmtSimplify()(mod)
    flattened = tvm.tirx.transform.FlattenBuffer()(simplified)
    (out / "simplified.py").write_text(simplified.script())
    (out / "flattened.py").write_text(flattened.script())
    path = out / "copy.gtsir"
    lower_copy(tvm, func).write(path)
    # Prove the unsupported-semantics path is executable as well.
    a, b = func.params
    bx, tx_loop = func.body, func.body.body
    tx = tx_loop.loop_var
    bad_store = tvm.tirx.BufferStore(b, tvm.tirx.BufferLoad(a, [0, tx]) + 1, [0, tx])
    bad_thread = tvm.tirx.For(tx, tx_loop.min, tx_loop.extent, tx_loop.kind, bad_store,
                            thread_binding=tx_loop.thread_binding)
    bad = func.with_body(tvm.tirx.For(bx.loop_var, bx.min, bx.extent, bx.kind, bad_thread,
                                     thread_binding=bx.thread_binding))
    try:
        lower_copy(tvm, bad)
    except ValueError:
        pass
    else:
        raise RuntimeError("copy frontend accepted unsupported arithmetic")
    command = [str(args.build_dir.resolve() / "gtsim_gemm_ir"),
               str(ROOT / "configs/v0.1.0/h100-sxm.yaml"), "--memory-config",
               str(ROOT / "configs/memconf/full-hbm.yaml"), str(path)]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "simulation.log").write_text(result.stdout)
    (out / "provenance.json").write_text(json.dumps({"tvm_commit": TVM_COMMIT, "command": command,
        "extraction": "scheduled TIR before FlattenBuffer/LowerIntrin", "returncode": result.returncode}, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(result.stdout)
    print(result.stdout)
    print(f"TVM smoke artifacts: {out}")


if __name__ == "__main__":
    main()
