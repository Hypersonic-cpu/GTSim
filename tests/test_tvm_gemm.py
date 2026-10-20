#!/usr/bin/env python3
"""Focused native-TVM boundary, parity and rejection checks (opt-in CTest)."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/tvm"))
from gemm import generate, hardware_config
from h100_schedule import GemmSchedule, scheduled_gemm
from lower_gemm import lower_gemm
from manual_gemm import manual_gemm
from pinned_tvm import load_tvm


def replace_calls(tvm, stmt, operation):
    t = tvm.tirx
    if isinstance(stmt, t.SeqStmt):
        return t.SeqStmt([replace_calls(tvm, child, operation) for child in stmt.seq])
    if isinstance(stmt, t.For):
        return t.For(stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
                     replace_calls(tvm, stmt.body, operation), thread_binding=stmt.thread_binding,
                     annotations=stmt.annotations, step=stmt.step)
    if isinstance(stmt, t.IfThenElse):
        return t.IfThenElse(stmt.condition, replace_calls(tvm, stmt.then_case, operation),
                            None if stmt.else_case is None else replace_calls(tvm, stmt.else_case, operation))
    if isinstance(stmt, t.Evaluate) and isinstance(stmt.value, tvm.ir.Call):
        if isinstance(stmt.value.op, tvm.ir.Op) and stmt.value.op.name == operation:
            return t.Evaluate(0)
    return stmt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    tvm = load_tvm()
    hardware = hardware_config(args.build_dir)
    with tempfile.TemporaryDirectory() as directory:
        path, _ = generate(tvm, (128, 256, 128), GemmSchedule(), hardware, directory)
        if path.read_text() != manual_gemm(128, 256, 128).text():
            raise RuntimeError("automatic/manual fixture mismatch")
        subprocess.run([str(args.build_dir.resolve() / "tile_ir_parity_tests"), str(path), "128", "256", "128"], check=True)
        # Shape/stride/intrinsic axes change the actual TIR and DAG. These
        # checks use the untouched builder with explicit legal tile options.
        for schedule in (GemmSchedule(64, 128, 32), GemmSchedule(64, 256, 128), GemmSchedule(128, 128, 64)):
            path, _ = generate(tvm, (256, 512, 256), schedule, hardware, directory)
            subprocess.run([str(args.build_dir.resolve() / "tile_ir_parity_tests"), str(path), "256", "512", "256",
                            str(hardware["num_sms"]), str(schedule.bm), str(schedule.bn), str(schedule.bk)], check=True)
        # Eleven output columns exercise the final group when its width does
        # not divide eight; the full native coordinate expression is matched.
        path, _ = generate(tvm, (128, 2816, 64), GemmSchedule(), hardware, directory)
        if path.read_text() != manual_gemm(128, 2816, 64).text():
            raise RuntimeError("clamped original CTA order mismatch")
    def reject(action, label):
        try:
            action()
        except ValueError:
            return
        raise RuntimeError(f"accepted unsupported {label}")
    func = scheduled_gemm(tvm, 128, 256, 128)
    reject(lambda: lower_gemm(tvm, func.with_attr("gtsim.tensor_tiers", ["HBF", "HBM", "HBM"])), "HBF placement")
    for operation in ("tirx.ptx.cp_async_bulk_tensor_g2s_cluster", "tirx.ptx.wgmma_f16_ss",
                      "tirx.cuda.mbarrier_wait", "tirx.cuda.cta_sync", "tirx.ptx.wgmma_wait_group", "tirx.ptx.setmaxnreg"):
        bad = func.with_body(replace_calls(tvm, func.body, operation))
        reject(lambda: lower_gemm(tvm, bad), f"missing {operation}")
    reject(lambda: scheduled_gemm(tvm, 128, 256, 128, GemmSchedule(128, 256, 128)), "shared overflow")
    reject(lambda: scheduled_gemm(tvm, 127, 256, 128), "divisibility")
    reject(lambda: scheduled_gemm(tvm, 128, 256, 128, GemmSchedule(128, 64, 64)), "partial warpgroup")
    reject(lambda: scheduled_gemm(tvm, 128, 256, 128, sms=0), "SM count")
    reject(lambda: scheduled_gemm(tvm, 1024, 1024, 128, sms=2), "multiple waves")
    from smoke import copy_program
    reject(lambda: lower_gemm(tvm, copy_program(tvm)), "generic TIR")
    low_capacity = dict(hardware, register_file_per_sm_bytes=1)
    reject(lambda: GemmSchedule().validate(128, 256, 128, 132, low_capacity), "hardware registers")
    result = subprocess.run([str(args.build_dir.resolve() / "gtsim_gemm_ir"),
                             str(ROOT / "configs/v0.1.0/h100-sxm.yaml"), "--memory-config",
                             str(ROOT / "configs/memconf/full-hbf.yaml"), "--describe-config"], capture_output=True)
    if result.returncode == 0:
        raise RuntimeError("accepted HBF topology in preflight")
    print("Native TVM lowering, graph parity and unsupported-semantics checks passed")


if __name__ == "__main__":
    main()
