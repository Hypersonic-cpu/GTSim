#!/usr/bin/env python3
"""Focused native-TVM boundary, parity and rejection checks (opt-in CTest)."""

import argparse
import json
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
from semantic_checks import check_semantics, multiwave_cases


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
    for dimensions, tile in multiwave_cases(2):
        func = scheduled_gemm(tvm, *dimensions, GemmSchedule(*tile), sms=2)
        # Consume a saved/reloaded native module, with no C++ parity oracle.
        func = tvm.ir.load_json(tvm.ir.save_json(tvm.IRModule({"gemm": func})))["gemm"]
        ir, evidence = lower_gemm(tvm, func)
        check_semantics(tvm, func, ir, selected_ctas=range(len(ir.ctas)))
        assert evidence["waves"] > 1
    # A completion wait must not accidentally become an issue-only wait.
    func = scheduled_gemm(tvm, 128, 768, 192, sms=2)
    ir, _ = lower_gemm(tvm, func)
    changed = [op for op in ir.operations if op.name.startswith("TB0_Tile2_TMA_Load") and '_k0' in op.name]
    for op in changed:
        op.after_issue += op.after_complete
        op.after_complete = []
    try:
        check_semantics(tvm, func, ir)
    except AssertionError:
        pass
    else:
        raise RuntimeError("semantic audit accepted issue-only cross-wave store waits")
    def reject(action, label):
        try:
            action()
        except ValueError:
            return
        raise RuntimeError(f"accepted unsupported {label}")
    func = scheduled_gemm(tvm, 128, 256, 128)
    reject(lambda: lower_gemm(tvm, func.with_attr("gtsim.tensor_tiers", ["HBF", "HBM", "HBM"])), "tier/scope mismatch")
    for operation in ("tirx.ptx.cp_async_bulk_tensor_g2s_cluster", "tirx.ptx.wgmma_f16_ss",
                      "tirx.cuda.mbarrier_wait", "tirx.cuda.cta_sync", "tirx.ptx.wgmma_wait_group", "tirx.ptx.setmaxnreg",
                      "tirx.ptx.cp_async_bulk_wait_group", "tirx.ptx.cp_async_bulk_commit_group"):
        bad = func.with_body(replace_calls(tvm, func.body, operation))
        reject(lambda: lower_gemm(tvm, bad), f"missing {operation}")
    reject(lambda: scheduled_gemm(tvm, 128, 256, 128, GemmSchedule(128, 256, 128)), "shared overflow")
    reject(lambda: scheduled_gemm(tvm, 127, 256, 128), "divisibility")
    reject(lambda: scheduled_gemm(tvm, 128, 256, 128, GemmSchedule(128, 64, 64)), "partial warpgroup")
    reject(lambda: scheduled_gemm(tvm, 128, 256, 128, sms=0), "SM count")
    from smoke import copy_program
    reject(lambda: lower_gemm(tvm, copy_program(tvm)), "generic TIR")
    low_capacity = dict(hardware, register_file_per_sm_bytes=1)
    reject(lambda: GemmSchedule().validate(128, 256, 128, 132, low_capacity), "hardware registers")
    result = subprocess.run([str(args.build_dir.resolve() / "gtsim_gemm_ir"),
                             str(ROOT / "configs/v0.1.0/h100-sxm.yaml"), "--memory-config",
                             str(ROOT / "configs/memconf/full-hbf.yaml"), "--describe-config"], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError("HBF topology hardware query failed")
    if json.loads(result.stdout)["topology"] != "FullHBF":
        raise RuntimeError("hardware query lost topology identity")
    bad = scheduled_gemm(tvm, 128, 768, 192, sms=2)
    reject(lambda: lower_gemm(tvm, bad.with_body(replace_calls(tvm, bad.body, "tirx.ptx.cp_async_bulk_wait_group"))),
           "missing cross-wave bulk completion wait")
    print("Native TVM single-wave parity, multi-wave TIR semantics and rejection checks passed")


if __name__ == "__main__":
    main()
