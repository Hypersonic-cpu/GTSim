#!/usr/bin/env python3
"""Generate explicit Hopper TVM TIR, lower to .gtsir, optionally run GTSim."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from h100_schedule import GemmSchedule, scheduled_gemm
from lower_gemm import lower_gemm
from pinned_tvm import ROOT, TVM_COMMIT, load_tvm

ARCH = ROOT / "configs/v0.1.0/h100-sxm.yaml"
MEMORY = ROOT / "configs/memconf/full-hbm.yaml"


def cycles(text):
    matches = re.findall(r"^Total cycles:\s*(\d+)\s*$", text, re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError("missing or ambiguous simulator cycle result")
    return int(matches[0])


def source_hashes():
    patterns = ("scripts/tvm/*.py", "include/*.h", "src/*.cpp", "workloads/gemm/*.cpp",
                "configs/v0.1.0/h100-sxm.yaml", "configs/memconf/full-hbm.yaml")
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for pattern in patterns for path in sorted(ROOT.glob(pattern))}


def hardware_config(build, architecture=ARCH, memory=MEMORY):
    result = subprocess.run([str(Path(build).resolve() / "gtsim_gemm_ir"), str(architecture),
                             "--memory-config", str(memory), "--describe-config"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise ValueError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def generate(tvm, dimensions, schedule, hardware, out):
    schedule.validate(*dimensions, hardware["num_sms"], hardware)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    func = scheduled_gemm(tvm, *dimensions, schedule, hardware["num_sms"])
    # Save the native TVM boundary and consume a deserialized copy: the lowerer
    # cannot obtain a secretly precomputed Tile IR from the generator.
    serialized = tvm.ir.save_json(tvm.IRModule({"gemm": func}))
    (out / "scheduled.json").write_text(serialized + "\n")
    (out / "scheduled.py").write_text(func.script() + "\n")
    # Exercise the pin's real post-extraction passes as a compiler check. The
    # frontend still consumes the scheduled IR, whose symbolic regions survive.
    lowered = tvm.IRModule({"gemm": func})
    for compiler_pass in (tvm.tirx.transform.StmtSimplify(), tvm.tirx.transform.FlattenBuffer(), tvm.tirx.transform.LowerIntrin()):
        lowered = compiler_pass(lowered)
    (out / "lowered-tvm.py").write_text(lowered.script() + "\n")
    return translate(tvm, tvm.ir.load_json(serialized)["gemm"], hardware, out)


def translate(tvm, func, hardware, out):
    ir, evidence = lower_gemm(tvm, func)
    schedule = GemmSchedule(*evidence["tile"])
    if evidence["sms"] != hardware["num_sms"]:
        raise ValueError("TIR SM mapping does not match selected hardware")
    schedule.validate(*evidence["dimensions"], hardware["num_sms"], hardware)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "kernel.gtsir"
    ir.write(path)
    graph = {"tensors": ir.tensors, "ctas": ir.ctas,
             "operations": [(op.op, op.cta, op.warp, op.tensor, op.tile, op.dtype,
                             op.after_complete, op.after_issue) for op in ir.operations]}
    evidence.update({"tvm_commit": TVM_COMMIT,
        "ir_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dag_spec_sha256": hashlib.sha256(json.dumps(graph, separators=(",", ":")).encode()).hexdigest(),
        "tir_sha256": hashlib.sha256(tvm.ir.save_json(func).encode()).hexdigest(), "hardware": hardware,
        "scope": "persistent multi-wave CTAs, two stages, 12 warps, split=1, original CTA order; HBM only"})
    (out / "lowering.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return path, evidence


def run_simulation(build, architecture, memory, path, log):
    command = [str(Path(build).resolve() / "gtsim_gemm_ir"), str(architecture),
               "--memory-config", str(memory), str(path.resolve())]
    with Path(log).open("w") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"GTSim failed ({result.returncode}); see {log}")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dimensions", type=int, nargs="*", metavar="M N K")
    parser.add_argument("--bm", type=int, default=128)
    parser.add_argument("--bn", type=int, default=256)
    parser.add_argument("--bk", type=int, default=64)
    parser.add_argument("--input-tir", type=Path, help="lower an existing native TVM JSON module")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--architecture", type=Path, default=ARCH)
    parser.add_argument("--memory-config", type=Path, default=MEMORY)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "validation/results/tvm/generated")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.input_tir is None and len(args.dimensions) != 3:
        parser.error("provide M N K, or --input-tir module.json")
    if args.input_tir is not None and args.dimensions:
        parser.error("dimensions must come from the input TIR when --input-tir is used")
    try:
        hardware = hardware_config(args.build_dir, args.architecture, args.memory_config)
        tvm = load_tvm()
        if args.input_tir:
            module = tvm.ir.load_json(args.input_tir.read_text())
            if not isinstance(module, tvm.IRModule) or len(module.functions) != 1:
                raise ValueError("expected a single-function TVM module")
            func = next(iter(module.functions.values()))
            path, evidence = translate(tvm, func, hardware, args.out_dir)
        else:
            path, evidence = generate(tvm, args.dimensions, GemmSchedule(args.bm, args.bn, args.bk), hardware, args.out_dir)
        if args.run:
            command = run_simulation(args.build_dir, args.architecture, args.memory_config, path, args.out_dir / "simulation.log")
            evidence["simulation_command"] = command
            (args.out_dir / "lowering.json").write_text(json.dumps(evidence, indent=2) + "\n")
            print(f"Total cycles: {cycles((args.out_dir / 'simulation.log').read_text())}")
        print(f"Generated {evidence['nodes']} nodes: {path}")
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"TVM GEMM frontend error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
