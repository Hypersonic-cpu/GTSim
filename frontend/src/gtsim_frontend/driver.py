"""Compilation and invocation of GTSim's execution-program runner."""

import hashlib
import json
from pathlib import Path
import subprocess
import platform
from collections import Counter
from dataclasses import asdict
from .serialization.printer import print_ir
from .passes.instantiate import instantiate
from .passes.lower_hopper_gemm import lower as gemm_lower
from .passes.lower_hopper_fa3 import lower as fa3_lower
from .passes.lower_execution import lower as execution_lower
from .passes.compose import compose
from .serialization.json_codec import dumps, digest


def hardware(runner, architecture, memory_config):
    command = [
        str(Path(runner).resolve()),
        str(Path(architecture).resolve()),
        "--memory-config",
        str(Path(memory_config).resolve()),
        "--describe-config",
    ]
    return json.loads(subprocess.check_output(command, text=True))


def compile_program(program, out_dir, profile, launch_overhead_cycles=0):
    program.verify()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "program.progir").write_text(dumps(program))
    kernels = {}
    files = {}
    manifest = []
    stage_statistics = {}
    for invocation in program.launches:
        directory = out if len(program.launches) == 1 else out / invocation.id
        directory.mkdir(parents=True, exist_ok=True)
        algo = invocation.algorithm
        flow = instantiate(algo, profile["num_sms"])
        arch = (gemm_lower if algo.family == "gemm" else fa3_lower)(flow)
        for c in (arch.resources,):
            if (
                c.shared_bytes > profile["shared_memory_per_cta_bytes"]
                or c.register_bytes > profile["register_file_per_sm_bytes"]
                or c.warps > profile["max_warps_per_sm"]
            ):
                raise ValueError("reference kernel exceeds hardware resource capacity")
        kernel = execution_lower(arch)
        for name, obj in (
            ("kernel.algoir", algo),
            ("kernel.flowir", flow),
            ("kernel.nvir", arch),
            ("kernel.gtsir", kernel),
        ):
            (directory / name).write_text(dumps(obj))
        stage_statistics[invocation.id] = {
            "flow_operations": len(flow.ops),
            "native_patterns": len(arch.ops),
            "execution_nodes": len(kernel.ops),
            "bytes": {
                name: (directory / name).stat().st_size
                for name in ("kernel.flowir", "kernel.nvir", "kernel.gtsir")
            },
        }
        manifest.extend(
            {
                "pass": name,
                "source_hash": obj.source_hash,
                "target_hash": digest(obj),
                "symbol": obj.symbol,
            }
            for name, obj in (
                ("instantiate", flow),
                ("lower_hopper", arch),
                ("lower_execution", kernel),
            )
        )
        kernels[invocation.id] = kernel
        files[invocation.id] = str((directory / "kernel.gtsir").relative_to(out))
    execution = compose(program, kernels, files, launch_overhead_cycles)
    (out / "program.gtsir").write_text(dumps(execution))
    (out / "program.txt").write_text(print_ir(program))
    manifest.append(
        {
            "pass": "compose",
            "inputs": [digest(program)] + [digest(k) for k in kernels.values()],
            "target_hash": digest(execution),
            "hardware_profile_id": hashlib.sha256(
                json.dumps(profile, sort_keys=True).encode()
            ).hexdigest(),
            "invocation_mapping": {l.id: l.kernel_file for l in execution.launches},
        }
    )
    (out / "lowering-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        key: {
            "nodes": len(k.ops),
            "ir_statistics": stage_statistics[key],
            "ctas": len(k.ctas),
            "op_counts": dict(Counter(o.op for o in k.ops)),
            "cta_resources": [asdict(c) for c in k.ctas],
            "tensor_count": len(k.tensors),
            "after_issue": sum(len(o.after_issue) for o in k.ops),
            "after_complete": sum(len(o.after_complete) for o in k.ops),
        }
        for key, k in kernels.items()
    }
    (out / "graph-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    artifacts = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.suffix in (
            ".progir",
            ".algoir",
            ".flowir",
            ".nvir",
            ".gtsir",
        ):
            artifacts.append(
                {
                    "path": str(path.relative_to(out)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    (out / "manifest.json").write_text(
        json.dumps({"format": "gtsim.ir.manifest/1", "artifacts": artifacts}, indent=2)
        + "\n"
    )
    return execution, kernels


def run(
    program, out_dir, runner, architecture, memory_config, launch_overhead_cycles=0
):
    profile = hardware(runner, architecture, memory_config)
    execution, kernels = compile_program(
        program, out_dir, profile, launch_overhead_cycles
    )
    out = Path(out_dir)
    spec = {
        "python_version": platform.python_version(),
        "frontend_version": "0.1.0",
        "runner_sha256": hashlib.sha256(Path(runner).read_bytes()).hexdigest(),
        "hardware": profile,
        "architecture": str(Path(architecture).resolve()),
        "memory_config": str(Path(memory_config).resolve()),
        "yaml_sha256": {
            str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (architecture, memory_config)
        },
    }
    try:
        spec["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        spec["git_commit"] = "unknown"
    source_root = Path(__file__).resolve().parents[3]
    paths = list((source_root / "frontend/src").rglob("*.py"))
    paths += list((source_root / "src/execution_ir").glob("*.cpp")) + list(
        (source_root / "src/runtime").glob("*.cpp")
    )
    paths += [
        source_root / "src/simulator.cpp",
        source_root / "include/simulator.h",
        source_root / "include/memory.h",
    ]
    spec["source_sha256"] = {
        str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }
    (out / "run-spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    command = [
        str(Path(runner).resolve()),
        str(Path(architecture).resolve()),
        "--memory-config",
        str(Path(memory_config).resolve()),
        str((out / "program.gtsir").resolve()),
    ]
    with (out / "simulation.log").open("w") as log:
        result = subprocess.run(
            command, text=True, stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(
            f"GTSim failed ({result.returncode}); see {out}/simulation.log"
        )
    timing = None
    for line in (out / "simulation.log").read_text().splitlines():
        if line.startswith("Program timing JSON: "):
            timing = json.loads(line.split(": ", 1)[1])
    if timing is None:
        raise RuntimeError("runner produced no execution timing")
    (out / "kernel-timings.json").write_text(json.dumps(timing, indent=2) + "\n")
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "program_cycles": timing["total_cycles"],
                "kernels": timing["per_kernel"],
                "per_tier_traffic_bytes": timing["traffic_bytes"],
                "outcome": "PASS",
            },
            indent=2,
        )
        + "\n"
    )
    return timing
