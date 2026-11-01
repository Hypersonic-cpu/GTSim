"""Compilation and invocation of GTSim's execution-program runner."""

import hashlib
import json
from pathlib import Path
import subprocess
import platform
from collections import Counter
from dataclasses import asdict
from .serialization.printer import print_ir
from .passes.hopper.lower import lower as lower_patterns
from .passes.instantiate import instantiate
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


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def compile_program(program, out_dir, profile, launch_overhead_cycles=0):
    """One compiler for every workload; keep at most one eager kernel DAG."""
    from .passes.compose import KernelInterface

    program.verify()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "program.progir").write_text(dumps(program))
    interfaces = {}
    files = {}
    manifest = []
    summary = {}
    artifacts = []
    for invocation in program.launches:
        directory = out if len(program.launches) == 1 else out / invocation.id
        directory.mkdir(parents=True, exist_ok=True)
        algorithm = invocation.algorithm
        flow = instantiate(algorithm, profile["num_sms"])
        native = lower_patterns(flow)
        resource = native.resources
        if (
            resource.shared_bytes > profile["shared_memory_per_cta_bytes"]
            or resource.register_bytes > profile["register_file_per_sm_bytes"]
            or resource.warps > profile["max_warps_per_sm"]
        ):
            raise ValueError(
                f"{invocation.id}: kernel resource footprint exceeds hardware"
            )
        kernel = execution_lower(native)
        hashes = {}
        for name, obj in (
            ("kernel.algoir", algorithm),
            ("kernel.flowir", flow),
            ("kernel.nvir", native),
            ("kernel.gtsir", kernel),
        ):
            path = directory / name
            path.write_text(dumps(obj))
            hashes[name] = file_hash(path)
            artifacts.append(
                {"path": str(path.relative_to(out)), "sha256": hashes[name]}
            )
        for name, obj, filename in (
            ("instantiate", flow, "kernel.flowir"),
            ("lower_hopper", native, "kernel.nvir"),
            ("lower_execution", kernel, "kernel.gtsir"),
        ):
            manifest.append(
                {
                    "pass": name,
                    "source_hash": obj.source_hash,
                    "target_hash": hashes[filename],
                    "symbol": obj.symbol,
                }
            )
        requested = Counter()
        physical_mma_flops = 0
        for op in kernel.ops:
            if op.op in ("cp.dram2sram_tma", "cp.sram2dram_tma"):
                requested["read" if op.op == "cp.dram2sram_tma" else "write"] += (
                    op.tile[2]
                    * op.tile[3]
                    * (4 if op.dtype in ("fp32", "int32") else 2)
                )
            if op.op in ("mma.sync.m16n8k16", "wgmma.m64n64k16"):
                physical_mma_flops += 2 * op.tile[2] * op.tile[3] * 16
        logical = 0
        if algorithm.family == "gemm":
            m, n, k = algorithm.dimensions
            logical = 2 * m * n * k
        elif algorithm.schedule_id == "decode_simt":
            logical = (
                4
                * sum(algorithm.valid_lengths)
                * algorithm.dimensions[1]
                * algorithm.head_dim
            )
        elif algorithm.schedule_id == "decode_combine":
            logical = (
                2
                * algorithm.dimensions[0]
                * algorithm.dimensions[1]
                * algorithm.split_count
                * algorithm.head_dim
            )
        stats = {
            "flow_operations": len(flow.ops),
            "native_patterns": len(native.ops),
            "execution_nodes": len(kernel.ops),
            "bytes": {
                name: (directory / name).stat().st_size
                for name in ("kernel.flowir", "kernel.nvir", "kernel.gtsir")
            },
        }
        summary[invocation.id] = {
            "nodes": len(kernel.ops),
            "ir_statistics": stats,
            "ctas": len(kernel.ctas),
            "op_counts": dict(Counter(o.op for o in kernel.ops)),
            "cta_resources": [asdict(c) for c in kernel.ctas],
            "tensor_count": len(kernel.tensors),
            "after_issue": sum(len(o.after_issue) for o in kernel.ops),
            "after_complete": sum(len(o.after_complete) for o in kernel.ops),
            "logical_matrix_flops": logical,
            "physical_tensor_mma_flops": physical_mma_flops,
            "physical_simt_matrix_flops": logical
            if algorithm.schedule_id in ("decode_simt", "decode_combine")
            else 0,
            "global_memory_requested_bytes": dict(requested),
            "schedule_id": algorithm.schedule_id,
            "family": algorithm.family,
            "timing_calibrated": algorithm.schedule_id == "reference",
        }
        interfaces[invocation.id] = KernelInterface(
            kernel.tensors, kernel.outputs, hashes["kernel.gtsir"]
        )
        files[invocation.id] = str((directory / "kernel.gtsir").relative_to(out))
        del kernel, native, flow
    execution = compose(program, interfaces, files, launch_overhead_cycles)
    (out / "program.gtsir").write_text(dumps(execution))
    (out / "program.txt").write_text(print_ir(program))
    manifest.append(
        {
            "pass": "compose",
            "inputs": [digest(program)] + [k.content_hash for k in interfaces.values()],
            "target_hash": digest(execution),
            "hardware_profile_id": hashlib.sha256(
                json.dumps(profile, sort_keys=True).encode()
            ).hexdigest(),
            "invocation_mapping": {l.id: l.kernel_file for l in execution.launches},
        }
    )
    (out / "lowering-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "graph-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    artifacts.extend(
        {"path": name, "sha256": file_hash(out / name)}
        for name in ("program.progir", "program.gtsir")
    )
    (out / "manifest.json").write_text(
        json.dumps({"format": "gtsim.ir.manifest/1", "artifacts": artifacts}, indent=2)
        + "\n"
    )
    return execution, interfaces


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


def run_decode(
    program,
    workload_summary,
    out_dir,
    runner,
    architecture,
    memory_config,
    launch_overhead_cycles=0,
):
    """Use the standard compiler/executor, then add workload-only reporting."""
    raw = run(
        program, out_dir, runner, architecture, memory_config, launch_overhead_cycles
    )
    from .reporting import decode_report

    return decode_report(
        program, workload_summary, out_dir, raw, launch_overhead_cycles
    )
