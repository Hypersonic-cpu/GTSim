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


def compile_decode_program(program, out_dir, profile, launch_overhead_cycles=0):
    """Lower each typed decode invocation independently to executable IR-05."""
    from collections import Counter
    from .ir.decode import DecodeProgram, KernelFamily
    from .ir.execution import ExecProgram, ExecutionLaunch
    from .ir.graph import KernelTensor
    from .ir.program import Command
    from .ir.decode import AttentionWork, ExpertGemmWork, GemmWork

    if not isinstance(program, DecodeProgram):
        raise TypeError("expected typed DecodeProgram")
    program.verify()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "program.progir").write_text(dumps(program))

    global_ids = {tensor.id: index for index, tensor in enumerate(program.tensors)}
    global_tensors = tuple(
        KernelTensor(
            id=global_ids[tensor.id],
            name=tensor.id,
            shape=tensor.shape,
            strides=(tensor.shape[1], 1),
            dtype=tensor.dtype,
            tier=tensor.tier,
            preloaded=(
                tensor.read_only or tensor.id == "input.hidden" or ".kv." in tensor.id
            ),
            read_only=tensor.read_only,
        )
        for tensor in program.tensors
    )

    launches = []
    kernel_statistics = {}
    manifest = []
    for invocation in program.launches:
        directory = out / invocation.id
        directory.mkdir(parents=True, exist_ok=True)
        flow = instantiate(invocation.algorithm, profile["num_sms"])
        hopper_lower = (
            fa3_lower
            if invocation.algorithm.family
            in (KernelFamily.FA3_DECODE, KernelFamily.FA3_COMBINE)
            else gemm_lower
        )
        native = hopper_lower(flow)
        kernel = execution_lower(native, profile["num_sms"])
        if any(
            c.shared_bytes > profile["shared_memory_per_cta_bytes"]
            or c.register_bytes > profile["register_file_per_sm_bytes"]
            or c.warps > profile["max_warps_per_sm"]
            for c in kernel.ctas
        ):
            raise ValueError(
                f"{invocation.id}: decode kernel exceeds H100 CTA resources"
            )
        for name, obj in (
            ("kernel.algoir", invocation.algorithm),
            ("kernel.flowir", flow),
            ("kernel.nvir", native),
            ("kernel.gtsir", kernel),
        ):
            (directory / name).write_text(dumps(obj))
            manifest.append(
                {
                    "path": str((directory / name).relative_to(out)),
                    "sha256": hashlib.sha256(
                        (directory / name).read_bytes()
                    ).hexdigest(),
                    "symbol": obj.symbol,
                }
            )

        global_bindings = dict(invocation.bindings)
        execution_bindings = tuple(
            (tensor.id, global_ids[global_bindings[tensor.name]])
            for tensor in kernel.tensors
        )
        launches.append(
            ExecutionLaunch(
                invocation.id,
                str((directory / "kernel.gtsir").relative_to(out)),
                execution_bindings,
            )
        )
        requested_bytes = Counter()
        physical_tensor_flops = 0
        vector_stage_elements = 0
        for op in kernel.ops:
            if op.op == "mma.sync.m16n8k16":
                physical_tensor_flops += 2 * 16 * 8 * 16
            elif op.op.startswith("simd.") or op.op.startswith("sfu."):
                vector_stage_elements += op.tile[2] * op.tile[3]
            if op.op not in ("cp.dram2sram_tma", "cp.sram2dram_tma"):
                continue
            dtype_bytes = 4 if op.dtype in ("fp32", "int32") else 2
            key = "read" if op.op == "cp.dram2sram_tma" else "write"
            requested_bytes[key] += op.tile[2] * op.tile[3] * dtype_bytes
        work = invocation.algorithm.work
        semantic_flops = 0
        if isinstance(work, GemmWork):
            semantic_flops = 2 * work.rows * work.columns * work.reduction
        elif isinstance(work, ExpertGemmWork):
            semantic_flops = (
                2
                * sum(len(group.assignments) for group in work.groups)
                * work.columns
                * work.reduction
            )
        elif isinstance(work, AttentionWork):
            semantic_flops = (
                4
                * sum(length + 1 for length in work.context_lengths)
                * work.query_heads
                * work.head_dim
            )
        kernel_statistics[invocation.id] = {
            "flow_operations": len(flow.ops),
            "native_patterns": len(native.ops),
            "execution_nodes": len(kernel.ops),
            "ctas": len(kernel.ctas),
            "cta_resources": {
                "shared_bytes": kernel.ctas[0].shared_bytes,
                "register_bytes": kernel.ctas[0].register_bytes,
                "warps": kernel.ctas[0].warps,
            },
            "op_counts": dict(Counter(op.op for op in kernel.ops)),
            "semantic_flops": semantic_flops,
            "physical_tensor_mma_flops": physical_tensor_flops,
            "vector_stage_elements": vector_stage_elements,
            "global_memory_requested_bytes": dict(requested_bytes),
            "tensor_count": len(kernel.tensors),
            "partial_outputs": [
                kernel.tensors[index].name for index in kernel.partial_outputs
            ],
        }
        del kernel

    commands = [Command(kind, target) for kind, target in program.commands]
    commands.append(Command("StreamSynchronize"))
    execution = ExecProgram(
        symbol=program.symbol,
        source_hash=digest(program),
        streams=("compute",),
        tensors=global_tensors,
        launches=tuple(launches),
        commands=tuple(commands),
        launch_overhead_cycles=launch_overhead_cycles,
    )
    execution.verify()
    (out / "program.gtsir").write_text(dumps(execution))
    manifest.append(
        {
            "path": "program.progir",
            "sha256": hashlib.sha256((out / "program.progir").read_bytes()).hexdigest(),
            "symbol": program.symbol,
        }
    )
    manifest.append(
        {
            "path": "program.gtsir",
            "sha256": hashlib.sha256((out / "program.gtsir").read_bytes()).hexdigest(),
            "symbol": execution.symbol,
        }
    )
    (out / "manifest.json").write_text(
        json.dumps({"format": "gtsim.ir.manifest/1", "artifacts": manifest}, indent=2)
        + "\n"
    )
    (out / "graph-summary.json").write_text(
        json.dumps(kernel_statistics, indent=2) + "\n"
    )
    return execution, kernel_statistics


def _decode_kernel_config(invocation, layer_summary, batch_size):
    from .ir.decode import (
        AttentionWork,
        CombineWork,
        ExpertGemmWork,
        GemmWork,
        KVUpdateWork,
        QKVTransformWork,
        RouterTopKWork,
        VectorWork,
        WeightedSumWork,
    )

    work = invocation.algorithm.work
    config = {"variant": invocation.variant}
    if isinstance(work, GemmWork):
        config.update(
            {
                "m": work.rows,
                "n": work.columns,
                "k": work.reduction,
                "tile_m": work.tile_m,
                "tile_n": work.tile_n,
                "tile_k": work.tile_k,
                "physical_m_rows": ((work.rows + work.tile_m - 1) // work.tile_m)
                * work.tile_m,
                "masked_m_rows": ((work.rows + work.tile_m - 1) // work.tile_m)
                * work.tile_m
                - work.rows,
            }
        )
    elif isinstance(work, QKVTransformWork):
        config.update(
            {
                "query_heads": work.query_heads,
                "kv_heads": work.kv_heads,
                "head_dim": work.head_dim,
                "norm_epsilon": work.epsilon,
                "rope_theta": work.rope_theta,
                "positions": list(work.positions),
            }
        )
    elif isinstance(work, KVUpdateWork):
        config.update(
            {
                "appends": len(work.appends),
                "head_dim": work.head_dim,
                "per_request_context_lengths": [
                    a.context_length for a in work.appends if a.head == 0
                ],
            }
        )
    elif isinstance(work, AttentionWork):
        config.update(
            {
                "query_heads": work.query_heads,
                "kv_heads": work.kv_heads,
                "gqa_ratio": work.query_heads // work.kv_heads,
                "head_dim": work.head_dim,
                "kv_tile": work.kv_tile,
                "split_count": work.split_count,
                "query_ctas": len(work.requests) * work.query_heads * work.split_count,
                "shared_memory_bytes_per_cta": 32768
                + 2 * work.kv_tile * work.head_dim * 2,
                "effective_context_lengths": [n + 1 for n in work.context_lengths],
                "chunk_count": len(work.chunks),
            }
        )
    elif isinstance(work, CombineWork):
        config.update(
            {
                "requests": work.requests,
                "heads": work.heads,
                "splits": work.splits,
                "head_dim": work.head_dim,
                "normalization": "stable_logsumexp_weighted_merge",
            }
        )
    elif isinstance(work, RouterTopKWork):
        config.update(
            {
                "experts": work.experts,
                "top_k": work.top_k,
                "probabilities": "softmax_then_selected_probability_renormalization",
            }
        )
    elif isinstance(work, ExpertGemmWork):
        config.update(
            {
                "phase": work.phase,
                "active_experts": len(work.groups),
                "assignments": sum(len(group.assignments) for group in work.groups),
                "tile_m": work.tile_m,
                "tile_n": work.tile_n,
                "tile_k": work.tile_k,
                "expert_rows": {
                    str(group.expert): len(group.assignments) for group in work.groups
                },
                "padded_rows": sum(
                    ((len(group.assignments) + work.tile_m - 1) // work.tile_m)
                    * work.tile_m
                    - len(group.assignments)
                    for group in work.groups
                ),
                "row_utilization": sum(len(group.assignments) for group in work.groups)
                / max(
                    1,
                    sum(
                        ((len(group.assignments) + work.tile_m - 1) // work.tile_m)
                        * work.tile_m
                        for group in work.groups
                    ),
                ),
            }
        )
    elif isinstance(work, VectorWork):
        config.update(
            {
                "rows": work.rows,
                "columns": work.columns,
                "output_width": work.output_width or work.columns,
                "stages": [stage.operation.value for stage in work.stages],
                "epsilon": work.epsilon,
            }
        )
    elif isinstance(work, WeightedSumWork):
        config.update(
            {
                "top_k": work.top_k,
                "assignments": work.batch * work.top_k,
                "direct_order": work.direct_order,
            }
        )
    if invocation.name.startswith("expert_") or invocation.name == "weighted_moe_sum":
        config["moe_variant"] = layer_summary["moe_variant"]
        config["active_experts"] = layer_summary["active_experts"]
        config["expert_token_counts"] = layer_summary["expert_token_counts"]
        config["grouped_padded_rows"] = layer_summary["grouped_padded_rows"]
        config["grouped_row_utilization"] = layer_summary["grouped_row_utilization"]
        config["grouped_alignment_amplification"] = layer_summary[
            "grouped_alignment_amplification"
        ]
        config["expert_alignment"] = layer_summary["expert_alignment"]
    return config


def run_decode(
    program,
    workload_summary,
    out_dir,
    runner,
    architecture,
    memory_config,
    launch_overhead_cycles=0,
):
    """Compile, execute, and report a typed one/two-layer decode stream."""
    profile = hardware(runner, architecture, memory_config)
    execution, kernel_statistics = compile_decode_program(
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
        "launch_overhead_cycles": launch_overhead_cycles,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "routing_source": workload_summary["routing_source"],
        "routing_seed": workload_summary["routing_seed"],
    }
    spec["yaml_sha256"] = {
        str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (architecture, memory_config)
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
    raw_timing = None
    for line in (out / "simulation.log").read_text().splitlines():
        if line.startswith("Program timing JSON: "):
            raw_timing = json.loads(line.split(": ", 1)[1])
    if raw_timing is None:
        raise RuntimeError("runner produced no execution timing")
    (out / "simulator-timing.json").write_text(json.dumps(raw_timing, indent=2) + "\n")

    invocation_by_id = {launch.id: launch for launch in program.launches}
    layer_summaries = {
        entry["layer"]: entry for entry in workload_summary["layer_summaries"]
    }
    timeline_kernels = []
    for result in raw_timing["per_kernel"]:
        invocation = invocation_by_id[result["id"]]
        layer_summary = layer_summaries[invocation.layer]
        config = _decode_kernel_config(invocation, layer_summary, program.batch_size)
        timeline_kernels.append(
            {
                "layer": invocation.layer,
                "name": invocation.name,
                "variant": invocation.variant,
                "launch_cycle": result["launch_cycle"],
                "start_cycle": result["start_cycle"],
                "device_complete_cycle": result["device_complete_cycle"],
                "complete_cycle": result["complete_cycle"],
                "duration_cycles": result["complete_cycle"] - result["start_cycle"],
                "device_duration_cycles": result["device_cycles"],
                "hbm_read_bytes": result["hbm_read_bytes"],
                "hbm_write_bytes": result["hbm_write_bytes"],
                "hbf_read_bytes": result["hbf_read_bytes"],
                "hbf_write_bytes": result["hbf_write_bytes"],
                "l2_lines_before": result["l2_lines_before"],
                "l2_lines_after": result["l2_lines_after"],
                "configuration": config,
                "graph": kernel_statistics[invocation.id],
            }
        )

    layer_timeline = []
    for layer in range(program.layers):
        items = [item for item in timeline_kernels if item["layer"] == layer]
        first = items[0]
        last = items[-1]
        layer_timeline.append(
            {
                "layer": layer,
                "start_cycle": first["launch_cycle"],
                "end_cycle": last["complete_cycle"],
                "duration_cycles": last["complete_cycle"] - first["launch_cycle"],
                "new_tokens": program.batch_size,
                "output_shape": [program.batch_size, 2048],
                "fa3_variant": layer_summaries[layer]["fa3_variant"],
                "moe_variant": layer_summaries[layer]["moe_variant"],
                "active_experts": layer_summaries[layer]["active_experts"],
            }
        )

    timeline = {
        "model": program.model,
        "mode": program.mode,
        "batch_size": program.batch_size,
        "layers": program.layers,
        "context_lengths": list(program.context_lengths),
        "effective_context_lengths": [length + 1 for length in program.context_lengths],
        "routing_source": workload_summary["routing_source"],
        "routing_seed": workload_summary["routing_seed"],
        "stream": {
            "id": 0,
            "start_cycle": 0,
            "end_cycle": raw_timing["total_cycles"],
            "duration_cycles": raw_timing["total_cycles"],
            "launch_overhead_cycles_per_kernel": launch_overhead_cycles,
        },
        "layers_summary": layer_timeline,
        "layer_work_summary": workload_summary["layer_summaries"],
        "kernels": timeline_kernels,
        "memory_statistics": {
            "program_traffic_bytes": raw_timing["traffic_bytes"],
            "persistent_gpu": raw_timing["persistent_gpu"],
            "persistent_memory": raw_timing["persistent_memory"],
            "persistent_l2": raw_timing["persistent_l2"],
            "stable_allocations": raw_timing["stable_allocations"],
            "hbf_accessed": sum(
                item["hbf_read_bytes"] + item["hbf_write_bytes"]
                for item in timeline_kernels
            )
            > 0,
        },
        "outcome": "PASS",
    }
    (out / "kernel-timings.json").write_text(json.dumps(timeline, indent=2) + "\n")
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "program_cycles": raw_timing["total_cycles"],
                "per_kernel": timeline_kernels,
                "memory_statistics": timeline["memory_statistics"],
                "outcome": "PASS",
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        f"# {program.model} single decode step",
        "",
        f"- Batch: {program.batch_size}; layers: {program.layers}",
        f"- Context lengths: {', '.join(map(str, program.context_lengths))}",
        f"- Effective KV lengths after append: {', '.join(map(str, (n + 1 for n in program.context_lengths)))}",
        f"- Stream cycles: {raw_timing['total_cycles']}",
        f"- HBM read/write bytes: {raw_timing['traffic_bytes']['HBM_read']} / {raw_timing['traffic_bytes']['HBM_write']}",
        f"- HBF read/write bytes: {raw_timing['traffic_bytes']['HBF_read']} / {raw_timing['traffic_bytes']['HBF_write']}",
        f"- Routing source: {workload_summary['routing_source']}",
        "",
        "| # | Layer | Kernel | Variant | Launch | Start | Device done | Complete | Duration | HBM read | HBM write |",
        "|--:|--:|---|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for index, item in enumerate(timeline_kernels):
        lines.append(
            f"| {index} | {item['layer']} | {item['name']} | {item['variant']} | {item['launch_cycle']} | {item['start_cycle']} | {item['device_complete_cycle']} | {item['complete_cycle']} | {item['duration_cycles']} | {item['hbm_read_bytes']} | {item['hbm_write_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Fidelity limits",
            "",
            "- KV history uses separate contiguous request/head allocations; this does not model Paged KV addressing.",
            "- Routing is a supplied trace or deterministic synthetic Top-8 input. Router GEMM and softmax/TopK still execute as timed kernels; tensor values are not numerically inferred by GTSim.",
            "- Decode and grouped-expert instruction patterns use the simulator's current Hopper pipeline profile. New kernel timing calibration against an H100 serving implementation has not been established by this simulator run.",
            "- No prefill, sampling, LM head, generation loop, prefetch, or concurrent stream is included.",
            "",
        ]
    )
    for layer in workload_summary["layer_summaries"]:
        alignment_summary = (
            f"{layer['grouped_padded_rows']} padded rows, utilization "
            f"{layer['grouped_row_utilization']}, amplification "
            f"{layer['grouped_alignment_amplification']}"
            if layer["moe_variant"] == "grouped"
            else f"{layer['direct_padded_rows']} padded rows, utilization "
            f"{layer['direct_row_utilization']}, amplification "
            f"{layer['direct_alignment_amplification']}"
        )
        lines.extend(
            [
                f"## Layer {layer['layer']} routing and alignment",
                "",
                f"- FA3: {layer['fa3_variant']}, split count {layer['fa3_split_count']}, {layer['fa3_query_ctas']} active query CTAs.",
                f"- MoE: {layer['moe_variant']}, {layer['active_experts']} active experts, {layer['expert_assignment_count']} token/expert assignments.",
                f"- M=16 alignment: {alignment_summary}.",
                "- Expert assignment counts: "
                + ", ".join(
                    f"{expert}:{count}"
                    for expert, count in layer["expert_token_counts"].items()
                ),
                "",
            ]
        )
    (out / "results.md").write_text("\n".join(lines))
    return timeline
