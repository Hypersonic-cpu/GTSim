"""Compile one serving program and execute it through the persistent IPC runner."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import resource
import socket
import subprocess
import traceback
from time import perf_counter

from compile import compile_program, lower_flow
from compile.static import EMPTY, Resolver, bound_resolver
from compile.streaming import (
    ExecutionIRProducer,
    SocketFramedTransport,
    _strides,
    make_program_plan,
)
from compile.opt.partition import (
    PartitionPolicy,
    PhysicalLayout,
    memory_topology,
    partition_flow,
)
from compile.opt.gemm_tiling import (
    GemmTiling,
    apply_gemm_tiling,
    summarize_gemm_flow,
    summaries_json,
    validate_gemm_flow,
    validate_native_gemm,
)
from compile.opt.prefetch import PrefetchPlan, Residency, optimize
from parser import ast as A, parse_file


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def isolate_launch(flow, root, name, expert, params):
    from compile.opt.prefetch import scope

    (region,) = (region for region in flow.regions if region.symbol == name)
    if expert is not None:
        if params["B"] != 1:
            raise ValueError("isolated expert measurement requires one row per expert")
        resolver = scope(region, root)
        active = resolver.value("%ACTIVE")
        if len(active) != len(params["ROUTE_IDS"]):
            raise ValueError("expected exactly one assignment per expert")
        lo = active.index(expert) * resolver.value("%NTILES")
        loop, *outer = region.domain.loops
        region = replace(
            region,
            domain=replace(
                region.domain,
                loops=(
                    replace(loop, start=lo, stop=lo + resolver.value("%NTILES")),
                    *outer,
                ),
            ),
        )
    return replace(flow, regions=(replace(region, after=()),), outputs=())


def plan_program(
    source,
    lib,
    params,
    residency=None,
    layouts=None,
    focus=None,
    expert=None,
    placements=None,
    flashaccel_policy=None,
    physical_layout=None,
    prefetch_enabled=True,
    gemm_tiling=None,
):
    layouts = layouts or {}
    placements = placements or {}
    flow = compile_program(
        source, params, list(lib.glob("*.algoir")), list(lib.glob("*.flowir"))
    )
    root = bound_resolver(flow.static, Resolver(EMPTY, {})).materialize_constants()
    if gemm_tiling is not None:
        flow = apply_gemm_tiling(
            flow,
            gemm_tiling.bm,
            gemm_tiling.bn,
            gemm_tiling.bk,
            gemm_tiling.stages,
        )
        root = bound_resolver(flow.static, Resolver(EMPTY, {})).materialize_constants()
    compiled_tiling = summaries_json(summarize_gemm_flow(flow))
    if gemm_tiling is not None:
        validate_gemm_flow(flow, root)
    known = {tensor.id for tensor in flow.tensors}
    if placements.keys() - known:
        raise ValueError(
            f"unknown tensor placements: {sorted(placements.keys() - known)}"
        )
    tiers = {name: A.MemorySpace(tier) for name, tier in placements.items()}
    for tensor in flow.tensors:
        if tiers.get(tensor.id) == A.MemorySpace.HBF and not (
            tensor.external and tensor.read_only
        ):
            raise ValueError(
                f"HBF placement requires read-only external weights: {tensor.id}"
            )
    flow = replace(
        flow,
        tensors=tuple(
            replace(tensor, tier=tiers.get(tensor.id, tensor.tier))
            for tensor in flow.tensors
        ),
    )
    if focus:
        flow = isolate_launch(flow, root, focus, expert, params)
    specs = {spec.id: spec for spec in flow.tensors}
    for name, layout in layouts.items():
        spec = specs[name]
        if not (spec.read_only and spec.external and spec.tier == A.MemorySpace.HBF):
            raise ValueError(
                f"tile-major experiment requires read-only HBF weights: {name}"
            )
        layout.validate(tuple(root.value(d) for d in spec.shape))
    partition = None
    if flashaccel_policy is not None and physical_layout is not None:
        partition = partition_flow(flow, flashaccel_policy, physical_layout)
        flow = partition.flow

    prefetch = None
    if flashaccel_policy is not None:
        if prefetch_enabled and physical_layout is not None:
            residency = Residency(
                physical_layout.stacks,
                physical_layout.dies_per_stack,
                physical_layout.planes_per_die,
                physical_layout.planes_per_command_group,
                physical_layout.page_bytes,
                physical_layout.sram_bytes_per_plane,
                physical_layout.base_die_sram_bytes_per_stack,
                buffer_count=flashaccel_policy.buffer_count,
                streaming=True,
                address_mapping_policy=physical_layout.address_mapping_policy,
                align_tensors_to_command_group=(
                    physical_layout.align_tensors_to_command_group
                ),
            )
            prefetch = optimize(
                flow,
                residency,
                layouts,
                partition.origin_launches if partition is not None else None,
            )
        else:
            root = bound_resolver(
                flow.static, Resolver(EMPTY, {})
            ).materialize_constants()
            bases = (
                physical_layout.bases(flow.tensors, root)
                if physical_layout is not None
                else None
            )
            prefetch = PrefetchPlan(
                flow,
                (),
                None,
                bases,
                partition.origin_launches if partition is not None else None,
            )
    elif residency is not None:
        prefetch = optimize(flow, Residency(**residency), layouts)
    if prefetch is not None:
        flow = prefetch.flow
    arch = lower_flow(flow)
    if gemm_tiling is not None:
        validate_native_gemm(arch)
    native_tiling = summaries_json(summarize_gemm_flow(flow, arch))
    root = bound_resolver(arch.static, Resolver(EMPTY, {})).materialize_constants()
    tensors = []
    for spec in arch.tensors:
        shape = tuple(root.value(d) for d in spec.shape)
        tensors.append(
            A.KernelTensor(
                len(tensors),
                spec.id,
                shape,
                _strides(shape),
                spec.dtype,
                spec.tier or A.MemorySpace.HBM,
                spec.external,
                spec.read_only,
                layouts.get(spec.id),
            )
        )
    regions = []
    for region in arch.regions:
        arguments = {
            arg.id: root.value(arg.expression) for arg in region.static.arguments
        }
        resolver = Resolver(
            replace(region.static, parameters=(), constraints=()),
            {**root.values, **arguments},
        )
        if not any(
            not resolver.value(g.condition)
            for g in region.static_ifs
            if not g.iterations
        ):
            regions.append(region)
    metadata = {
        "gemm_tiling": compiled_tiling,
        "native_tiling": native_tiling,
        "partition": {
            "enabled": partition is not None,
            "views": len(partition.views) if partition is not None else 0,
            "generated_slabs": sum(
                len(children)
                for children in (partition.origin_launches.values() if partition else ())
                if len(children) > 1
            ),
            "origin_launches": partition.origin_launches
            if partition is not None
            else {},
        },
    }
    return (
        parse_file(source),
        regions,
        replace(arch, regions=tuple(regions)),
        prefetch,
        tuple(tensors),
        metadata,
    )


def run_streaming_interface(
    args,
    params,
    source_program,
    regions,
    arch,
    prefetch,
    tensors,
    layouts,
    residency,
    raw_layouts,
    placements,
    flashaccel=False,
    compile_metadata=None,
):
    plan = make_program_plan(source_program, arch, regions, tensors, prefetch)
    topology = memory_topology(args.memory)
    requested_prefetch_mode = args.prefetch or "default"
    effective_prefetch_mode = (
        "disabled_fullhbm"
        if topology == "FullHBM"
        else (
            "on" if prefetch is not None and prefetch.streaming is not None else "off"
        )
    )
    producer = ExecutionIRProducer(
        arch,
        plan,
        {
            "prefetch_config": prefetch.streaming if prefetch is not None else None,
            "prefetch_bases": prefetch.bases if prefetch is not None else None,
            "layouts": layouts,
            "prefetch_windows": not flashaccel,
        },
    )
    args.out.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(
        [
            *Path("frontend/src").rglob("*.py"),
            *args.lib.glob("*ir"),
            args.source,
            args.params,
        ]
    )
    provenance = {
        "version": "stream-v2",
        "params": params,
        "residency": residency,
        "memory_topology": topology,
        "requested_prefetch_mode": requested_prefetch_mode,
        "effective_prefetch_mode": effective_prefetch_mode,
        "tensor_layouts": raw_layouts,
        "tensor_placements": placements,
        "focus_launch": args.focus_launch,
        "focus_expert": args.focus_expert,
        "launch_overhead_cycles": plan.launch_overhead_cycles,
        "max_frame": args.max_frame,
        "runner_sha256": sha(args.runner),
        "max_cycles": args.max_cycles,
        "context_length": args.context_length,
        "gemm_tiling": {
            "bm": args.gemm_bm,
            "bn": args.gemm_bn,
            "bk": args.gemm_bk,
            "stages": args.gemm_stages,
        },
        "sram_policy": args.sram_policy,
        "architecture_sha256": sha(args.architecture),
        "memory_sha256": sha(args.memory),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {str(path): sha(path) for path in source_paths},
        "transport": "AF_UNIX socketpair; length-prefixed JSON; bounded InstructionBatch",
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if compile_metadata is not None:
        (args.out / "ir_summary.json").write_text(
            json.dumps(compile_metadata, indent=2) + "\n"
        )
    start = perf_counter()
    python_before = resource.getrusage(resource.RUSAGE_SELF)
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    log = (args.out / "simulation.log").open("w")
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    command = [
        str(args.runner),
        str(args.architecture),
        "--memory-config",
        str(args.memory),
        "--stream-fd",
        str(child.fileno()),
        "--max-frame",
        str(args.max_frame),
    ]
    if args.max_cycles:
        command += ["--max-cycles", str(args.max_cycles)]
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, pass_fds=(child.fileno(),)
    )
    child.close()
    transport = SocketFramedTransport(parent, args.max_frame)
    completed, failures, program_end_sent = set(), [], False
    try:
        producer.emit_program_manifest(transport)
        while True:
            message = transport.receive()
            kind = message["type"]
            if kind == "RequestKernel":
                producer.emit_kernel(message["launch_id"], transport)
            elif kind == "KernelDone":
                completed.add(message["launch_id"])
                if len(completed) == len(plan.launches) and not program_end_sent:
                    transport.send("ProgramEnd", symbol=plan.symbol)
                    program_end_sent = True
            elif kind == "ProgramDone":
                break
            elif kind == "Error":
                raise RuntimeError(message.get("message", "GTSim stream error"))
            else:
                raise RuntimeError(f"unexpected GTSim stream message: {kind}")
        returncode = process.wait()
    except BaseException:
        failures.append(traceback.format_exc())
        process.terminate()
        returncode = process.wait()
        raise
    finally:
        parent.close()
        log.close()
    python_after = resource.getrusage(resource.RUSAGE_SELF)
    children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    python_cpu_seconds = (
        python_after.ru_utime
        + python_after.ru_stime
        - python_before.ru_utime
        - python_before.ru_stime
    )
    cpp_cpu_seconds = (
        children_after.ru_utime
        + children_after.ru_stime
        - children_before.ru_utime
        - children_before.ru_stime
    )
    python_peak_rss_kib = int(python_after.ru_maxrss)
    cpp_peak_rss_kib = int(children_after.ru_maxrss)
    outcome = {
        "version": "stream-v2",
        "batch": params.get("B"),
        "program": plan.symbol,
        "memory_topology": topology,
        "requested_prefetch_mode": requested_prefetch_mode,
        "effective_prefetch_mode": effective_prefetch_mode,
        "returncode": returncode,
        "elapsed_seconds": perf_counter() - start,
        "python_cpu_seconds": python_cpu_seconds,
        "cpp_cpu_seconds": cpp_cpu_seconds,
        "python_peak_rss_kib": python_peak_rss_kib,
        "cpp_peak_rss_kib": cpp_peak_rss_kib,
        "peak_rss_kib": max(python_peak_rss_kib, cpp_peak_rss_kib),
        "compile_metadata": compile_metadata,
        "kernels": [],
        "errors": failures,
        "transport": "unix socketpair framed JSON InstructionBatch",
    }
    if returncode:
        raise RuntimeError(f"GTSim streaming runner failed with {returncode}")
    timing_line = next(
        line
        for line in (args.out / "simulation.log").read_text().splitlines()
        if line.startswith("Program timing JSON: ")
    )
    outcome["timing"] = json.loads(timing_line.split(": ", 1)[1])
    outcome["prefetch_stats"] = outcome["timing"].get(
        "prefetch_stats",
        {
            "queued": 0,
            "admitted": 0,
            "completed": 0,
            "resident_read_bytes": 0,
            "fallback_read_bytes": 0,
            "canceled": 0,
        },
    )
    (args.out / "result.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(
        json.dumps(
            {
                "version": "stream-v2",
                "batch": params.get("B"),
                "program": plan.symbol,
                "returncode": returncode,
                "cycles": outcome["timing"].get("total_cycles"),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=Path("frontend/asm/qwen3/one_block.progir")
    )
    parser.add_argument("--lib", type=Path, default=Path("frontend/lib"))
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--prefetch-config", type=Path)
    parser.add_argument("--flashaccel-config", type=Path)
    parser.add_argument("--prefetch", choices=("on", "off"))
    parser.add_argument(
        "--context-length", type=int, choices=(10000, 100000, 200000)
    )
    parser.add_argument("--gemm-bm", type=int)
    parser.add_argument("--gemm-bn", type=int)
    parser.add_argument("--gemm-bk", type=int)
    parser.add_argument("--gemm-stages", type=int)
    parser.add_argument(
        "--sram-policy", choices=("none", "pool", "double"), default="none"
    )
    parser.add_argument("--tensor-layouts", type=Path)
    parser.add_argument("--tensor-placements", type=Path)
    parser.add_argument("--focus-launch")
    parser.add_argument("--focus-expert", type=int)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--max-frame", type=int, default=1 << 20)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.max_frame <= 0xFFFFFFFF:
        parser.error("--max-frame must be in [1, 4294967295]")
    params = json.loads(args.params.read_text())
    if args.context_length is not None:
        params["CONTEXTS"] = [args.context_length]
    tiling_values = (args.gemm_bm, args.gemm_bn, args.gemm_bk, args.gemm_stages)
    if any(value is not None for value in tiling_values) and any(
        value is None for value in tiling_values
    ):
        parser.error(
            "--gemm-bm/--gemm-bn/--gemm-bk/--gemm-stages must be given together"
        )
    residency = (
        json.loads(args.prefetch_config.read_text()) if args.prefetch_config else None
    )
    raw_layouts = (
        json.loads(args.tensor_layouts.read_text()) if args.tensor_layouts else {}
    )
    placements = (
        json.loads(args.tensor_placements.read_text()) if args.tensor_placements else {}
    )
    layout_objects = {
        name: A.TileMajorLayout(**layout) for name, layout in raw_layouts.items()
    }
    if args.flashaccel_config and args.sram_policy != "none":
        parser.error("use either --flashaccel-config or --sram-policy, not both")
    flashaccel_policy = None
    if args.flashaccel_config:
        flashaccel_policy = PartitionPolicy.from_json(args.flashaccel_config)
    elif args.sram_policy != "none":
        flashaccel_policy = PartitionPolicy(
            partition_policy="n_slab_fit_sram",
            buffer_count=1 if args.sram_policy == "pool" else 2,
            bm=args.gemm_bm if args.gemm_bm is not None else 16,
            bn=args.gemm_bn if args.gemm_bn is not None else 64,
            bk=args.gemm_bk if args.gemm_bk is not None else 16,
            stages=args.gemm_stages if args.gemm_stages is not None else 2,
        )
        flashaccel_policy.validate()
    if flashaccel_policy is not None:
        if any(value is not None for value in tiling_values):
            supplied = GemmTiling(*tiling_values)
            configured = GemmTiling(
                flashaccel_policy.bm,
                flashaccel_policy.bn,
                flashaccel_policy.bk,
                flashaccel_policy.stages,
                flashaccel_policy.split,
            )
            if supplied != configured:
                parser.error("direct GEMM tiling conflicts with --flashaccel-config")
        gemm_tiling = GemmTiling(
            flashaccel_policy.bm,
            flashaccel_policy.bn,
            flashaccel_policy.bk,
            flashaccel_policy.stages,
            flashaccel_policy.split,
        )
    else:
        gemm_tiling = GemmTiling(
            args.gemm_bm if args.gemm_bm is not None else 16,
            args.gemm_bn if args.gemm_bn is not None else 64,
            args.gemm_bk if args.gemm_bk is not None else 16,
            args.gemm_stages if args.gemm_stages is not None else 2,
        )
        gemm_tiling.validate()
    # Provenance should describe the effective geometry even when the legacy
    # JSON policy supplied it.
    args.gemm_bm, args.gemm_bn = gemm_tiling.bm, gemm_tiling.bn
    args.gemm_bk, args.gemm_stages = gemm_tiling.bk, gemm_tiling.stages
    topology = memory_topology(args.memory)
    physical_layout = (
        PhysicalLayout.from_memory_yaml(args.memory) if flashaccel_policy else None
    )
    source_program, regions, arch, prefetch, tensors, compile_metadata = plan_program(
        args.source,
        args.lib,
        params,
        None if topology == "FullHBM" else residency,
        layout_objects,
        args.focus_launch,
        args.focus_expert,
        placements,
        flashaccel_policy,
        physical_layout,
        args.prefetch != "off" and topology != "FullHBM",
        gemm_tiling,
    )
    if args.compile_only:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "ir_summary.json").write_text(
            json.dumps(compile_metadata, indent=2) + "\n"
        )
        (args.out / "compile_result.json").write_text(
            json.dumps(
                {
                    "returncode": 0,
                    "context_length": args.context_length,
                    "gemm_tiling": {
                        "bm": args.gemm_bm,
                        "bn": args.gemm_bn,
                        "bk": args.gemm_bk,
                        "stages": args.gemm_stages,
                    },
                    "sram_policy": args.sram_policy,
                    "memory_topology": topology,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({"compiled": True, "returncode": 0}), flush=True)
        return
    run_streaming_interface(
        args,
        params,
        source_program,
        regions,
        arch,
        prefetch,
        tensors,
        layout_objects,
        residency,
        raw_layouts,
        placements,
        flashaccel=flashaccel_policy is not None,
        compile_metadata=compile_metadata,
    )


if __name__ == "__main__":
    main()
