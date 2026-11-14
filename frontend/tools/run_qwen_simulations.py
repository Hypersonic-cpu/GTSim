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
from compile.streaming import ExecutionIRProducer, SocketFramedTransport, _strides, make_program_plan
from compile.opt.partition import (
    PartitionPolicy,
    PhysicalLayout,
    memory_topology,
    partition_flow,
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
                loops=(replace(loop, start=lo, stop=lo + resolver.value("%NTILES")), *outer),
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
):
    layouts = layouts or {}
    placements = placements or {}
    flow = compile_program(source, params, list(lib.glob("*.algoir")), list(lib.glob("*.flowir")))
    known = {tensor.id for tensor in flow.tensors}
    if placements.keys() - known:
        raise ValueError(f"unknown tensor placements: {sorted(placements.keys() - known)}")
    tiers = {name: A.MemorySpace(tier) for name, tier in placements.items()}
    for tensor in flow.tensors:
        if tiers.get(tensor.id) == A.MemorySpace.HBF and not (tensor.external and tensor.read_only):
            raise ValueError(f"HBF placement requires read-only external weights: {tensor.id}")
    flow = replace(
        flow,
        tensors=tuple(replace(tensor, tier=tiers.get(tensor.id, tensor.tier)) for tensor in flow.tensors),
    )
    root = bound_resolver(flow.static, Resolver(EMPTY, {})).materialize_constants()
    if focus:
        flow = isolate_launch(flow, root, focus, expert, params)
    specs = {spec.id: spec for spec in flow.tensors}
    for name, layout in layouts.items():
        spec = specs[name]
        if not (spec.read_only and spec.external and spec.tier == A.MemorySpace.HBF):
            raise ValueError(f"tile-major experiment requires read-only HBF weights: {name}")
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
            )
            prefetch = optimize(
                flow,
                residency,
                layouts,
                partition.origin_launches if partition is not None else None,
            )
        else:
            root = bound_resolver(flow.static, Resolver(EMPTY, {})).materialize_constants()
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
    root = bound_resolver(arch.static, Resolver(EMPTY, {})).materialize_constants()
    tensors = []
    for spec in arch.tensors:
        shape = tuple(root.value(d) for d in spec.shape)
        tensors.append(
            A.KernelTensor(
                len(tensors), spec.id, shape, _strides(shape), spec.dtype,
                spec.tier or A.MemorySpace.HBM, spec.external, spec.read_only,
                layouts.get(spec.id),
            )
        )
    regions = []
    for region in arch.regions:
        arguments = {arg.id: root.value(arg.expression) for arg in region.static.arguments}
        resolver = Resolver(replace(region.static, parameters=(), constraints=()),
                            {**root.values, **arguments})
        if not any(not resolver.value(g.condition) for g in region.static_ifs if not g.iterations):
            regions.append(region)
    return parse_file(source), regions, replace(arch, regions=tuple(regions)), prefetch, tuple(tensors)


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
):
    plan = make_program_plan(source_program, arch, regions, tensors, prefetch)
    producer = ExecutionIRProducer(
        arch, plan,
        {"prefetch_config": prefetch.streaming if prefetch is not None else None,
         "prefetch_bases": prefetch.bases if prefetch is not None else None,
         "layouts": layouts,
         "prefetch_windows": not flashaccel},
    )
    args.out.mkdir(parents=True, exist_ok=True)
    source_paths = sorted([*Path("frontend/src").rglob("*.py"), *args.lib.glob("*ir"),
                           args.source, args.params])
    provenance = {
        "version": "stream-v2",
        "params": params,
        "residency": residency,
        "tensor_layouts": raw_layouts,
        "tensor_placements": placements,
        "focus_launch": args.focus_launch,
        "focus_expert": args.focus_expert,
        "launch_overhead_cycles": plan.launch_overhead_cycles,
        "max_frame": args.max_frame,
        "runner_sha256": sha(args.runner),
        "max_cycles": args.max_cycles,
        "architecture_sha256": sha(args.architecture),
        "memory_sha256": sha(args.memory),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(path): sha(path) for path in source_paths},
        "transport": "AF_UNIX socketpair; length-prefixed JSON; bounded InstructionBatch",
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    start = perf_counter()
    python_before = resource.getrusage(resource.RUSAGE_SELF)
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    log = (args.out / "simulation.log").open("w")
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    command = [str(args.runner), str(args.architecture), "--memory-config", str(args.memory),
               "--stream-fd", str(child.fileno()), "--max-frame", str(args.max_frame)]
    if args.max_cycles:
        command += ["--max-cycles", str(args.max_cycles)]
    if args.verify_memory_reads:
        command += ["--verify-memory-reads"]
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               pass_fds=(child.fileno(),))
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
    outcome = {"version": "stream-v2", "batch": params.get("B"), "program": plan.symbol,
               "returncode": returncode, "elapsed_seconds": perf_counter() - start,
               "python_cpu_seconds": python_cpu_seconds,
               "cpp_cpu_seconds": cpp_cpu_seconds,
               "python_peak_rss_kib": python_peak_rss_kib,
               "cpp_peak_rss_kib": cpp_peak_rss_kib,
               "peak_rss_kib": max(python_peak_rss_kib, cpp_peak_rss_kib),
               "kernels": [], "errors": failures,
               "transport": "unix socketpair framed JSON InstructionBatch"}
    if returncode:
        raise RuntimeError(f"GTSim streaming runner failed with {returncode}")
    timing_line = next(line for line in (args.out / "simulation.log").read_text().splitlines()
                       if line.startswith("Program timing JSON: "))
    outcome["timing"] = json.loads(timing_line.split(": ", 1)[1])
    outcome["data_check"] = outcome["timing"].get(
        "data_check",
        {
            "checked_reads": 0,
            "checked_bytes": 0,
            "mismatched_bytes": 0,
            "logical_read_digest": "",
            "shadow_payload_mode": "disabled",
            "partition_oracle_pass": False,
        },
    )
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
    print(json.dumps({"version": "stream-v2", "batch": params.get("B"),
                      "program": plan.symbol, "returncode": returncode,
                      "cycles": outcome["timing"].get("total_cycles")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("frontend/asm/qwen3/one_block.progir"))
    parser.add_argument("--lib", type=Path, default=Path("frontend/lib"))
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--prefetch-config", type=Path)
    parser.add_argument("--flashaccel-config", type=Path)
    parser.add_argument("--prefetch", choices=("on", "off"))
    parser.add_argument("--tensor-layouts", type=Path)
    parser.add_argument("--tensor-placements", type=Path)
    parser.add_argument("--focus-launch")
    parser.add_argument("--focus-expert", type=int)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--max-frame", type=int, default=1 << 20)
    parser.add_argument("--verify-memory-reads", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.max_frame <= 0xFFFFFFFF:
        parser.error("--max-frame must be in [1, 4294967295]")
    params = json.loads(args.params.read_text())
    residency = json.loads(args.prefetch_config.read_text()) if args.prefetch_config else None
    raw_layouts = json.loads(args.tensor_layouts.read_text()) if args.tensor_layouts else {}
    placements = json.loads(args.tensor_placements.read_text()) if args.tensor_placements else {}
    layout_objects = {name: A.TileMajorLayout(**layout) for name, layout in raw_layouts.items()}
    flashaccel_policy = (
        PartitionPolicy.from_json(args.flashaccel_config)
        if args.flashaccel_config
        else None
    )
    physical_layout = (
        PhysicalLayout.from_memory_yaml(args.memory)
        if flashaccel_policy
        else None
    )
    source_program, regions, arch, prefetch, tensors = plan_program(
        args.source, args.lib, params, residency, layout_objects, args.focus_launch,
        args.focus_expert,
        placements,
        flashaccel_policy,
        physical_layout,
        args.prefetch != "off" and memory_topology(args.memory) != "FullHBM",
    )
    run_streaming_interface(args, params, source_program, regions, arch, prefetch, tensors,
                            layout_objects, residency, raw_layouts, placements,
                            flashaccel=flashaccel_policy is not None)


if __name__ == "__main__":
    main()
