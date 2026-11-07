"""Run a textual Qwen program through ProgramExecutor, streaming via FIFOs.

Only manifests, provenance and timing logs are stored. Each kernel is expanded
once in RAM; the existing C++ reader consumes its JSON from a named pipe.
"""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
from time import perf_counter
import traceback

from compile import compile_program, lower_flow
from compile.arch_to_exec import lower_region, strides
from compile.static import EMPTY, Resolver, bound_resolver
from compile.export_runtime import PIPELINES, tensor_record
from parser import ast as A


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def isolate_launch(flow, root, name, expert, params):
    """Select a cold launch or one expert's original CTA range for measurement."""
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
    # Retain every allocation so the selected weights keep their full-block
    # base addresses and HBF group placement.
    return replace(flow, regions=(replace(region, after=()),), outputs=())


def plan_program(
    source, lib, params, residency=None, layouts=None, focus=None, expert=None
):
    layouts = layouts or {}
    flow = compile_program(
        source, params, list(lib.glob("*.algoir")), list(lib.glob("*.flowir"))
    )
    root = bound_resolver(flow.static, Resolver(EMPTY, {})).materialize_constants()
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
    prefetch = None
    if residency is not None:
        from compile.opt.prefetch import Residency, optimize

        prefetch = optimize(flow, Residency(**residency), layouts)
        flow = prefetch.flow
    arch = lower_flow(flow)
    root = bound_resolver(arch.static, Resolver(EMPTY, {})).materialize_constants()
    tensors = []
    for spec in arch.tensors:
        shape = tuple(root.value(d) for d in spec.shape)
        tensors.append(
            A.KernelTensor(
                len(tensors),
                spec.id,
                shape,
                strides(shape),
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
    launches = [
        dict(
            record="ExecutionLaunch",
            id=r.symbol,
            kernel_file=r.symbol + ".fifo",
            bindings=[[t.id, t.id] for t in tensors],
        )
        for r in regions
    ]
    manifest = dict(
        record="ExecProgram",
        format="gtsim.exec.program/1",
        schema_version=1,
        symbol=arch.symbol,
        source_hash="",
        launch_overhead_cycles=0,
        streams=[arch.stream],
        tensors=[tensor_record(t) for t in tensors],
        launches=launches,
        commands=[
            dict(record="Command", kind="LaunchKernel", target=r.symbol)
            for r in regions
        ]
        + [dict(record="Command", kind="StreamSynchronize", target="")],
    )
    if prefetch is not None:
        prefetch.decorate(manifest)

    def expand(region):
        kernel = lower_region(arch, region)
        if layouts:
            kernel = replace(
                kernel,
                tensors=tuple(
                    replace(t, tile_major=layouts.get(t.name)) for t in kernel.tensors
                ),
            )
        if prefetch is not None and prefetch.streaming is not None:
            from compile.opt.panels import lower_panels

            kernel = lower_panels(kernel, prefetch.streaming, prefetch.bases)
        return kernel

    return manifest, regions, expand


def write_kernel(kernel, output):
    """Stream instructions without building another giant list of dictionaries."""
    digest = hashlib.sha256()
    counts = Counter()
    areas = Counter()
    nodes = 0
    buffer = []
    buffer_bytes = 0

    def write(text):
        digest.update(text.encode())
        output.write(text)

    def flush():
        nonlocal buffer_bytes
        if buffer:
            write("".join(buffer))
            buffer.clear()
            buffer_bytes = 0

    write('{"ops":[')
    for node in kernel.ops:
        opcode = node.opcode
        fields = dict(
            record="ArchInstruction",
            id=node.id,
            name=node.name,
            op=opcode,
            pipeline=PIPELINES[node.resource],
            cta=node.cta,
            warp=node.warp,
            tensor=node.tensor if node.tensor is not None else 0,
            dtype=node.dtype.value,
            leading_dim=node.leading_dim or 0,
            tile=node.tile,
            after_issue=node.after_issue,
            after_complete=node.after_complete,
        )
        text = ("," if nodes else "") + json.dumps(fields, separators=(",", ":"))
        buffer.append(text)
        buffer_bytes += len(text)
        if buffer_bytes >= 1024 * 1024:
            flush()
        nodes += 1
        counts[opcode] += 1
        if opcode == "cp.sram2dram_tma":
            areas[node.tensor] += node.tile[2] * node.tile[3]
    flush()
    write("]")
    tensors = [tensor_record(t) for t in kernel.tensors]
    outputs = kernel.regions[0].outputs
    partial = [
        t.id
        for t in kernel.tensors
        if t.id in outputs and areas[t.id] < t.shape[0] * t.shape[1]
    ]
    header = dict(
        record="ExecKernel",
        format="gtsim.exec.kernel/1",
        schema_version=1,
        symbol=kernel.symbol,
        source_hash="",
        target="sm90",
        sync_semantics_version="",
        tensors=tensors,
        outputs=outputs,
        partial_outputs=partial,
        ctas=[
            dict(
                record="CTA",
                id=c.id,
                sm=c.sm,
                warps=c.warps,
                shared_bytes=c.shared_bytes,
                register_bytes=c.register_bytes,
            )
            for c in kernel.ctas
        ],
    )
    if kernel.prefetch_windows:
        header["prefetch_windows"] = [asdict(w) for w in kernel.prefetch_windows]
    for key, value in header.items():
        write("," + json.dumps(key) + ":" + json.dumps(value, separators=(",", ":")))
    write("}\n")
    return dict(
        nodes=nodes,
        ctas=len(kernel.ctas),
        sms=len({c.sm for c in kernel.ctas}),
        shared_bytes_per_cta=sorted({c.shared_bytes for c in kernel.ctas}),
        register_bytes_per_cta=sorted({c.register_bytes for c in kernel.ctas}),
        opcodes=dict(counts),
        dag_sha256=digest.hexdigest(),
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
    parser.add_argument("--tensor-layouts", type=Path)
    parser.add_argument("--focus-launch")
    parser.add_argument("--focus-expert", type=int)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    params = json.loads(args.params.read_text())
    start = perf_counter()
    residency = (
        json.loads(args.prefetch_config.read_text()) if args.prefetch_config else None
    )
    layouts = json.loads(args.tensor_layouts.read_text()) if args.tensor_layouts else {}
    manifest, plans, expand = plan_program(
        args.source,
        args.lib,
        params,
        residency,
        {name: A.TileMajorLayout(**layout) for name, layout in layouts.items()},
        args.focus_launch,
        args.focus_expert,
    )
    for launch in manifest["launches"]:
        path = args.out / launch["kernel_file"]
        if path.exists():
            raise FileExistsError(path)
        os.mkfifo(path)
    (args.out / "program.gtsir").write_text(json.dumps(manifest, indent=2) + "\n")
    source_paths = sorted(
        [
            *Path("frontend/src").rglob("*.py"),
            *args.lib.glob("*ir"),
            args.source,
            args.params,
        ]
    )
    provenance = dict(
        version="new",
        params=params,
        residency=residency,
        tensor_layouts=layouts,
        focus_launch=args.focus_launch,
        focus_expert=args.focus_expert,
        launch_overhead_cycles=0,
        runner_sha256=sha(args.runner),
        max_cycles=args.max_cycles,
        architecture_sha256=sha(args.architecture),
        memory_sha256=sha(args.memory),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256={str(path): sha(path) for path in source_paths},
        transport="POSIX named pipes; no kernel DAG files persisted",
    )
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    log = (args.out / "simulation.log").open("w")
    process = subprocess.Popen(
        [
            str(args.runner),
            str(args.architecture),
            "--memory-config",
            str(args.memory),
            str(args.out / "program.gtsir"),
            *(["--max-cycles", str(args.max_cycles)] if args.max_cycles else []),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    records = []
    failures = []

    def produce():
        try:
            for plan, launch in zip(plans, manifest["launches"]):
                # Opening blocks until ProgramExecutor requests this launch.
                # No next-kernel expansion while the GPU is executing this one.
                with (args.out / launch["kernel_file"]).open(
                    "w", buffering=1024 * 1024
                ) as output:
                    at = perf_counter()
                    kernel = expand(plan)
                    expanded = perf_counter()
                    record = write_kernel(kernel, output)
                    record.update(
                        id=launch["id"],
                        expand_seconds=expanded - at,
                        pipe_seconds=perf_counter() - expanded,
                    )
                records.append(record)
                del kernel
                gc.collect()
        except BaseException:
            failures.append(traceback.format_exc())
            process.terminate()

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    returncode = process.wait()
    worker.join(timeout=1)
    log.close()
    outcome = dict(
        returncode=returncode,
        elapsed_seconds=perf_counter() - start,
        kernels=records,
        errors=failures,
    )
    if returncode == 0:
        line = next(
            line
            for line in (args.out / "simulation.log").read_text().splitlines()
            if line.startswith("Program timing JSON: ")
        )
        timing = json.loads(line.split(": ", 1)[1])
        outcome["timing"] = timing
        assert len(timing["per_kernel"]) == len(plans)
        assert all(k["state"] == "COMPLETED" for k in timing["per_kernel"])
        assert all(
            timing[key]
            for key in (
                "persistent_gpu",
                "persistent_memory",
                "persistent_l2",
                "stable_allocations",
            )
        )
    (args.out / "result.json").write_text(json.dumps(outcome, indent=2) + "\n")
    for launch in manifest["launches"]:
        (args.out / launch["kernel_file"]).unlink(missing_ok=True)
    print(
        json.dumps(
            dict(
                version="new",
                batch=params["B"],
                returncode=returncode,
                cycles=outcome.get("timing", {}).get("total_cycles"),
            )
        ),
        flush=True,
    )
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
