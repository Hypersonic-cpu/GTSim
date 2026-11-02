"""Compare text and legacy IR in RAM, releasing each DAG before the next.

Only the compact report is serialized. This module is an independent oracle;
the production compiler never imports the legacy frontend.
"""

import argparse
from bisect import bisect_right
from collections import Counter
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import re
import struct
from time import perf_counter
from compile import compile_program, lower_flow
from compile.arch_to_exec import lower_region
from parser import ast as A, parse_file
from gtsim_frontend.programs.qwen3_decode import build as legacy_program
from gtsim_frontend.passes.instantiate import instantiate
from gtsim_frontend.passes.hopper.lower import lower as old_native
from gtsim_frontend.passes.lower_execution import lower as old_expand


def canonical_name(name):
    if name.startswith("l0_"):
        name = name[3:]
    if name.endswith("_direct"):
        name = name[:-7]
    if name.endswith("_grouped"):
        name = name[:-8]
    return {"fa3_decode_split": "fa3_decode", "kv_cache_update": "kv_cache_update"}.get(
        name, name
    )


def snapshot(kernel, params, formal_names, is_new, gemm):
    tensors = {t.id: t for t in kernel.tensors}
    prefixes = [0]
    for length in params["CONTEXTS"]:
        prefixes.append(prefixes[-1] + (length + 1) * params["HKV"])
    memory = Counter()
    counts = Counter()
    read_bytes = write_bytes = 0
    native = hashlib.blake2b()
    complete = hashlib.blake2b()
    issue = hashlib.blake2b()

    def update_corner(key, delta):
        memory[key] += delta
        if memory[key] == 0:
            del memory[key]

    for node in kernel.ops:
        opcode = node.opcode if is_new else node.op
        counts[opcode] += 1
        r, c, h, w = node.tile
        role = "register"
        if node.tensor is not None and node.tensor in tensors:
            tensor = tensors[node.tensor]
            role = formal_names.get(tensor.name, tensor.name)
        if opcode in ("cp.dram2sram_tma", "cp.sram2dram_tma"):
            if is_new and tensor.name in ("K_cache", "V_cache"):
                request = bisect_right(prefixes, r) - 1
                length = params["CONTEXTS"][request] + 1
                head = (r - prefixes[request]) // length
                position = (r - prefixes[request]) % length
                assert position + h <= length
                role = (tensor.name[0], request, head)
                r = position
            else:
                cache = re.fullmatch(r"([KV])_r(\d+)_h(\d+)", role)
                if cache:
                    role = (cache[1], int(cache[2]), int(cache[3]))
            if role == "Knew":
                role = "Key"
            if role == "Vnew":
                role = "Value"
            if gemm and gemm.get("grouped") and role == "B":
                expert, r = divmod(r, gemm["K"])
                role = f"W_e{expert}"
            key = (
                opcode,
                str(role),
                node.dtype.value if hasattr(node.dtype, "value") else node.dtype,
            )
            for row, col, delta in (
                (r, c, 1),
                (r + h, c, -1),
                (r, c + w, -1),
                (r + h, c + w, 1),
            ):
                update_corner((*key, row, col), delta)
            byte_count = h * w * (4 if node.dtype in ("fp32", "int32") else 2)
            if opcode == "cp.dram2sram_tma":
                read_bytes += byte_count
            else:
                write_bytes += byte_count
        elif gemm:
            # Shared/accumulator rows are CTA-local; global output identity is
            # checked independently through the actual global output stores.
            r = 0
            role = "C"
        native.update(opcode.encode() + b"\0" + str(role).encode() + b"\0")
        native.update(
            struct.pack(
                "8q",
                node.cta,
                node.warp,
                r,
                c,
                h,
                w,
                node.leading_dim or 0,
                0 if node.dtype == "bf16" else 1,
            )
        )
        for digest, edges in (
            (complete, node.after_complete),
            (issue, node.after_issue),
        ):
            edges = sorted(set(edges))
            digest.update(
                struct.pack(f"{len(edges) + 2}q", node.id, len(edges), *edges)
            )
    return dict(
        nodes=len(kernel.ops),
        ctas=len(kernel.ctas),
        resources=tuple(
            (c.sm, c.warps, c.shared_bytes, c.register_bytes) for c in kernel.ctas
        ),
        opcodes=dict(counts),
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        memory=memory,
        native_hash=native.hexdigest(),
        complete_hash=complete.hexdigest(),
        issue_hash=issue.hexdigest(),
    )


def run(source, lib, parameters):
    start = perf_counter()
    flow = compile_program(
        source, parameters, list(lib.glob("*.algoir")), list(lib.glob("*.flowir"))
    )
    arch = lower_flow(flow)
    trace = [
        [
            dict(
                expert_ids=parameters["ROUTE_IDS"][
                    i * parameters["TK"] : (i + 1) * parameters["TK"]
                ],
                probabilities=parameters["ROUTE_PROBS"][
                    i * parameters["TK"] : (i + 1) * parameters["TK"]
                ],
            )
            for i in range(parameters["B"])
        ]
    ]
    program, metadata = legacy_program(
        parameters["B"], 1, parameters["CONTEXTS"], parameters["S"], routing_trace=trace
    )
    legacy = {canonical_name(inv.id): inv for inv in program.launches}
    results = []
    for region in arch.regions:
        before = perf_counter()
        kernel = lower_region(arch, region)
        if not kernel.ops:
            continue
        name = canonical_name(region.symbol)
        invocation = legacy[name]
        algorithm = invocation.algorithm
        is_gemm = algorithm.family == "gemm"
        gemm = (
            dict(grouped=bool(algorithm.groups), K=algorithm.dimensions[2])
            if is_gemm
            else None
        )
        formal = parse_file(lib / (region.origin + ".algoir"))
        names = {
            tensor.id: definition.id
            for tensor, definition in zip(region.tensors, formal.tensors)
        }
        current = snapshot(kernel, parameters, names, True, gemm)
        new_seconds = perf_counter() - before
        del kernel
        gc.collect()
        before = perf_counter()
        kernel = old_expand(old_native(instantiate(algorithm, parameters["NUM_SMS"])))
        previous = snapshot(kernel, parameters, {}, False, gemm)
        old_seconds = perf_counter() - before
        del kernel
        gc.collect()
        memory_equal = current.pop("memory") == previous.pop("memory")
        native_equal = current["native_hash"] == previous["native_hash"]
        record = dict(
            id=name,
            new_nodes=current["nodes"],
            old_nodes=previous["nodes"],
            memory_regions_equal=memory_equal,
            read_bytes_equal=current["read_bytes"] == previous["read_bytes"],
            write_bytes_equal=current["write_bytes"] == previous["write_bytes"],
            normalized_native_equal=native_equal,
            complete_edges_equal=current["complete_hash"] == previous["complete_hash"],
            issue_edges_equal=current["issue_hash"] == previous["issue_hash"],
            resources_equal=current["resources"] == previous["resources"],
            new_seconds=round(new_seconds, 3),
            old_seconds=round(old_seconds, 3),
            new=current,
            old=previous,
        )
        results.append(record)
        print(
            json.dumps({k: v for k, v in record.items() if k not in ("new", "old")}),
            flush=True,
        )
    assert [r["id"] for r in results] == list(legacy)
    return dict(
        batch=parameters["B"],
        kernel_count=len(results),
        total_new_nodes=sum(r["new_nodes"] for r in results),
        total_old_nodes=sum(r["old_nodes"] for r in results),
        seconds=round(perf_counter() - start, 3),
        kernels=results,
        assignments=parameters["B"] * parameters["TK"],
        active_experts=len(set(parameters["ROUTE_IDS"])),
        physical_expert_rows=sum(
            ((parameters["ROUTE_IDS"].count(e) + 15) // 16) * 16
            for e in set(parameters["ROUTE_IDS"])
        ),
        memory_regions_all_equal=all(r["memory_regions_equal"] for r in results),
        scope="per-kernel in-memory normalized native graph/edge/memory comparison; source Program preserves one ordered full-block stream",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--lib", type=Path, required=True)
    p.add_argument("--params", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    result = run(a.source, a.lib, json.loads(a.params.read_text()))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "kernels"}))


if __name__ == "__main__":
    main()
