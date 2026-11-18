#!/usr/bin/env python3
"""Compile and simulate one Experiment 002 Q projection schedule."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from compile.flow_to_arch import lower_flow
from compile.opt.partition import PhysicalLayout, _yaml_scalars
from compile.opt.prefetch_regions import build_regions
from compile.opt.q_projection import BM, Infeasible, ORDERINGS, QSchedule, specialize_q_projection
from compile.prog_to_flow import compile_program
from compile.static import EMPTY, Resolver, bound_resolver
from compile.streaming import _strides
from parser import ast as A, parse_file, serialize_file
from run_qwen_simulations import run_streaming_interface


SOURCE = Path("frontend/asm/qwen3/experiments/q_projection.progir")
LIBRARY = Path("frontend/lib")
ARCHITECTURE = Path("configs/v0.1.0/h100-sxm.yaml")
MEMORIES = {
    "fullhbm": Path("configs/memconf/0F5M_channels.yaml"),
    "4f1m": Path("configs/memconf/4F1M_flashaccel_32m.yaml"),
}


def compile_case(schedule, hardware, *, diagnostic=None):
    source = parse_file(SOURCE)
    params = dict(BN=schedule.bn, BK=schedule.bk, SPLIT_K=schedule.split_k)
    params.update(diagnostic or {})
    flow = compile_program(SOURCE, params, LIBRARY.glob("q_*.algoir"), LIBRARY.glob("q_*.flowir"))
    flow = specialize_q_projection(flow, schedule, hbf=hardware == "4f1m")
    arch = lower_flow(flow)
    root = bound_resolver(flow.static, Resolver(EMPTY, {})).materialize_constants()
    tensors = tuple(
        A.KernelTensor(i, tensor.id, shape, _strides(shape), tensor.dtype,
                       tensor.tier, tensor.external, tensor.read_only, None)
        for i, tensor in enumerate(flow.tensors)
        for shape in [tuple(root.value(d) for d in tensor.shape)]
    )
    lookahead = None
    if schedule.ahead:
        layout = PhysicalLayout.from_memory_yaml(MEMORIES[hardware])
        base = layout.bases(flow.tensors, root)["W_q"]
        capacity = int(_yaml_scalars(MEMORIES[hardware])["hbf.prefetch_engine.queue_capacity"])
        lookahead = build_regions(schedule, next(t for t in tensors if t.name == "W_q"),
                                  layout, base, capacity)
    summary = validate_native_contract(arch, tensors, root, schedule)
    summary["prefetch"] = lookahead.summary if lookahead else None
    return params, source, flow, arch, tensors, lookahead, summary


def region_resolver(arch, region):
    root = bound_resolver(arch.static, Resolver(EMPTY, {})).materialize_constants()
    arguments = {arg.id: root.value(arg.expression) for arg in region.static.arguments}
    return Resolver(region.static, {**root.values, **arguments}).materialize_constants()


def validate_native_contract(arch, tensors, root, schedule):
    by_name = {tensor.name: tensor for tensor in tensors}
    gemm = arch.regions[0]
    num_cta = min(root.value("NUM_SMS"), len(schedule.tasks))
    waves = (len(schedule.tasks) + num_cta - 1) // num_cta
    if gemm.resources.warps != 12 or gemm.registers.consumer_warps != 8 or (
            gemm.registers.fragment_shape != (16, 64)) or gemm.async_protocol.mbarrier_phases != 2:
        raise ValueError("native WS resources differ from the required geometry")
    n, k = root.value("N"), root.value("K")
    iterations = k // schedule.split_k // schedule.bk
    if k != iterations * schedule.split_k * schedule.bk:
        raise Infeasible("Split-K iterations do not exactly cover the weight rows")
    gemm_resolver = region_resolver(arch, gemm)
    visits = set()
    for cta in range(num_cta):
        for wave in range(waves):
            rank = wave * num_cta + cta
            point = gemm_resolver.at(dict(cta=cta, wave=wave))
            enabled = all(point.value(guard.condition) for guard in gemm.static_ifs
                          if guard.iterations)
            if enabled != (rank < len(schedule.tasks)):
                raise ValueError("native task guard differs from the persistent rank bound")
            if not enabled:
                continue
            split_id, tile_id = schedule.tasks[rank]
            col = point.value("%column")
            first = point.at(dict(k=0)).value("%k_row")
            last = point.at(dict(k=iterations - 1)).value("%k_row")
            if (point.value("%s"), col, first, last + schedule.bk) != (
                    split_id, tile_id * schedule.bn, split_id * k // schedule.split_k,
                    (split_id + 1) * k // schedule.split_k):
                raise ValueError("native addresses differ from the scheduled task rectangle")
            if not (0 <= col <= n - schedule.bn and 0 <= first <= last < k):
                raise Infeasible("native task has an out-of-bounds weight extent")
            visits.add((split_id, col))
    if len(visits) != len(schedule.tasks):
        raise ValueError("incorrect persistent CTA coverage")
    for region in arch.regions:
        resolver = region_resolver(arch, region)
        for op in region.ops:
            for step in op.steps:
                if step.opcode not in (A.NativeOpcode.TMA_LOAD, A.NativeOpcode.TMA_STORE):
                    continue
                tensor = by_name[step.region.tensor]
                height, width = (resolver.value(n) for n in step.region.shape)
                width_bytes = width * (4 if step.dtype == A.DataType.FP32 else 2)
                if not (1 <= height <= 256 and 1 <= width <= 256 and width_bytes % 16 == 0):
                    raise Infeasible(f"illegal TMA box {step.region.tensor}: {height}x{width}")
                if step.dtype != tensor.dtype or (tensor.strides[0] * (
                        4 if step.dtype == A.DataType.FP32 else 2)) % 16:
                    raise Infeasible(f"illegal TMA dtype or stride for {tensor.name}")
        if not 0 < region.resources.shared_bytes <= 227 * 1024:
            raise Infeasible(f"{region.symbol} exceeds shared memory per CTA")
    if len(arch.regions) != (2 if schedule.split_k > 1 else 1):
        raise ValueError("incorrect dependent reduction count")
    if schedule.split_k > 1 and (
            by_name["Partial"].shape != (schedule.split_k, root.value("N")) or
            arch.regions[1].after != (gemm.symbol,)):
        raise ValueError("incorrect reduction buffer or launch relationship")
    return dict(bm=BM, **asdict(schedule), warps=12, producer_warps=[8, 9, 10, 11],
                consumer_warps=list(range(8)), stages=2, matrix_tile=[64, 64, 16],
                compute_tasks=len(schedule.tasks), persistent_ctas=num_cta, waves=waves,
                k_iterations_per_task=iterations, smem_bytes=gemm.resources.shared_bytes,
                register_bytes=gemm.resources.register_bytes,
                partial_bytes=schedule.split_k * n * 4 if schedule.split_k > 1 else 0,
                tensor_layouts={t.name: dict(shape=t.shape, strides=t.strides,
                                             dtype=t.dtype.value, tier=t.tier.value)
                                for t in tensors})


def write_ir(output, flow, arch):
    serialize_file(flow, output / "specialized.flowir")
    serialize_file(arch, output / "specialized.nvir")


def simulate(args, schedule, compiled):
    params, source, _, arch, tensors, lookahead, summary = compiled
    simulation = SimpleNamespace(
        **vars(args), memory=MEMORIES[args.hardware], architecture=ARCHITECTURE, lib=LIBRARY,
        source=SOURCE, params=None, prefetch="on" if lookahead else "off",
        max_frame=1 << 20, max_cycles=10**9, focus_launch=None, focus_expert=None,
        context_length=None, gemm_bm=BM, gemm_bn=schedule.bn, gemm_bk=schedule.bk,
        gemm_stages=2, sram_policy=None)
    run_streaming_interface(
        simulation, params, source, arch.regions, arch, None, tensors, {}, None, {},
        {"W_q": "HBF" if args.hardware == "4f1m" else "HBM"}, compile_metadata=summary,
        producer_options=dict(prefetch_windows=False, lookahead_regions=(
            {arch.regions[0].symbol: lookahead} if lookahead else {})),
        provenance_extra=dict(candidate_fingerprint=args.fingerprint, hardware=args.hardware,
                              effective_flowir_parameters=summary))
    result = json.loads((args.out / "result.json").read_text())
    timing = result["timing"]
    if timing["total_cycles"] <= 0 or len(timing["per_kernel"]) != len(arch.regions):
        raise RuntimeError("missing validated simulation timing")
    gemm = timing["per_kernel"][0]
    if gemm["cta_count"] != summary["persistent_ctas"] or (
            len(gemm["prefetch_regions"]) != (len(lookahead.regions) if lookahead else 0)):
        raise RuntimeError("runtime did not execute the specialized FlowIR")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bn", type=int, required=True)
    parser.add_argument("--bk", type=int, required=True)
    parser.add_argument("--split-k", type=int, required=True)
    parser.add_argument("--ordering", choices=ORDERINGS, required=True)
    parser.add_argument("--pf-bn", type=int, default=0)
    parser.add_argument("--pf-bk", type=int, default=0)
    parser.add_argument("--ahead", type=int, default=0)
    parser.add_argument("--hardware", choices=MEMORIES, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fingerprint")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--save-ir", action="store_true")
    args = parser.parse_args()
    schedule = QSchedule(args.bn, args.bk, args.split_k, args.ordering,
                         args.pf_bn, args.pf_bk, args.ahead)
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "compile_summary.json"
    try:
        compiled = compile_case(schedule, args.hardware)
    except Infeasible as error:
        summary_path.write_text(json.dumps(dict(status="INFEASIBLE", reason=str(error),
                                                parameters=asdict(schedule)), indent=2) + "\n")
        print(f"Infeasible: {error}", flush=True)
        return 78
    _, _, flow, arch, _, _, summary = compiled
    summary_path.write_text(json.dumps(dict(status="SUCCESS", **summary), indent=2) + "\n")
    if args.save_ir:
        write_ir(args.out, flow, arch)
    if args.preflight:
        return 0
    simulate(args, schedule, compiled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
