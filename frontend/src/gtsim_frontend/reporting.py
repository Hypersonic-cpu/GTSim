"""Workload reporting over actual generic ProgramExecutor timing results."""

from dataclasses import asdict
import json
from pathlib import Path
from .ir.profiles import SMALL_M, DECODE


def configuration(algorithm):
    config = {
        "schedule_id": algorithm.schedule_id,
        "family": algorithm.family,
        "dtype": algorithm.dtype,
        "timing_calibrated": False,
    }
    if algorithm.family == "gemm":
        m, n, k = algorithm.dimensions
        config.update(
            m=m, n=n, k=k, tile_m=SMALL_M.bm, tile_n=SMALL_M.bn, tile_k=SMALL_M.bk
        )
        if algorithm.groups:
            counts = {str(g.identity): len(g.input_rows) for g in algorithm.groups}
            physical = sum(((count + 15) // 16) * 16 for count in counts.values())
            config.update(
                active_experts=len(counts),
                expert_token_counts=counts,
                assignments=m,
                physical_rows=physical,
                padded_rows=physical - m,
                row_utilization=m / physical,
                padding_amplification=physical / m,
                weight_bindings={str(g.identity): g.weight for g in algorithm.groups},
            )
        else:
            config.update(
                physical_rows=((m + 15) // 16) * 16,
                padded_rows=((m + 15) // 16) * 16 - m,
            )
    elif algorithm.family == "fa3":
        b, hq, hkv, _, _ = algorithm.dimensions
        config.update(
            query_heads=hq,
            kv_heads=hkv,
            gqa_ratio=hq // hkv,
            head_dim=algorithm.head_dim,
            effective_context_lengths=list(algorithm.valid_lengths),
            split_count=algorithm.split_count,
            kv_tile=DECODE.kv_tile,
            query_ctas=b
            * hq
            * (algorithm.split_count if algorithm.schedule_id == "decode_simt" else 1),
            normalization="stable_lse_weighted_merge"
            if algorithm.schedule_id == "decode_combine"
            else "streaming_online_softmax",
        )
    else:
        config.update(
            primitive=type(algorithm.primitive).__name__,
            parameters=asdict(algorithm.primitive),
        )
    return config


def decode_report(program, workload, out_dir, raw, launch_overhead_cycles=0):
    out = Path(out_dir)
    (out / "simulator-timing.json").write_text(json.dumps(raw, indent=2) + "\n")
    (out / "workload.json").write_text(json.dumps(workload, indent=2) + "\n")
    graphs = json.loads((out / "graph-summary.json").read_text())
    algorithms = {inv.id: inv.algorithm for inv in program.launches}
    kernels = []
    for timing in raw["per_kernel"]:
        meta = workload["kernel_metadata"][timing["id"]]
        record = {
            **meta,
            **timing,
            "duration_cycles": timing["complete_cycle"] - timing["start_cycle"],
            "device_duration_cycles": timing["device_complete_cycle"]
            - timing["start_cycle"],
            "configuration": configuration(algorithms[timing["id"]]),
            "graph": graphs[timing["id"]],
        }
        kernels.append(record)
    layers = []
    for layer in range(workload["layers"]):
        items = [k for k in kernels if k["layer"] == layer]
        layers.append(
            {
                "layer": layer,
                "start_cycle": items[0]["launch_cycle"],
                "end_cycle": items[-1]["complete_cycle"],
                "duration_cycles": items[-1]["complete_cycle"]
                - items[0]["launch_cycle"],
                **workload["layer_summaries"][layer],
            }
        )
    timeline = {
        key: workload[key]
        for key in (
            "model",
            "mode",
            "batch_size",
            "layers",
            "context_lengths",
            "effective_context_lengths",
            "routing_source",
            "routing_seed",
        )
    }
    timeline.update(
        stream={
            "id": 0,
            "start_cycle": 0,
            "end_cycle": raw["total_cycles"],
            "duration_cycles": raw["total_cycles"],
            "launch_overhead_cycles_per_kernel": launch_overhead_cycles,
        },
        kernels=kernels,
        layers_summary=layers,
        layer_work_summary=workload["layer_summaries"],
        memory_statistics={
            "program_traffic_bytes": raw["traffic_bytes"],
            **{
                k: raw[k]
                for k in (
                    "persistent_gpu",
                    "persistent_memory",
                    "persistent_l2",
                    "stable_allocations",
                )
            },
        },
        outcome="PASS",
    )
    for name in ("kernel-timings.json", "stream-timeline.json"):
        (out / name).write_text(json.dumps(timeline, indent=2) + "\n")
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "program_cycles": raw["total_cycles"],
                "layers": layers,
                "memory_statistics": timeline["memory_statistics"],
                "kernels": kernels,
                "outcome": "PASS",
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        f"# {workload['model']} corrected single decode step",
        "",
        f"- Batch: {workload['batch_size']}; layers: {workload['layers']}",
        f"- Prior context lengths: {workload['context_lengths']}",
        f"- Effective KV lengths: {workload['effective_context_lengths']}",
        f"- Stream global cycles: 0–{raw['total_cycles']}; configured launch overhead per kernel: {launch_overhead_cycles}",
        f"- HBM read/write bytes: {raw['traffic_bytes']['HBM_read']} / {raw['traffic_bytes']['HBM_write']}",
        f"- HBF read/write bytes: {raw['traffic_bytes']['HBF_read']} / {raw['traffic_bytes']['HBF_write']}",
        "",
        "| # | Layer | Kernel | Variant | Launch | Start | Device done | Drain complete | Duration | HBM read | HBM write |",
        "|--:|--:|---|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for i, k in enumerate(kernels):
        lines.append(
            f"| {i} | {k['layer']} | {k['name']} | {k['variant']} | {k['launch_cycle']} | {k['start_cycle']} | {k['device_complete_cycle']} | {k['complete_cycle']} | {k['duration_cycles']} | {k['hbm_read_bytes']} | {k['hbm_write_bytes']} |"
        )
    for layer in layers:
        lines.extend(
            [
                "",
                f"## Layer {layer['layer']}",
                "",
                f"- Global range: {layer['start_cycle']}–{layer['end_cycle']}; duration: {layer['duration_cycles']} cycles.",
                f"- {layer['new_tokens']} new tokens; output shape {layer['layer_output_shape']}.",
                f"- Attention: {layer['fa3_variant']}; splits: {layer['fa3_split_count']}; query CTAs: {layer['fa3_query_ctas']}.",
                f"- MoE: {layer['moe_variant']}; active experts: {layer['active_experts']}; assignments: {layer['expert_assignment_count']}.",
                f"- Physical rows per expert GEMM stage: {layer['physical_rows']}; padded rows: {layer['padded_rows']}; row utilization: {layer['row_utilization']}; amplification: {layer['padding_amplification']}.",
                f"- Per-expert counts: {layer['expert_token_counts']}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Fidelity limits",
            "",
            "- The new Small-M, SIMT Decode Attention, vector and grouped Small-M MMA baseline timings are uncalibrated H100 estimates. They do not inherit the reference GEMM/FA3 accuracy figures.",
            "- Numerical oracles interpret typed Native math separately; GTSim itself executes timing and memory operations, not numerical Transformer inference.",
            "- KV storage is contiguous per request/head with exact valid regions; Paged KV addressing is unsupported.",
            "- Routing is a supplied or seeded synthetic Top-8 trace. Router/TopK and assignment kernels remain timed. Expert weight accesses occur only after those ordered launches.",
            "- No prefill, extra generation steps, sampling, LM head, prefetch, caching, DSE, or concurrent streams.",
            "",
        ]
    )
    (out / "results.md").write_text("\n".join(lines))
    return timeline
