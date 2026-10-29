#!/usr/bin/env python3
"""Audit client TPOT and complete HostRAM layer windows, retaining all gaps.

Raw profiler/request files stay outside the repository. Only derived timing
summaries and file hashes are exported. Client TPOT is never replaced with a
sum of kernel durations or with a sum of simulated components.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


GDN = "fused_recurrent_gated_delta_rule_packed_decode_kernel"
FA2 = "flash_fwd_splitkv_kernel"
NORM = "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def describe(values):
    ordered = sorted(values)
    require(bool(ordered), "Empty timing sample")

    def percentile(p):
        index = (len(ordered) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return dict(count=len(values), mean=statistics.mean(values),
                p50=percentile(.5), p95=percentile(.95),
                minimum=ordered[0], maximum=ordered[-1])


def interval_union_ns(intervals):
    """Count busy time once, including when intervals overlap."""
    ordered = sorted(intervals)
    require(bool(ordered), "Empty kernel window")
    start, end = ordered[0]
    require(end >= start, "Negative interval")
    total = 0
    for next_start, next_end in ordered[1:]:
        require(next_end >= next_start, "Negative interval")
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def read_client(folder):
    metrics = json.loads((folder / "trace-metrics.json").read_text())
    times = []
    for line in (folder / "trace-events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if any(choice.get("text") for choice in event["event"].get("choices", [])):
            times.append(event["monotonic_ns"])
    count = metrics["usage"]["completion_tokens"]
    require(metrics["success"] and metrics["done"], "Request did not finish")
    require(len(times) == count and count > 1, "Output event/token coverage mismatch")
    require(all(b >= a for a, b in zip(times, times[1:])), "Non-monotonic SSE events")
    require(times[0] == metrics["first_text_monotonic_ns"] and
            times[-1] == metrics["last_text_monotonic_ns"], "Metric/event endpoints differ")
    intervals = [(b - a) / 1e6 for a, b in zip(times, times[1:])]
    tpot = (times[-1] - times[0]) / (count - 1) / 1e6
    require(math.isclose(tpot, metrics["tpot_ms"], abs_tol=1e-9), "Stored TPOT mismatch")
    request = json.loads((folder / "trace-request.json").read_text())
    require(len(request["prompt"]) == metrics["usage"]["prompt_tokens"] == 32768,
            "Expected 32K input")
    require(request["max_tokens"] == count == 256, "Expected 256 output tokens")
    return dict(tpot_ms=tpot, decode_elapsed_ms=(times[-1] - times[0]) / 1e6,
                output_tokens=count, intervals=count - 1,
                inter_token_latency_ms=describe(intervals),
                ttft_s=metrics["ttft_s"], first_text_monotonic_ns=times[0],
                last_text_monotonic_ns=times[-1],
                definition="(last nonempty SSE arrival - first nonempty SSE arrival) / 255",
                includes="All intervening compute, exposed transfers, waits, scheduling and serving latency"), request


def audit_layers(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("start_ns", "end_ns", "duration_ns"):
            row[key] = int(row[key])
        require(row["end_ns"] - row["start_ns"] == row["duration_ns"], "Bad kernel duration")
    gdn_indices = [i for i, row in enumerate(rows) if row["short_name"] == GDN]
    require(len(gdn_indices) == 24 * 255, "Incomplete GDN decode capture")
    first = gdn_indices[0] - 6
    anchors = []
    for i, row in enumerate(rows):
        if i < first:
            continue
        if row["short_name"] == GDN:
            anchors.append(("GDN", i - 6, i + 6))
        elif row["short_name"] == FA2:
            anchors.append(("GQA", i - 11, i + 8))
    require(len(anchors) == 32 * 255, "Incomplete 32-layer decode capture")
    identities = {(r["device_id"], r["stream_id"], r["global_pid"]) for r in rows[first:]}
    require(len(identities) == 1, "Layer attribution requires this single compute stream capture")
    previous_end = None
    layers = {"GQA": [], "GDN": []}
    steps = []
    for step in range(255):
        chunk = anchors[step * 32:(step + 1) * 32]
        require([x[0] for x in chunk] == ["GDN", "GDN", "GDN", "GQA"] * 8,
                "Unexpected model layer order")
        require(rows[chunk[0][1] - 1]["short_name"] == "indexSelectSmallIndex",
                "Decode block must follow input embedding kernel")
        block_start = rows[chunk[0][1] - 1]["end_ns"]
        by_type = {"GQA": 0, "GDN": 0}
        busy = 0
        for layer, (kind, start, end) in enumerate(chunk):
            require(rows[start]["short_name"] == NORM, "Layer start is not input RMSNorm")
            require(rows[end]["short_name"].startswith("ampere_bf16_s16816gemm"),
                    "Layer end is not MLP down projection")
            if layer:
                require(start == previous_end + 1, "Unassigned kernels between layers")
            previous_end = end
            entry = rows[start - 1]["end_ns"]
            exit_time = rows[end]["end_ns"]
            require(entry <= rows[start]["start_ns"], "Overlapping layer boundary")
            wall = exit_time - entry
            active = interval_union_ns([(r["start_ns"], r["end_ns"]) for r in rows[start:end + 1]])
            require(0 <= active <= wall, "Invalid wall/busy decomposition")
            layers[kind].append(dict(wall_ns=wall, busy_ns=active,
                                     non_kernel_ns=wall - active,
                                     entry_gap_ns=rows[start]["start_ns"] - entry))
            by_type[kind] += wall
            busy += active
        block_end = rows[chunk[-1][2]]["end_ns"]
        require(sum(by_type.values()) == block_end - block_start, "Layer windows do not close")
        record = dict(step=step + 1, gqa_layers_wall_ms=by_type["GQA"] / 1e6,
                      gdn_layers_wall_ms=by_type["GDN"] / 1e6,
                      all_32_layers_wall_ms=(block_end - block_start) / 1e6,
                      layer_kernels_busy_ms=busy / 1e6,
                      layer_non_kernel_ms=(block_end - block_start - busy) / 1e6,
                      gpu_step_period_ms=None, outside_32_layers_ms=None)
        if step < 254:
            next_start = rows[anchors[(step + 1) * 32][1] - 1]["end_ns"]
            require(next_start >= block_end, "Invalid step order")
            record["gpu_step_period_ms"] = (next_start - block_start) / 1e6
            record["outside_32_layers_ms"] = (next_start - block_end) / 1e6
        steps.append(record)
    summary = {}
    for kind, samples in layers.items():
        summary[kind] = {key.replace("_ns", "_ms"): describe([s[key] / 1e6 for s in samples])
                         for key in samples[0]}
    summary["all_32_layers_wall_ms"] = describe([s["all_32_layers_wall_ms"] for s in steps])
    summary["closed_gpu_step_windows"] = dict(
        count=254,
        period_ms=statistics.mean(s["gpu_step_period_ms"] for s in steps[:-1]),
        layers_wall_ms=statistics.mean(s["all_32_layers_wall_ms"] for s in steps[:-1]),
        outside_layers_ms=statistics.mean(s["outside_32_layers_ms"] for s in steps[:-1]),
        definition="Input embedding completion to next input embedding completion",
        note="Separate GPU anchors and 254 samples; not the 255 client token intervals")
    summary["layer_window_definition"] = (
        "Previous layer MLP down completion to current layer MLP down completion; "
        "first layer starts at input embedding completion. Includes entry and internal waits.")
    summary["non_kernel_attribution"] = (
        "Time without a recorded compute kernel, including exposed transfer and host/sync/queue stalls. "
        "Kernel CSV alone cannot separate these causes. Overlapped copies are not added twice.")
    return summary, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sim-components", type=Path,
                        help="Matched normal/hbf simulated components for ratio-calibrated TPOT")
    args = parser.parse_args()
    folders = dict(hostram="Qwen3.5-9B-batch1-HostRAM", normal="Qwen3.5-9B-batch1-NoOff")
    clients, requests, provenance = {}, {}, []
    for kind, folder in folders.items():
        path = args.data_root / folder
        clients[kind], requests[kind] = read_client(path)
        names = ["trace-metrics.json", "trace-events.jsonl", "trace-request.json", "launch-command.sh"]
        if kind == "hostram":
            names.append("trace-kernels.csv")
        for name in names:
            content = (path / name).read_bytes()
            provenance.append(dict(file=f"{folder}/{name}", bytes=len(content),
                                   sha256=hashlib.sha256(content).hexdigest()))
    require(requests["hostram"] == requests["normal"], "Requests differ")
    layers, steps = audit_layers(args.data_root / folders["hostram"] / "trace-kernels.csv")
    result = dict(schema_version=1, client_measurements=clients,
                  hostram_over_normal_tpot_ratio=clients["hostram"]["tpot_ms"] / clients["normal"]["tpot_ms"],
                  hostram_layer_wall_audit=layers,
                  comparison=dict(
                      GQA_layer_ms=dict(HostRAM=layers["GQA"]["wall_ms"]["mean"], normal=None, hbf=None),
                      GDN_layer_ms=dict(HostRAM=layers["GDN"]["wall_ms"]["mean"], normal=None, hbf=None),
                      decode_step_tpot_ms=dict(HostRAM=clients["hostram"]["tpot_ms"],
                                               normal=clients["normal"]["tpot_ms"], hbf=None)),
                  missing="NoOff per-layer wall trace and full 3M2F serving model have not been produced",
                  conditions=dict(
                      common="Same request, batch 1, BF16, input 32768, output 256, eager execution",
                      hostram="MLP CPU offload + prefetch, 10 GiB KV cache",
                      normal="No CPU offload, 5 GiB KV cache",
                      limitations="Separate single runs on shared A100; measured ratio is not an isolated PCIe penalty"),
                  provenance=provenance)
    if args.sim_components:
        components = json.loads(args.sim_components.read_text())
        simulated = {}
        for kind in ("normal", "hbf"):
            # Recompute the total from the model's actual layer counts rather
            # than trusting the historical decode_component_sum field.
            simulated[kind] = (8 * components[kind]["GQA"] +
                               24 * components[kind]["GDN"]) / 1000
            require(simulated[kind] > 0, "Non-positive simulation time")
        ratio = simulated["hbf"] / simulated["normal"]
        estimate = clients["normal"]["tpot_ms"] * ratio
        result["hybrid_tpot_estimate"] = dict(
            architecture="3M2F: 3 HBM + 2 HBF stacks",
            method="ratio calibration to measured NoOff client TPOT",
            formula="TPOT_hybrid_est = TPOT_normal_measured * T_sim_hybrid / T_sim_normal",
            normal_simulated_components_ms=simulated["normal"],
            hybrid_simulated_components_ms=simulated["hbf"],
            simulated_ratio=ratio,
            normal_measured_tpot_ms=clients["normal"]["tpot_ms"],
            calibrated_tpot_ms=estimate,
            hostram_over_hybrid_estimate=clients["hostram"]["tpot_ms"] / estimate,
            scope="8 GQA (projections + MLP + FA2) and 24 GDN (four projections); matched coverage",
            assumption="The complete serving TPOT scales with the covered simulation time ratio",
            limitation="Auxiliary kernels, LM head and host/scheduler overhead are not explicitly simulated; "
                       "their scaling is assumed. This is a model-based estimate, not measured or full-step simulated TPOT.")
        result["comparison"]["decode_step_tpot_ms"]["hbf"] = estimate
        result["missing"] = "NoOff per-layer wall trace and full 3M2F serving model; hybrid TPOT is ratio-estimated"
        provenance.append(dict(file=args.sim_components.name,
                               bytes=args.sim_components.stat().st_size,
                               sha256=hashlib.sha256(args.sim_components.read_bytes()).hexdigest()))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "wall-time-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "hostram-step-wall-times.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=steps[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(steps)
    print(json.dumps(dict(client_measurements=clients,
                          ratio=result["hostram_over_normal_tpot_ratio"],
                          comparison=result["comparison"],
                          closed_gpu_step_windows=layers["closed_gpu_step_windows"],
                          hybrid_tpot_estimate=result.get("hybrid_tpot_estimate")), indent=2))


if __name__ == "__main__":
    main()
