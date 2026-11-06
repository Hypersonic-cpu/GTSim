"""Summarize seeded Qwen runs and plot measured HBF completion bandwidth."""

import argparse
import csv
from collections import Counter
import json
from math import ceil
from pathlib import Path
from run_flash_accel_batches import experiment_modes


def write_csv(path, records):
    if records:
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)


def sram_capacity(root):
    r = json.loads((root / "inputs/prefetch.json").read_text())
    return r["stacks"] * (
        r["dies_per_stack"] * r["planes_per_die"] * r["plane_bytes"] + r["stack_bytes"]
    )


def validate_buffer_plan(root, case, capacity):
    program = json.loads((root / case / "program.gtsir").read_text())
    reservations = {p["id"]: p for p in program.get("prefetches", [])}
    groups = {}
    for p in reservations.values():
        if buffer := p.get("buffer"):
            assert buffer["epoch"] % buffer["count"] == buffer["slot"]
            groups.setdefault(buffer["epoch"], []).append(p)
    for items in groups.values():
        count = items[0]["buffer"]["count"]
        assert sum(ceil(p["bytes"] / 4096) * 4096 for p in items) <= capacity // count
    released = set()
    for command in program["commands"]:
        if command["kind"] == "Release":
            released.add(command["target"])
        elif command["kind"] == "QueuePrefetch":
            item = reservations[command["target"]]
            if buffer := item.get("buffer"):
                prior = buffer["epoch"] - buffer["count"]
                if prior >= 0:
                    assert all(p["id"] in released for p in groups[prior])


def collect(root):
    summary, kernels, series, windows, buffers, timing = [], [], [], [], [], {}
    signatures = {}
    capacity = sram_capacity(root)
    config = json.loads((root / "inputs/experiment.json").read_text())
    modes = experiment_modes(config.get("buffer_counts", [1]))
    for batch in (4, 8):
        trace = json.loads((root / f"inputs/b{batch}-routing.json").read_text())
        for mode in modes:
            case = f"b{batch}-{mode}"
            result = json.loads((root / case / "result.json").read_text())
            assert result["returncode"] == 0, case
            validate_buffer_plan(root, case, capacity)
            signature = Counter()
            for graph in result["kernels"]:
                signature.update(graph["opcodes"])
            if batch in signatures:
                assert signature == signatures[batch], (
                    case,
                    "changed logical/native work",
                )
            else:
                signatures[batch] = signature
            t = result["timing"]
            timing[case] = t
            counters = {}
            peaks = {}
            hbf_demand = unadmitted = 0
            for kernel in t["per_kernel"]:
                hbf = kernel["memory"].get("HBF", {})
                c = hbf.get("counters", {})
                demand = hbf.get("read_bytes", 0)
                resident = c.get("resident_read_bytes", 0)
                fallback = c.get("fallback_read_bytes", 0)
                late_queue = hbf.get("unadmitted_alias_read_bytes", 0)
                gauges = hbf.get("gauges", {})
                count = gauges.get("buffer_count", 0)
                if count:
                    reserved = 0
                    for slot in range(count):
                        prefix = f"buffer_{slot}"
                        current = gauges.get(prefix + "_reserved_bytes", 0)
                        peak = gauges.get(prefix + "_peak_reserved_bytes", 0)
                        epoch = gauges.get(prefix + "_epoch")
                        assert peak <= capacity // count
                        assert epoch is None or epoch % count == slot
                        reserved += current
                        buffers.append(
                            dict(
                                case=case,
                                kernel=kernel["id"],
                                cycle=kernel["complete_cycle"],
                                buffer_count=count,
                                slot=slot,
                                epoch=epoch,
                                reserved_bytes=current,
                                peak_reserved_bytes=peak,
                            )
                        )
                    assert reserved == gauges["sram_reserved_bytes"]
                    assert gauges["sram_peak_reserved_bytes"] <= capacity
                for key, value in gauges.items():
                    if "peak_reserved_bytes" in key:
                        peaks[key] = max(peaks.get(key, 0), value)
                hbf_demand += demand
                unadmitted += late_queue
                for key, value in c.items():
                    counters[key] = counters.get(key, 0) + value
                assert resident + fallback + late_queue <= demand, (case, kernel["id"])
                kernels.append(
                    dict(
                        case=case,
                        kernel=kernel["id"],
                        start_cycle=kernel["start_cycle"],
                        complete_cycle=kernel["complete_cycle"],
                        duration_cycles=kernel["complete_cycle"]
                        - kernel["start_cycle"],
                        hbf_read_bytes=demand,
                        hbm_read_bytes=kernel["memory"]["HBM"]["read_bytes"],
                        nand_sensed_bytes=c.get("sensed_bytes", 0),
                        prefetch_submitted_bytes=c.get("prefetch_bytes", 0),
                        sram_hit_bytes=resident,
                        admitted_not_ready_bytes=fallback,
                        not_admitted_bytes=late_queue,
                        total_not_sram_bytes=demand - resident,
                        not_sram_percent=100 * (demand - resident) / demand
                        if demand
                        else "",
                        prefetch_completed=c.get("prefetch_completed", 0),
                        pending_at_launch=kernel["background_at_launch"],
                        pending_at_completion=kernel["background_at_completion"],
                    )
                )
            resident = counters.get("resident_read_bytes", 0)
            for key in (
                "persistent_gpu",
                "persistent_l2",
                "persistent_memory",
                "stable_allocations",
            ):
                assert t[key], (case, key)
            assert all(
                a["complete_cycle"] <= b["launch_cycle"]
                for a, b in zip(t["per_kernel"], t["per_kernel"][1:])
            )
            assert t["traffic_bytes"].get("HBF_write", 0) == 0
            assert t["per_kernel"][-1]["background_at_completion"] == 0
            bins = t.get("memory_bandwidth", {}).get("HBF", [])
            if mode != "full-hbm":
                assert bins, case
                totals = {
                    key: sum(b["bytes"][key] for b in bins)
                    for key in (
                        "media_read",
                        "demand_read",
                        "prefetch_fill",
                        "resident_read",
                    )
                }
                assert totals["media_read"] == counters["sensed_bytes"], (case, totals)
                assert (
                    totals["demand_read"]
                    == hbf_demand
                    == t["traffic_bytes"]["HBF_read"]
                )
                assert totals["resident_read"] == resident
                assert totals["prefetch_fill"] == counters.get("prefetch_bytes", 0)
                for b in bins:
                    delta = b["end_ps"] - b["start_ps"]
                    series.append(
                        dict(
                            case=case,
                            start_us=b["start_ps"] / 1e6,
                            end_us=b["end_ps"] / 1e6,
                            **{
                                key + "_bytes": value
                                for key, value in b["bytes"].items()
                            },
                            **{
                                key + "_GBps": value * 1000 / delta
                                for key, value in b["bytes"].items()
                            },
                            sense_commands=b["counts"]["sense_commands"],
                        )
                    )
            count = ceil((t["total_cycles"] / 1980) / 10)
            grouped = [
                dict(media_read=0, demand_read=0, prefetch_fill=0, resident_read=0)
                for _ in range(count)
            ]
            for bin in bins:
                index = int(bin["start_ps"] // 10000000)
                for key, value in bin["bytes"].items():
                    grouped[index][key] += value
            for index, values in enumerate(grouped):
                windows.append(
                    dict(
                        case=case,
                        start_us=index * 10,
                        end_us=(index + 1) * 10,
                        **{
                            key + "_GBps": value / 10000
                            for key, value in values.items()
                        },
                    )
                )
            summary.append(
                dict(
                    case=case,
                    batch=batch,
                    mode=mode,
                    cycles=t["total_cycles"],
                    sram_peak_reserved_bytes=peaks.get("sram_peak_reserved_bytes", 0),
                    buffer_peaks=json.dumps(
                        {k: v for k, v in peaks.items() if k.startswith("buffer_")}
                    ),
                    duration_us=t["total_cycles"] / 1980,
                    kernel_launches=len(t["per_kernel"]),
                    ctas=sum(k["ctas"] for k in result["kernels"]),
                    peak_10us_nand_GBps=max(row["media_read"] for row in grouped)
                    / 10000,
                    peak_10us_demand_GBps=max(row["demand_read"] for row in grouped)
                    / 10000,
                    active_experts=trace["active_experts"],
                    assignments=trace["assignments"],
                    hbf_read_bytes=hbf_demand,
                    hbm_read_bytes=t["traffic_bytes"].get("HBM_read", 0),
                    hbm_write_bytes=t["traffic_bytes"].get("HBM_write", 0),
                    nand_sensed_bytes=counters.get("sensed_bytes", 0),
                    prefetch_submitted_bytes=counters.get("prefetch_bytes", 0),
                    sram_hit_bytes=resident,
                    admitted_not_ready_bytes=counters.get("fallback_read_bytes", 0),
                    not_admitted_bytes=unadmitted,
                    uncovered_bytes=hbf_demand
                    - resident
                    - counters.get("fallback_read_bytes", 0)
                    - unadmitted,
                    total_not_sram_bytes=hbf_demand - resident,
                    not_sram_percent=100 * (hbf_demand - resident) / hbf_demand
                    if hbf_demand
                    else "",
                    mean_hbf_demand_GBps=hbf_demand * 1980 / t["total_cycles"] / 1000,
                    mean_nand_completion_GBps=counters.get("sensed_bytes", 0)
                    * 1980
                    / t["total_cycles"]
                    / 1000,
                    prefetch_capacity_stalls=t["prefetch_capacity_stalls"],
                    prefetch_canceled=t["prefetch_canceled"],
                )
            )
    write_csv(root / "summary.csv", summary)
    write_csv(root / "kernels.csv", kernels)
    write_csv(root / "bandwidth-1us.csv", series)
    write_csv(root / "bandwidth-10us.csv", windows)
    write_csv(root / "sram-buffers.csv", buffers)
    return summary, timing


def plot(root, timing):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), layout="constrained")
    names = [
        ("media_read", "NAND Sense completion"),
        ("demand_read", "HBF demand delivered"),
        ("prefetch_fill", "Prefetch SRAM fill"),
    ]
    modes = [
        m
        for m in experiment_modes(
            json.loads((root / "inputs/experiment.json").read_text()).get(
                "buffer_counts", [1]
            )
        )
        if m != "full-hbm"
    ]
    colors = {
        "no-prefetch": "#d55e00",
        "prefetch": "#777777",
        "prefetch-2buffer": "#0072b2",
        "prefetch-4buffer": "#009e73",
    }
    for column, batch in enumerate((4, 8)):
        end_us = max(
            timing[f"b{batch}-{mode}"]["total_cycles"] / 1980 for mode in modes
        )
        for mode in modes:
            color = colors[mode]
            t = timing[f"b{batch}-{mode}"]
            bins = t["memory_bandwidth"]["HBF"]
            for row, (field, label) in enumerate(names):
                if field == "prefetch_fill" and mode == "no-prefetch":
                    continue
                windows = [0] * ceil((t["total_cycles"] / 1980) / 10)
                for b in bins:
                    index = int(b["start_ps"] // 10000000)
                    while len(windows) <= index:
                        windows.append(0)
                    windows[index] += b["bytes"][field]
                x = [(i + 0.5) * 10 for i in range(len(windows))]
                axes[row, column].plot(
                    x,
                    [value / 10000 for value in windows],
                    label=mode,
                    color=color,
                    linewidth=1,
                )
                axes[row, column].set(
                    ylabel=label + " (GB/s)",
                    xlabel="Global stream time (us)",
                    xlim=(0, end_us),
                )
                axes[row, column].grid(alpha=0.2)
                axes[row, column].legend(loc="upper right")
        axes[0, column].set_title(f"Qwen3-30B-A3B B={batch}, Row-major")
    fig.suptitle(
        "HBF bandwidth over time — 10 us windows, from 1 us completion-event samples"
    )
    fig.savefig(root / "flash-bandwidth.png", dpi=170)
    fig.savefig(root / "flash-bandwidth.svg")
    plt.close(fig)


def plot_buffers(root):
    import matplotlib.pyplot as plt

    path = root / "sram-buffers.csv"
    if not path.exists():
        return
    with path.open() as source:
        rows = list(csv.DictReader(source))
    capacity_mib = sram_capacity(root) / 2**20
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout="constrained")
    for row, count in enumerate((2, 4)):
        for column, batch in enumerate((4, 8)):
            ax = axes[row, column]
            selected = [
                r for r in rows if r["case"] == f"b{batch}-prefetch-{count}buffer"
            ]
            for slot in range(count):
                values = [r for r in selected if int(r["slot"]) == slot]
                ax.step(
                    [int(r["cycle"]) / 1980 for r in values],
                    [int(r["reserved_bytes"]) / 2**20 for r in values],
                    where="post",
                    marker=".",
                    label=f"buffer {slot}",
                    linewidth=1,
                )
            ax.set(
                title=f"B={batch}, {count} buffers",
                xlabel="Global stream time (us)",
                ylabel="Reserved SRAM (MiB)",
                ylim=(0, capacity_mib / count * 1.05),
            )
            ax.grid(alpha=0.2)
            ax.legend(loc="upper right")
    fig.suptitle("Rotating SRAM buffers — kernel-completion snapshots")
    fig.savefig(root / "sram-buffers.png", dpi=170)
    fig.savefig(root / "sram-buffers.svg")
    plt.close(fig)


def report(root, summary):
    lines = [
        "# Seeded Qwen B=4/8 FlashAccel comparison",
        "",
        "| B | Mode | Cycles | HBM read MiB | HBF demand MiB | NAND MiB | Prefetch MiB | Not SRAM MiB (%) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        miss = (
            f"{r['total_not_sram_bytes'] / 2**20:.2f} ({r['not_sram_percent']:.2f}%)"
            if r["hbf_read_bytes"]
            else "—"
        )
        lines.append(
            f"| {r['batch']} | {r['mode']} | {r['cycles']:,} | {r['hbm_read_bytes'] / 2**20:.2f} | {r['hbf_read_bytes'] / 2**20:.2f} | {r['nand_sensed_bytes'] / 2**20:.2f} | {r['prefetch_submitted_bytes'] / 2**20:.2f} | {miss} |"
        )
    lines += [
        "",
        "Not-SRAM includes admitted-but-not-ready, not-yet-admitted and uncovered HBF demand reads.",
        "The denominator is actual HBF demand bytes after L2; it excludes L2 hits.",
        "The complete-region readiness rule is used by the existing baseline.",
        "",
        "Bandwidth samples record event completions in 1 us buckets. The plot aggregates them into 10 us windows.",
        "NAND rates count complete Sense output at Sense completion, including grouped 128 KiB outputs;",
        "they are not a continuous wire-rate measurement. Host delivery and prefetch SRAM fills are separate series.",
        "FullHBM has zero Flash traffic. See summary.csv, kernels.csv and bandwidth-1us.csv.",
        "",
    ]
    (root / "results.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Frozen original-pool results to compare cycles and per-kernel DAG hashes",
    )
    args = parser.parse_args()
    summary, timing = collect(args.out)
    validation = dict(
        completed_cases=len(summary),
        buffer_quota_and_rotation=True,
        native_operation_counts_equal=True,
        bandwidth_bytes_conserved=True,
    )
    if args.baseline:
        baseline = []
        for batch in (4, 8):
            for mode in ("no-prefetch", "prefetch", "full-hbm"):
                case = f"b{batch}-{mode}"
                old = json.loads((args.baseline / case / "result.json").read_text())
                new = json.loads((args.out / case / "result.json").read_text())
                cycles = old["timing"]["total_cycles"] == new["timing"]["total_cycles"]
                graphs = [k["dag_sha256"] for k in old["kernels"]] == [
                    k["dag_sha256"] for k in new["kernels"]
                ]
                assert cycles and graphs, case
                baseline.append(
                    dict(case=case, cycles_equal=cycles, dag_hashes_equal=graphs)
                )
        validation["frozen_baselines"] = baseline
    (args.out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    plot(args.out, timing)
    plot_buffers(args.out)
    report(args.out, summary)
    print(f"Validated {len(summary)} cases; wrote tables and bandwidth plots")


if __name__ == "__main__":
    main()
