"""Validate and report the reduced-SRAM double-buffer sweep."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from report_gemm_tiling_search import write_csv


def analyze(folder, size, bn, bk):
    result = json.loads((folder / "result.json").read_text())
    assert result["returncode"] == 0, folder
    timing = result["timing"]
    tag = dict(sram_mib_per_stack=size, BM=16, BN=bn, BK=bk)
    kernels = timing["per_kernel"]
    assert [k["id"] for k in kernels] == [g["id"] for g in result["kernels"]]
    assert all(
        timing[key]
        for key in (
            "persistent_gpu",
            "persistent_memory",
            "persistent_l2",
            "stable_allocations",
        )
    )
    assert all(k["state"] == "COMPLETED" for k in kernels)
    assert all(
        a["complete_cycle"] <= b["launch_cycle"] for a, b in zip(kernels, kernels[1:])
    )
    topk = next(
        k["complete_cycle"] for k in kernels if k["id"] == "router_softmax_topk"
    )
    counters = Counter()
    windows, kernel_rows = [], []
    unadmitted = 0
    peak = 0
    for k in kernels:
        hbf = k["memory"].get("HBF", {})
        c = hbf.get("counters", {})
        gauges = hbf.get("gauges", {})
        counters.update(c)
        unadmitted += hbf.get("unadmitted_alias_read_bytes", 0)
        peak = max(peak, gauges.get("sram_peak_reserved_bytes", 0))
        assert peak <= size * 2 * 2**20
        prior = {}
        for w in k.get("prefetch_windows", []):
            slot = w["slot"]
            assert size > 0 and slot == w["epoch"] % 2
            assert 0 < w["bytes"] <= size * 2**20
            assert k["start_cycle"] <= w["consume_cycle"] <= k["device_complete_cycle"]
            if w["skipped"]:
                assert w["enqueue_cycle"] == w["release_request_cycle"] == -1
            else:
                assert k["launch_cycle"] <= w["enqueue_cycle"] <= w["consume_cycle"]
                assert w["release_request_cycle"] == w["consume_cycle"]
                assert w["enqueue_cycle"] >= prior.get(slot, 0)
                if k["id"].startswith("expert_gemm"):
                    assert w["enqueue_cycle"] >= topk
                prior[slot] = w["consume_cycle"]
            windows.append(dict(**tag, kernel=k["id"], **w))
        kernel_rows.append(
            dict(
                **tag,
                kernel=k["id"],
                start_cycle=k["start_cycle"],
                device_complete_cycle=k["device_complete_cycle"],
                complete_cycle=k["complete_cycle"],
                device_cycles=k["device_cycles"],
                hbm_read_bytes=k["memory"].get("HBM", {}).get("read_bytes", 0),
                hbf_read_bytes=hbf.get("read_bytes", 0),
                nand_bytes=c.get("sensed_bytes", 0),
                sram_hit_bytes=c.get("resident_read_bytes", 0),
                prefetch_bytes=c.get("prefetch_bytes", 0),
                unadmitted_bytes=hbf.get("unadmitted_alias_read_bytes", 0),
            )
        )
    bandwidth = defaultdict(Counter)
    total = Counter()
    sense_commands = 0
    for b in timing["memory_bandwidth"].get("HBF", []):
        total.update(b["bytes"])
        sense_commands += b["counts"]["sense_commands"]
        bandwidth[b["start_ps"] // 10_000_000].update(b["bytes"])
    traffic = timing["traffic_bytes"]
    demand = traffic.get("HBF_read", 0)
    resident = total["resident_read"]
    assert total["demand_read"] == demand
    assert total["media_read"] == sense_commands * 128 * 1024
    assert traffic.get("HBF_write", 0) == 0
    assert resident == counters["resident_read_bytes"]
    miss = demand - resident
    late = counters["fallback_read_bytes"]
    assert miss >= late + unadmitted
    cycles = timing["total_cycles"]
    rows = []
    keys = ("media_read", "demand_read", "prefetch_fill", "resident_read")
    for i in range((cycles + 19799) // 19800):
        values = bandwidth[i]
        rows.append(
            dict(
                **tag,
                start_us=i * 10,
                end_us=(i + 1) * 10,
                **{key + "_GBps": values[key] / 10000 for key in keys},
            )
        )
    assert not size or windows
    signature = [
        (g["id"], g["nodes"], g["ctas"], g["opcodes"]) for g in result["kernels"]
    ]
    return (
        dict(
            **tag,
            cycles=cycles,
            duration_us=cycles / 1980,
            hbm_read_bytes=traffic.get("HBM_read", 0),
            hbm_write_bytes=traffic.get("HBM_write", 0),
            hbf_read_bytes=demand,
            nand_bytes=total["media_read"],
            sense_commands=sense_commands,
            repeated_senses_during_kernels=counters["repeated_sense_commands"],
            prefetch_bytes=total["prefetch_fill"],
            sram_hit_bytes=resident,
            not_sram_bytes=miss,
            not_sram_percent=100 * miss / demand if demand else "",
            admitted_not_ready_bytes=late,
            not_admitted_bytes=unadmitted,
            uncovered_bytes=miss - late - unadmitted,
            peak_sram_bytes=peak,
            windows=len(windows),
            skipped_windows=sum(w["skipped"] for w in windows),
            peak_nand_GBps_10us=max((r["media_read_GBps"] for r in rows), default=0),
            mean_nand_GBps=total["media_read"] * 1.98 / cycles,
            wall_seconds=result["elapsed_seconds"],
        ),
        kernel_rows,
        windows,
        rows,
        signature,
    )


def collect(root):
    config = json.loads((root / "sram-8mib/inputs/experiment.json").read_text())
    rows, kernels, windows, bandwidth = [], [], [], []
    for _, bn, bk in config["candidates"]:
        control = None
        for size in (0, 8, 4):
            folder = (
                root
                / f"sram-{size or 8}mib"
                / f"{'double-buffer' if size else 'full-hbm'}-n{bn}-k{bk}"
            )
            summary, k, w, bw, signature = analyze(folder, size, bn, bk)
            if size:
                assert signature == control, (
                    size,
                    bn,
                    bk,
                    "changed compute graph counts",
                )
            else:
                control = signature
            rows.append(summary)
            kernels.extend(k)
            windows.extend(w)
            if size:
                bandwidth.extend(bw)
    return config, rows, kernels, windows, bandwidth


def figures(root, rows, bandwidth):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np

    by_key = {(r["sram_mib_per_stack"], r["BN"], r["BK"]): r for r in rows}
    norm = LogNorm(min(r["cycles"] for r in rows), max(r["cycles"] for r in rows))
    bn_values = bk_values = (32, 64, 128, 256)
    for size in (8, 4):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        for ax, mem in zip(axes, (0, size)):
            grid = np.full((4, 4), np.nan)
            for i, bn in enumerate(bn_values):
                for j, bk in enumerate(bk_values):
                    r = by_key.get((mem, bn, bk))
                    if r:
                        grid[i, j] = r["cycles"]
            cmap = matplotlib.colormaps["RdYlGn_r"].with_extremes(bad="#dddddd")
            image = ax.imshow(grid, cmap=cmap, norm=norm)
            best = np.nanmin(grid)
            for i, bn in enumerate(bn_values):
                for j, bk in enumerate(bk_values):
                    v = grid[i, j]
                    style = dict(fontweight="bold") if v == best else {}
                    label = "SMEM limit" if np.isnan(v) else f"{v / 1e6:.3f}M"
                    color = (
                        "white"
                        if not np.isnan(v) and (norm(v) < 0.13 or norm(v) > 0.86)
                        else "black"
                    )
                    ax.text(j, i, label, ha="center", va="center", color=color, **style)
            ax.set(
                xticks=range(4),
                xticklabels=bk_values,
                yticks=range(4),
                yticklabels=bn_values,
                xlabel="BK",
                ylabel="BN",
                title=f"DoubleBuffer: {size} MiB per stack"
                if mem
                else "FullHBM: 5 stacks (reused)",
            )
        fig.suptitle("Qwen3 decode B=4: shared A/B tiling, BM=16")
        fig.colorbar(image, ax=axes, label="Stream cycles (log scale)", shrink=0.82)
        for ext in ("png", "svg"):
            fig.savefig(root / f"tiling-cycles-{size}mib.{ext}", dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), layout="constrained")
    for axis_row, size in zip(axes, (8, 4)):
        candidates = [r for r in rows if r["sram_mib_per_stack"] == size]
        for ax, label, r in zip(
            axis_row,
            ("fastest", "slowest"),
            (
                min(candidates, key=lambda r: r["cycles"]),
                max(candidates, key=lambda r: r["cycles"]),
            ),
        ):
            series = [
                b
                for b in bandwidth
                if (b["sram_mib_per_stack"], b["BN"], b["BK"])
                == (size, r["BN"], r["BK"])
            ]
            for key, name in (
                ("media_read", "Flash sense"),
                ("demand_read", "Demand"),
                ("prefetch_fill", "Prefetch fill"),
                ("resident_read", "SRAM hit"),
            ):
                ax.step(
                    [b["start_us"] for b in series],
                    [b[f"{key}_GBps"] for b in series],
                    label=name,
                    where="post",
                    linewidth=1,
                )
            ax.set(
                title=f"{size} MiB, {label}: BN{r['BN']}/BK{r['BK']}",
                xlabel="Time (µs)",
                ylabel="GB/s per package",
            )
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("HBF completion bandwidth in 10 µs windows")
    for ext in ("png", "svg"):
        fig.savefig(root / f"flash-bandwidth.{ext}", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    config, rows, kernels, windows, bandwidth = collect(args.out)
    assert len(rows) == 45
    for name, records in (
        ("summary", rows),
        ("kernels", kernels),
        ("prefetch-windows", windows),
        ("bandwidth-10us", bandwidth),
    ):
        write_csv(args.out / f"{name}.csv", records)
    figures(args.out, rows, bandwidth)
    best = {
        size: min(
            (r for r in rows if r["sram_mib_per_stack"] == size),
            key=lambda r: r["cycles"],
        )
        for size in (0, 8, 4)
    }
    worst = {
        size: max(
            (r for r in rows if r["sram_mib_per_stack"] == size),
            key=lambda r: r["cycles"],
        )
        for size in (0, 8, 4)
    }
    report = dict(
        completed_double_buffer=30,
        reused_full_hbm=15,
        config=config,
        best=best,
        worst=worst,
    )
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"best": best, "worst": worst}, indent=2))


if __name__ == "__main__":
    main()
