"""Snapshot only completed GEMM-tiling simulations and generate an interim report."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def snapshot(root):
    config = json.loads((root / "inputs/experiment.json").read_text())
    params = json.loads((root / "inputs/b4.json").read_text())
    experts = len(set(params["ROUTE_IDS"]))
    expected_mma = (
        5120 * 2048 + 2048 * 4096 + 128 * 2048 + experts * (1536 * 2048 + 2048 * 768)
    ) // 128
    rows = []
    status = {}
    failures = []
    for bm, bn, bk in config["candidates"]:
        for case in ("full-hbm", "double-buffer"):
            name = f"{case}-n{bn}-k{bk}"
            folder = root / name
            path = folder / "result.json"
            if not path.exists():
                status[name] = (
                    "running" if (folder / "driver.log").exists() else "queued"
                )
                continue
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                status[name] = "writing"
                continue
            if data["returncode"]:
                failures.append(name)
                status[name] = "failed"
                continue
            t = data["timing"]
            counts = Counter()
            resident = unadmitted = 0
            matrix = sum(
                g["opcodes"].get("mma.sync.m16n8k16", 0) for g in data["kernels"]
            )
            assert matrix == expected_mma, (name, matrix, expected_mma)
            for k in t["per_kernel"]:
                h = k["memory"].get("HBF", {})
                counts.update(h.get("counters", {}))
                unadmitted += h.get("unadmitted_alias_read_bytes", 0)
            demand = t["traffic_bytes"].get("HBF_read", 0)
            resident = counts.get("resident_read_bytes", 0)
            assert t["traffic_bytes"].get("HBF_write", 0) == 0
            if case == "double-buffer":
                p = json.loads((folder / "program.gtsir").read_text())
                names = {x["id"]: x["name"] for x in p["tensors"]}
                aliases = {x["id"]: x for x in p["prefetches"]}
                routed = False
                for c in p["commands"]:
                    if (
                        c["kind"] == "LaunchKernel"
                        and c["target"] == "router_softmax_topk"
                    ):
                        routed = True
                    elif c["kind"] in ("QueuePrefetch", "Prefetch"):
                        if names[aliases[c["target"]]["tensor"]] in (
                            "W_gate_up",
                            "W_down",
                        ):
                            assert routed, name
            status[name] = "completed"
            rows.append(
                dict(
                    case=case,
                    BM=bm,
                    BN=bn,
                    BK=bk,
                    cycles=t["total_cycles"],
                    duration_us=t["total_cycles"] / 1980,
                    shared_bytes=4 * bk * (16 + bn) + 32 * bn,
                    register_bytes=16384 + 64 * bn,
                    hbm_read_bytes=t["traffic_bytes"].get("HBM_read", 0),
                    hbf_read_bytes=demand,
                    nand_bytes=counts.get("sensed_bytes", 0),
                    prefetch_bytes=counts.get("prefetch_bytes", 0),
                    not_sram_bytes=demand - resident,
                    not_sram_percent=100 * (demand - resident) / demand
                    if demand
                    else "",
                    late_admitted_bytes=counts.get("fallback_read_bytes", 0),
                    unadmitted_bytes=unadmitted,
                    kernel_launches=len(t["per_kernel"]),
                    ctas=sum(k["ctas"] for k in data["kernels"]),
                    mma_instructions=matrix,
                    wall_seconds=data["elapsed_seconds"],
                )
            )
    return dict(
        snapshot_utc=datetime.now(timezone.utc).isoformat(),
        completed=len(rows),
        total=44,
        failures=failures,
        config=config,
        rows=rows,
        status=status,
        active_experts=experts,
    )


def render(out, data):
    rows = data["rows"]
    by_key = {(r["case"], r["BN"], r["BK"]): r for r in rows}
    lines = [
        "# B=4 shared A/B GEMM tiling — interim report",
        "",
        f"Snapshot {data['snapshot_utc']}. Completed {data['completed']}/44, failures {len(data['failures'])}.",
        "Only successful results present at snapshot time are included. Pending jobs continue independently.",
        "",
        "A: QKV + expert Gate/Up. B: attention output + expert Down. A/B share one tuple.",
        "C is fixed at (16,64,16); FA3 uses TILE=256, S=1. Seed 20261106, 29 active experts.",
        "Contexts [10000,16384,24576,32000], H100 132 SMs; FullHBM has five HBM stacks.",
        "DoubleBuffer uses the original 128 KiB FlashAccel geometry, two 32 MiB SRAM slots.",
        "BM=16 for every candidate. No hardware timing constants are tuned.",
        "",
        "| Case | Completed | Original cycles (BN64/BK16) | Best measured (BN,BK) | Best cycles | Reduction | Worst cycles | Worst/original |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    bests = []
    for case in ("full-hbm", "double-buffer"):
        current = [r for r in rows if r["case"] == case]
        base = by_key[case, 64, 16]
        best = min(current, key=lambda r: r["cycles"])
        worst = max(current, key=lambda r: r["cycles"])
        bests.append(best)
        lines.append(
            f"| {case} | {len(current)}/22 | {base['cycles']:,} | ({best['BN']},{best['BK']}) | {best['cycles']:,} | {100 * (1 - best['cycles'] / base['cycles']):.2f}% | {worst['cycles']:,} | {worst['cycles'] / base['cycles']:.3f}× |"
        )
    lines += [
        "",
        "## Traffic at the baseline and current best",
        "",
        "| Case | BN/BK | Cycles | HBM read MiB | HBF demand MiB | NAND MiB | Prefetch MiB | Not SRAM MiB (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for best in bests:
        for r in (by_key[best["case"], 64, 16], best):
            miss = (
                f"{r['not_sram_bytes'] / 2**20:.2f} ({r['not_sram_percent']:.2f}%)"
                if r["hbf_read_bytes"]
                else "—"
            )
            lines.append(
                f"| {r['case']} | {r['BN']}/{r['BK']} | {r['cycles']:,} | {r['hbm_read_bytes'] / 2**20:.2f} | {r['hbf_read_bytes'] / 2**20:.2f} | {r['nand_bytes'] / 2**20:.2f} | {r['prefetch_bytes'] / 2**20:.2f} | {miss} |"
            )
    lines += [
        "",
        "## Candidate results",
        "",
        "| BN | BK | FullHBM cycles | DoubleBuffer cycles |",
        "|---:|---:|---:|---:|",
    ]
    for _, bn, bk in data["config"]["candidates"]:
        cells = []
        for case in ("full-hbm", "double-buffer"):
            row = by_key.get((case, bn, bk))
            cells.append(
                f"{row['cycles']:,}" if row else data["status"][f"{case}-n{bn}-k{bk}"]
            )
        lines.append(f"| {bn} | {bk} | {' | '.join(cells)} |")
    lines += [
        "",
        "All completed cases retain the same independently checked MMA instruction count.",
        "All completed DoubleBuffer cases enqueue expert prefetch only after the timed Top-K launch completes.",
        "BN512 is lowered into 256-wide native TMA boxes. Its results are included only once completed.",
        "These are timing-model estimates; the report does not establish an optimum over unfinished candidates.",
        "",
        "Files: snapshot.json, summary.csv and tiling-cycles.png/.svg.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    write_csv(out / "summary.csv", rows)


def plot(out, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle
    import numpy as np

    bn_values = (32, 64, 128, 256, 512)
    bk_values = (16, 32, 64, 128, 256)
    valid = {(n, k) for _, n, k in data["config"]["candidates"]}
    rows = data["rows"]
    by_key = {(r["case"], r["BN"], r["BK"]): r for r in rows}
    ratios = [r["cycles"] / by_key[r["case"], 64, 16]["cycles"] for r in rows]
    norm = TwoSlopeNorm(
        vmin=min(0.5, min(ratios)), vcenter=1, vmax=max(1.5, max(ratios))
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    for ax, case in zip(axes, ("full-hbm", "double-buffer")):
        matrix = np.full((5, 5), np.nan)
        for i, n in enumerate(bn_values):
            for j, k in enumerate(bk_values):
                row = by_key.get((case, n, k))
                if row:
                    matrix[i, j] = row["cycles"] / by_key[case, 64, 16]["cycles"]
        image = ax.imshow(matrix, cmap="RdYlGn_r", norm=norm)
        for i, n in enumerate(bn_values):
            for j, k in enumerate(bk_values):
                row = by_key.get((case, n, k))
                if row:
                    text = f"{row['cycles'] / 1e6:.3f}M\n{matrix[i, j]:.2f}x"
                elif (n, k) not in valid:
                    ax.add_patch(
                        Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#bbbbbb")
                    )
                    text = "SMEM\nexcluded"
                else:
                    text = data["status"][f"{case}-n{n}-k{k}"]
                ax.text(j, i, text, ha="center", va="center", fontsize=9)
        ax.add_patch(
            Rectangle((-0.5, 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2)
        )
        ax.set(
            xticks=range(5),
            xticklabels=bk_values,
            yticks=range(5),
            yticklabels=bn_values,
            xlabel="BK",
            ylabel="BN",
            title=f"{case}: {sum(r['case'] == case for r in rows)}/22 completed",
        )
    fig.colorbar(
        image,
        ax=axes,
        label="Cycles / original BN64-BK16 (lower is better)",
        shrink=0.82,
    )
    fig.suptitle(
        f"Interim B=4 shared A/B tiling — {data['completed']}/44 completed, BM=16"
    )
    fig.savefig(out / "tiling-cycles.png", dpi=170)
    fig.savefig(out / "tiling-cycles.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report-name", default="interim")
    args = parser.parse_args()
    data = snapshot(args.out)
    out = args.out / args.report_name
    out.mkdir(exist_ok=True)
    (out / "snapshot.json").write_text(json.dumps(data, indent=2) + "\n")
    render(out, data)
    plot(out, data)
    print((out / "report.md").read_text().split("## Candidate results")[0])


if __name__ == "__main__":
    main()
