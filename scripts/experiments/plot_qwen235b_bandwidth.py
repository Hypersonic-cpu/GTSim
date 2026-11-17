"""Plot the existing best-case bandwidth replay CSVs without rerunning cases."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONTEXTS = (10000, 100000, 200000)


def plot(folder, name, series, references, title):
    with (folder / f"{name}.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for axis, context in zip(axes, CONTEXTS):
        selected = [r for r in rows if int(r["context_length"]) == context]
        stop = max(int(r["end_ps"]) for r in selected)
        # Explicit zero bins avoid connecting across periods with no service.
        times = range(0, stop, 1000000)
        x = [(start + 500000) / 1000000 for start in times]
        for tier, field, label, color in series:
            bins = {
                int(r["start_ps"]): float(r[field])
                for r in selected if r["tier"] == tier
            }
            axis.plot(x, [bins.get(start, 0) for start in times],
                      label=label, color=color, linewidth=1)
        for value, label, color in references:
            axis.axhline(value, linestyle="--", linewidth=0.8,
                         color=color, label=label)
        axis.set_title(f"Context {context:,}")
        axis.set_xlabel("Simulated elapsed time (µs)")
        axis.set_xlim(0, stop / 1000000)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Bandwidth (GB/s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(title + " — best BM/BN/BK = 16/64/256; 1 µs bins")
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path(
        "validation/results/qwen235b-flashaccel-sweep/figures"))
    folder = parser.parse_args().figures
    plot(folder, "bw_fullhbm", (
        ("HBM", "hbm_read_gbps", "HBM read", "#0072B2"),
        ("HBM", "hbm_write_gbps", "HBM write", "#D55E00"),
        ("HBM", "hbm_total_gbps", "HBM total", "#009E73"),
    ), ((3100, "HBM reference: 3100 GB/s", "black"),), "Full-HBM")
    plot(folder, "bw_4f1m_pool", (
        ("HBM", "hbm_read_gbps", "HBM read", "#0072B2"),
        ("HBF", "hbf_demand_read_gbps", "HBF demand read", "#D55E00"),
        ("HBF", "hbf_prefetch_fill_gbps", "HBF prefetch fill", "#E69F00"),
        ("HBF", "hbf_media_read_gbps", "HBF NAND sense (media read)", "#CC79A7"),
        ("HBF", "hbf_resident_read_gbps", "SRAM resident read", "#009E73"),
    ), ((620, "HBM reference: 620 GB/s", "#0072B2"),
        (3072, "HBF reference: 3072 GB/s", "#555555")),
        "4F1M Pool, prefetch ON (traffic domains shown separately)")


if __name__ == "__main__":
    main()
