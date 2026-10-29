"""Diagnostic plot of simulated components only; never a layer or TPOT baseline."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.3))
    colors = ["#4986b8", "#df9252"]
    for ax, key, title in zip(axes, ["GQA", "GDN"],
                             ["GQA: selected simulated components", "GDN: selected simulated projections"]):
        values = [data[s][key] for s in ["normal", "hbf"]]
        bars = ax.bar(range(2), values, color=colors, width=.6)
        ax.set_xticks(range(2), ["5M simulation", "3M2F simulation"])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("μs")
        ax.set_ylim(0, max(values) * 1.2)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
    fig.suptitle("Simulator component diagnostics", fontsize=12)
    fig.text(.5, .02, "Incomplete components only. No measured HostRAM baseline. No 32-layer or TPOT estimate.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .07, 1, .95])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    svg = args.output.with_suffix(".svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
