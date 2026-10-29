#!/usr/bin/env python3
"""Plot complete layer wall windows and client TPOT; missing values stay missing."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    colors = ["#8e8e8e", "#4986b8", "#df9252"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    for ax, key, title in zip(axes, ["GQA_layer_ms", "GDN_layer_ms", "decode_step_tpot_ms"],
                             ["GQA layer + entry wait", "GDN layer + entry wait", "Decode step: end-to-end TPOT"]):
        values = list(data["comparison"][key].values())
        ymax = max(v for v in values if v is not None)
        for i, value in enumerate(values):
            if value is None:
                ax.text(i, .04 * ymax, "N/A", ha="center", va="bottom", color="#777777")
            else:
                ax.bar(i, value, color=colors[i], width=.62)
                ax.text(i, value + .015 * ymax, f"{value:.2f}", ha="center", va="bottom")
        ax.set_xticks(range(3), ["HostRAM", "5M / NoOff", "3M2F"])
        ax.set_xlim(-.6, 2.6)
        ax.set_ylim(0, ymax * 1.24)
        ax.set_ylabel("ms")
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    ratio = data["hostram_over_normal_tpot_ratio"]
    axes[2].text(.97, .93, f"HostRAM / NoOff = {ratio:.2f}×", transform=axes[2].transAxes,
                 ha="right", va="top", fontsize=10)
    fig.suptitle("Qwen3.5-9B · A100 PCIe 40GB · B=1 · 32K input · 256 output", fontsize=12)
    fig.legend([Patch(color=c) for c in colors],
               ["HostRAM: measured wall time", "5M / NoOff: measured client TPOT", "3M2F: ratio-estimated TPOT"],
               loc="upper center", bbox_to_anchor=(.5, .92), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .052, "Layer windows retain entry + internal waits. TPOT includes the whole serving path, beyond 32 layers.",
             ha="center", fontsize=9)
    fig.text(.5, .018, "3M2F TPOT = NoOff TPOT × matched simulation ratio. N/A = no layer wall result. Shared GPU, single runs.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, .85])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    svg = args.output.with_suffix(".svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
