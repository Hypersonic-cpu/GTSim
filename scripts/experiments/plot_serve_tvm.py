#!/usr/bin/env python3
"""Twelve bars with one consistent amortized serving-time definition."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-only', action='store_true', help='Publish the nine-bar comparison before TVM optimization')
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ['#8e8e8e', '#4986b8', '#df9252', '#438d77']
    order = ['hostram', 'normal', 'hbf', 'optimized']
    if args.baseline_only:
        order, colors = order[:3], colors[:3]
    for ax, key, title in zip(axes, ['GQA_ms', 'GDN_ms', 'TPOT_ms'],
                             ['Amortized GQA-layer contribution', 'Amortized GDN-layer contribution', 'Decode step: end-to-end TPOT']):
        values = [data['calibrated_ms'][arch][key] for arch in order]
        bars = ax.bar(range(len(order)), values, color=colors, width=.65)
        ax.set_xticks(range(len(order)), ['HostRAM', '5M', '3M2F', '3M2F\n+ TVM'][:len(order)])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('ms')
        ax.set_ylim(0, max(values) * 1.18)
        ax.spines[['top', 'right']].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    fig.suptitle('Qwen3.5-9B · A100 PCIe 40GB · B=1 · 32K · Simple memory model', fontsize=13)
    fig.legend([Patch(color=color) for color in colors],
               ['HostRAM: measured TPOT', '5M: measured TPOT anchor', '3M2F: ratio estimate', '3M2F + TVM: ratio estimate'][:len(order)],
               loc='upper center', bbox_to_anchor=(.5, .92), ncol=len(order), frameon=False, fontsize=10)
    fig.text(.5, .058, 'Layer bars allocate the full serving step across logical layers: 8 × GQA + 24 × GDN = TPOT in every scenario.',
             ha='center', fontsize=10)
    fig.text(.5, .02, 'Layer contributions are amortized estimates, not kernel latency. Hybrid TPOT assumes the matched simulation ratio scales the whole serving path.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .11, 1, .85])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    svg = args.output.with_suffix('.svg')
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')


if __name__ == '__main__':
    main()
