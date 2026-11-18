"""Sweep standalone packed and legacy FA3 decode using the shared launcher."""

import argparse
from ast import literal_eval
from collections import Counter
import csv
from itertools import product
import json
from pathlib import Path
import sys

from sweep_qwen235b_flashaccel import main


CONTEXTS = (10_000, 100_000, 200_000)


def cases():
    model = dict(B=1, HQ=64, HKV=4, D=128, NUM_SMS=132, STAGES=2)
    for context in CONTEXTS:
        yield dict(model, CONTEXTS=[context], PACK_GQA=0, R=1, S=4, TILE=256)
        for r, splits, tile in product((4, 8, 16), (4, 8, 16, 32, 64, 128), (64, 128)):
            yield dict(model, CONTEXTS=[context], PACK_GQA=1, R=r, S=splits, TILE=tile)


def write_report(output, core_mhz=1980):
    with (output / "manifest.csv").open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    statuses = Counter(row["status"] for row in manifest)
    completed = {}
    topologies = set()
    lines = [
        "# Standalone GQA decode simulation",
        "",
        "Qwen3-235B-A22B dimensions, B=1, SM90. Context length excludes",
        "the current token. Cycles are the simulated split and combine device cycles.",
        "Cases: " + ", ".join(f"{count} {status.lower()}" for status, count in sorted(statuses.items())) + ".",
        "",
        "| Context | Path | R/S/BN/stages | Completed | Split cycles | Combine cycles | Total cycles | HBM read B | L2 hit / merged / miss B (split) | Split CTAs / SMs |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for context, packed in product(CONTEXTS, (0, 1)):
        candidates = []
        case_count = 0
        for row in manifest:
            params = literal_eval(row["parameters"])
            if params["CONTEXTS"] != [context] or params["PACK_GQA"] != packed:
                continue
            case_count += 1
            if row["status"] == "SUCCESS" and row["cycles"]:
                result = json.loads((output / "cases" / row["case"] / "result.json").read_text())
                topologies.add(result["memory_topology"])
                split, combine = result["timing"]["per_kernel"]
                total = split["device_cycles"] + combine["device_cycles"]
                candidates.append((total, row["case"], params, split, combine))
        path = "Packed" if packed else "Legacy"
        if not candidates:
            lines.append(f"| {context:,} | {path} | — | 0/{case_count} | — | — | — | — | — | — |")
            continue
        total, _, params, split, combine = min(candidates)
        hbm = sum(kernel["memory"]["HBM"]["read_bytes"] for kernel in (split, combine))
        completed[context, packed] = total, hbm, split
        l2 = " / ".join(f"{split[f'l2_read_{kind}_bytes']:,}" for kind in ("hit", "merged", "miss"))
        tiling = "/".join(str(params[k]) for k in ("R", "S", "TILE", "STAGES"))
        if not packed:
            tiling = f"1/{params['S']}/{params['TILE']}/1"
        lines.append(
            f"| {context:,} | {path} | {tiling} | {len(candidates)}/{case_count} | {split['device_cycles']:,} | "
            f"{combine['device_cycles']:,} | {total:,} | {hbm:,} | {l2} | "
            f"{split['cta_count']} / {split['active_sms']} |"
        )
    lines.extend((
        "",
        "Memory topology: " + (", ".join(sorted(topologies)) or "no completed runs") + ".",
        "",
        "| Context | Packed speedup | HBM read reduction | TMA load reduction (split) | Split HBM GB/s, legacy → packed |",
        "|---:|---:|---:|---:|---:|",
    ))
    for context in CONTEXTS:
        legacy, packed = (completed.get((context, flag)) for flag in (0, 1))
        if legacy is None or packed is None:
            continue
        def read_rate(split):
            return split["memory"]["HBM"]["read_bytes"] * core_mhz / (1000 * split["device_cycles"])

        lines.append(
            f"| {context:,} | {legacy[0] / packed[0]:.3f}× | "
            f"{1 - packed[1] / legacy[1]:.1%} | "
            f"{1 - packed[2]['tma_load_bytes'] / legacy[2]['tma_load_bytes']:.1%} | "
            f"{read_rate(legacy[2]):,.0f} → {read_rate(packed[2]):,.0f} |"
        )
    lines.extend((
        "",
        "Each result.json also records TMA bytes, per-pipeline in-flight fractions,",
        "active SM cycles, and a timestamp trace of the latest explicit dependencies.",
        "The fractions include completion waits and do not represent physical hardware busy counters.",
        "",
        "The repository H100 golden points are multi-query MHA. This experiment has no",
        "matching physical H100 decode measurements.",
    ))
    (output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, default=Path("validation/results/attn3fix-decode"))
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--report-core-mhz", type=float, default=1980)
    args, other = parser.parse_known_args(sys.argv[1:])
    if args.report_only:
        if other:
            parser.error("--report-only accepts --output and --report-core-mhz")
    else:
        main(list(cases()), [
            "--program", "frontend/asm/flash-accel/fa3_decode.progir",
            "--output", str(args.output),
            *other,
        ])
    write_report(args.output, args.report_core_mhz)
