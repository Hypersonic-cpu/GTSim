"""Run the fixed TileLens Qwen sweep from the repository root."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from itertools import product
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

LIB = Path("frontend/lib")
QWEN = Path("frontend/asm/qwen3")
TILINGS = tuple(product((64, 128, 256), (64, 128, 256), (64, 128)))
MODES = ("row-demand", "row-prefetch", "tile-demand", "tile-prefetch")
GROUPS = (
    ("qkv_projection", "expert_gemm1_direct", "expert_gemm1_grouped"),
    ("attention_output_projection", "expert_gemm2_direct", "expert_gemm2_grouped"),
    ("router_projection",),
)


def schedule_variants(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for source in LIB.glob("*ir"):
        link = directory / source.name
        if not link.exists():
            link.symlink_to(source.resolve())
    for family in ("small_m_gemm", "grouped_gemm"):
        template = (LIB / f"{family}.flowir").read_text()
        for bn in (128, 256):
            scratch = 2 * (16 * 16 + 16 * bn) * 2 + 16 * bn * 2
            text = template.replace(f"@{family}_flow", f"@{family}_flow_bn{bn}")
            text = re.sub(r"\b64\b|(?<=x)64(?=x|>)", str(bn), text)
            text = text.replace("bytes_per_slot<7168>", f"bytes_per_slot<{scratch}>")
            (directory / f"{family}_bn{bn}.flowir").write_text(text)


def program(source, tiling):
    for bn, group in zip(tiling, GROUPS):
        if bn == 64:
            continue
        for launch in group:
            source, count = re.subn(
                rf"(prog\.launch @{launch}\s+callee<@\w+> implementation<@\w+_flow)>",
                rf"\g<1>_bn{bn}>",
                source,
            )
            if count != 1:
                raise ValueError(f"expected one launch for {launch}")
    return source


def write_csv(path, records):
    if records:
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=tuple(records[0]))
            writer.writeheader()
            writer.writerows(records)


def geometry(out):
    config = json.loads((out / "inputs/prefetch.json").read_text())
    width = config["planes_per_group"]
    return (
        width * config["page_bytes"],
        config["stacks"] * config["dies_per_stack"] * config["planes_per_die"] // width,
    )


def summarize(out):
    sense_bytes, units = geometry(out)
    records, kernels = [], []
    paths = sorted((*out.glob("*/result.json"), *out.glob("focused/*/*/result.json")))
    for path in paths:
        result = json.loads(path.read_text())
        if not (timing := result.get("timing")):
            continue
        case = str(path.parent.relative_to(out))
        banks, counters, demand = {}, {}, 0
        for kernel, graph in zip(timing["per_kernel"], result["kernels"], strict=True):
            hbf = kernel["memory"].get("HBF", {})
            read = hbf.get("read_bytes", 0)
            counts = hbf.get("counters", {})
            demand += read
            for key, value in counts.items():
                counters[key] = counters.get(key, 0) + value
            for unit, ticks in hbf.get("unit_busy_ticks", {}).items():
                banks[unit] = banks.get(unit, 0) + ticks
            gauges = hbf.get("gauges", {})
            kernels.append(
                dict(
                    case=case,
                    launch=kernel["id"],
                    start_cycle=kernel["start_cycle"],
                    complete_cycle=kernel["complete_cycle"],
                    device_cycles=kernel["device_cycles"],
                    ctas=graph["ctas"],
                    sms=graph.get("sms", ""),
                    shared_bytes_per_cta=graph.get("shared_bytes_per_cta", ""),
                    register_bytes_per_cta=graph.get("register_bytes_per_cta", ""),
                    hbf_demand_bytes=read,
                    hbm_read_bytes=kernel["memory"]["HBM"]["read_bytes"],
                    nand_bytes=counts.get("sensed_bytes", 0),
                    amplification=counts.get("sensed_bytes", 0) / read if read else "",
                    sense_commands=counts.get("sense_commands", 0),
                    first_senses=counts.get("unique_sense_addresses", 0),
                    repeated_senses=counts.get("repeated_sense_commands", 0),
                    resident_bytes=counts.get("resident_read_bytes", 0),
                    fallback_bytes=counts.get("fallback_read_bytes", 0),
                    prefetch_completed=counts.get("prefetch_completed", 0),
                    pending_at_launch=kernel["background_at_launch"],
                    pending_at_completion=kernel["background_at_completion"],
                    sram_reserved_bytes=gauges.get("sram_reserved_bytes", 0),
                    sram_peak_reserved_bytes=gauges.get("sram_peak_reserved_bytes", 0),
                    prefetch_peak_outstanding=gauges.get(
                        "prefetch_peak_outstanding", 0
                    ),
                    active_scheduling_units=len(hbf.get("active_units", [])),
                    unit_busy_ps=json.dumps(
                        hbf.get("unit_busy_ticks", {}), separators=(",", ":")
                    ),
                )
            )
        records.append(
            dict(
                case=case,
                cycles=timing["total_cycles"],
                sense_bytes=sense_bytes,
                scheduling_units=units,
                demand_bytes=demand,
                nand_bytes=counters.get("sensed_bytes", 0),
                sense_commands=counters.get("sense_commands", 0),
                unique_sense_regions=counters.get("unique_sense_addresses", 0),
                repeated_senses=counters.get("repeated_sense_commands", 0),
                resident_bytes=counters.get("resident_read_bytes", 0),
                fallback_bytes=counters.get("fallback_read_bytes", 0),
                active_scheduling_units=len(banks),
                max_to_mean_unit_busy=(
                    max(banks.values()) * units / sum(banks.values()) if banks else ""
                ),
                trailing_stream_drain_cycles=timing["total_cycles"]
                - timing["per_kernel"][-1]["complete_cycle"],
            )
        )
    write_csv(out / "summary.csv", records)
    write_csv(out / "kernels.csv", kernels)
    return records


def report(out, records):
    sense_bytes, units = geometry(out)
    by_case = {row["case"]: row for row in records}
    labels = ["-".join(map(str, tiling)) for tiling in TILINGS]
    cells = {}
    for label, mode in product(labels, MODES):
        if row := by_case.get(f"{mode}-{label}"):
            cells[mode, label] = (
                f"{row['cycles']:,} / {row['nand_bytes'] / 2**20:,.2f}"
                f" / {row['repeated_senses']:,}"
            )
    if len(cells) != len(labels) * len(MODES) or "full-hbm-5-original" not in by_case:
        return
    for mode, label in cells:
        row = by_case[f"{mode}-{label}"]
        if (
            row["demand_bytes"] != by_case["row-demand-64-64-64"]["demand_bytes"]
            or row["nand_bytes"] != row["sense_commands"] * sense_bytes
            or row["sense_commands"]
            != row["unique_sense_regions"] + row["repeated_senses"]
            or row["trailing_stream_drain_cycles"]
        ):
            raise ValueError(f"inconsistent hybrid traffic accounting: {row['case']}")
    lines = [
        f"Sense granularity {sense_bytes:,} bytes; {units:,} independently scheduled units.",
        "",
        "Each cell reports cycles / NAND MiB / repeated Sense commands.",
        "",
        "| BN A/B/C | Row demand | Row prefetch | Tile demand | Tile prefetch |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in labels:
        lines.append(
            "| " + " | ".join([label, *(cells[mode, label] for mode in MODES)]) + " |"
        )
    lines += [
        "",
        "| Condition | Best A/B/C | Cycles | Original BN64 / best |",
        "|---|---|---:|---:|",
    ]
    best = {}
    for mode in MODES:
        best[mode] = min(
            labels, key=lambda label: (by_case[f"{mode}-{label}"]["cycles"], label)
        )
        a = by_case[f"{mode}-64-64-64"]["cycles"]
        b = by_case[f"{mode}-{best[mode]}"]["cycles"]
        lines.append(f"| {mode} | {best[mode]} | {b:,} | {a / b:.3f}× |")

    winner = min(
        (by_case[f"{mode}-{best[mode]}"] for mode in MODES),
        key=lambda row: row["cycles"],
    )
    lines += [
        "",
        "| Matched comparison | Numerator cycles | Denominator cycles | Ratio |",
        "|---|---:|---:|---:|",
    ]

    def compare(label, a, b):
        numerator, denominator = by_case[a]["cycles"], by_case[b]["cycles"]
        lines.append(
            f"| {label} | {numerator:,} | {denominator:,} | {numerator / denominator:.3f}× |"
        )

    for label in sorted({"64-64-64", *best.values()}):
        for mode in ("demand", "prefetch"):
            compare(
                f"Layout ({label}, {mode})",
                f"row-{mode}-{label}",
                f"tile-{mode}-{label}",
            )
        for layout in ("row", "tile"):
            compare(
                f"Prefetch ({label}, {layout})",
                f"{layout}-demand-{label}",
                f"{layout}-prefetch-{label}",
            )
    compare(
        "Row-major BN64 + prefetch / overall winner",
        "row-prefetch-64-64-64",
        winner["case"],
    )
    compare(
        "Overall winner / five-stack FullHBM", winner["case"], "full-hbm-5-original"
    )
    (out / "comparisons.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--runner", default="build-backends/gtsim_program")
    parser.add_argument("--jobs", default=2, type=int)
    parser.add_argument(
        "--planes-per-command",
        type=int,
        choices=(1, 32),
        default=32,
        help="1 disables multi-plane Sense; 32 retains the archived Megaplane experiment",
    )
    parser.add_argument(
        "--focused",
        action="store_true",
        help="Isolate GEMMs from the fastest prefetched case of each layout",
    )
    parser.add_argument(
        "--case", action="append", help="Run only this case (repeatable)"
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Refresh CSVs without launching simulations",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    if args.summarize:
        rows = summarize(out)
        report(out, rows)
        print(f"Summarized {len(rows)} completed cases")
        return
    inputs = out / "inputs"
    if (inputs / "prefetch.json").exists():
        saved = json.loads((inputs / "prefetch.json").read_text())
        if saved["planes_per_group"] != args.planes_per_command:
            parser.error(
                "use a separate output directory for a different Sense granularity"
            )
    lib = inputs / "lib"
    schedule_variants(lib)

    params = json.loads((QWEN / "b1.json").read_text())
    if params["B"] != 1 or params["NUM_SMS"] != 132:
        raise ValueError("expected fixed Qwen B=1, 132 SM experiment parameters")
    matrix_heights = {
        "W_qkv": params["H"],
        "W_o": params["HQ"] * params["D"],
        "W_router": params["H"],
        "W_gate_up": params["H"],
        "W_down": params["I"],
    }
    layouts = inputs / "layouts.json"
    layouts.write_text(
        json.dumps(
            {
                name: dict(tile_k=16, tile_n=128, matrix_k=k)
                for name, k in matrix_heights.items()
            },
            indent=2,
        )
        + "\n"
    )
    prefetch = inputs / "prefetch.json"
    prefetch.write_text(
        json.dumps(
            dict(
                stacks=2,
                dies_per_stack=8,
                planes_per_die=96,
                planes_per_group=args.planes_per_command,
                page_bytes=4096,
                plane_bytes=32768,
                stack_bytes=8388608,
                line_bytes=128,
                align_tensors_to_command_group=True,
            ),
            indent=2,
        )
        + "\n"
    )

    repo = Path.cwd()
    hybrid_base = Path("configs/memconf/2F3M_grouped_planes.yaml")
    hybrid_text = hybrid_base.read_text()
    hybrid_text = re.sub(
        r"(config_file:\s*)(\S+)",
        lambda m: m[1] + str((hybrid_base.parent / m[2]).resolve()),
        hybrid_text,
    )
    hybrid_text = hybrid_text.replace(
        "    planes_per_command_group: 32\n",
        f"    planes_per_command_group: {args.planes_per_command}\n",
    )
    no_prefetch = inputs / "2F3M.yaml"
    no_prefetch.write_text(hybrid_text)
    with_prefetch = inputs / "2F3M-prefetch.yaml"
    with_prefetch.write_text(
        hybrid_text.replace(
            "    allow_program: false\n",
            "    allow_program: false\n    enable_read_sram: true\n"
            "    read_sram_access_latency_ns: 20\n    read_sram_bandwidth_GBps_per_stack: 1536\n",
        )
    )
    commands = {}
    common = [
        sys.executable,
        "frontend/tools/run_qwen_simulations.py",
        "--params",
        str(QWEN / "b1.json"),
        "--lib",
        str(lib),
        "--runner",
        args.runner,
        "--architecture",
        "configs/v0.1.0/h100-sxm.yaml",
    ]

    def add(case, source, memory, extra):
        commands[case] = [
            *common,
            "--source",
            str(source),
            "--memory",
            str(memory),
            "--out",
            str(out / case),
            *extra,
        ]

    original = (QWEN / "flash-accel/one_block.progir").read_text()
    add(
        "full-hbm-5-original",
        QWEN / "one_block.progir",
        "configs/ramulator/h100-sxm-ramulator.yaml",
        [],
    )
    for tiling in TILINGS:
        label = "-".join(map(str, tiling))
        source = inputs / f"qwen-{label}.progir"
        source.write_text(program(original, tiling))
        for layout, enabled in product(("row", "tile"), (False, True)):
            add(
                f"{layout}-{'prefetch' if enabled else 'demand'}-{label}",
                source,
                with_prefetch if enabled else no_prefetch,
                (["--tensor-layouts", str(layouts)] if layout == "tile" else [])
                + (["--prefetch-config", str(prefetch)] if enabled else []),
            )
    if args.focused:
        winners = [
            min(
                (
                    r
                    for r in summarize(out)
                    if r["case"].startswith(f"{layout}-prefetch-")
                ),
                key=lambda r: (r["cycles"], r["case"]),
            )["case"]
            for layout in ("row", "tile")
        ]
        focused = {}
        for winner in winners:
            for launch in (
                "qkv_projection",
                "attention_output_projection",
                "router_projection",
                "expert_gemm1_direct",
                "expert_gemm2_direct",
            ):
                experts = (
                    sorted(params["ROUTE_IDS"])
                    if launch.startswith("expert_")
                    else (None,)
                )
                for expert in experts:
                    suffix = launch + (f"-{expert}" if expert is not None else "")
                    case = f"focused/{winner}/{suffix}"
                    command = list(commands[winner])
                    command[command.index("--out") + 1] = str(out / case)
                    command.extend(("--focus-launch", launch))
                    if expert is not None:
                        command.extend(("--focus-expert", str(expert)))
                    focused[case] = command
        commands = focused
    selected = args.case or list(commands)
    if unknown := set(selected) - commands.keys():
        parser.error(f"unknown cases: {sorted(unknown)}")
    (out / ("focused-commands.sh" if args.focused else "commands.sh")).write_text(
        "#!/bin/sh\nset -e\ncd "
        + shlex.quote(str(repo))
        + "\nexport PYTHONPATH=frontend/src\n"
        + "".join(shlex.join(cmd) + "\n" for cmd in commands.values())
    )

    def run(case):
        directory = out / case
        directory.mkdir(parents=True, exist_ok=True)
        result = directory / "result.json"
        if result.exists() and json.loads(result.read_text()).get("returncode") == 0:
            return case, 0
        with (directory / "driver.log").open("w") as log:
            finished = subprocess.run(
                commands[case], stdout=log, stderr=subprocess.STDOUT
            )
        return case, finished.returncode

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run, selected))
    report(out, summarize(out))
    failed = [case for case, code in results if code]
    print(f"Completed {len(results) - len(failed)}/{len(results)} cases")
    if failed:
        raise SystemExit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
