"""Reproducible Row-major Qwen B=4/8 FlashAccel comparison."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

QWEN = Path("frontend/asm/qwen3")
CONTEXTS = (10000, 16384, 24576, 32000, 12000, 20000, 28000, 30000)


def parameters(batch, seed):
    """Resolve a synthetic router trace before static IR expansion."""
    params = json.loads((QWEN / "b1.json").read_text())
    ids, probabilities = [], []
    for token in range(batch):
        logits = []
        for expert in range(params["E"]):
            key = f"{seed}:layer0:token{token}:expert{expert}".encode()
            bits = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 11
            logits.append(-4.0 + 8.0 * bits / 2**53)
        selected = sorted(range(params["E"]), key=lambda e: (-logits[e], e))[
            : params["TK"]
        ]
        weights = [math.exp(logits[e] - logits[selected[0]]) for e in selected]
        total = math.fsum(weights)
        ids.extend(selected)
        probabilities.extend(w / total for w in weights)
    params.update(
        B=batch,
        S=1,
        CONTEXTS=CONTEXTS[:batch],
        ROUTE_IDS=ids,
        ROUTE_PROBS=probabilities,
    )
    return params


def experiment_modes(buffer_counts):
    return [
        "no-prefetch",
        *(
            "prefetch" if count == 1 else f"prefetch-{count}buffer"
            for count in buffer_counts
        ),
        "full-hbm",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--planes-per-command", type=int, choices=(1, 32), default=32)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--buffers",
        type=int,
        nargs="+",
        choices=(1, 2, 4),
        default=[1],
        help="1 uses the original shared pool; 2/4 use rotating fixed buffers",
    )
    parser.add_argument("--runner", default="build-backends/gtsim_program")
    args = parser.parse_args()
    inputs = args.out / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    settings = dict(
        seed=args.seed,
        planes_per_command=args.planes_per_command,
        statistics_interval_ps=1000000,
        tiling=[16, 64, 16],
        layout="RowMajor",
    )
    if args.buffers != [1]:
        settings["buffer_counts"] = args.buffers
    metadata = inputs / "experiment.json"
    if metadata.exists() and json.loads(metadata.read_text()) != settings:
        parser.error("use a separate output directory for changed experiment settings")
    metadata.write_text(json.dumps(settings, indent=2) + "\n")
    source = Path("configs/memconf/2F3M_grouped_planes.yaml")
    memory = source.read_text()
    profile = "../ramulator/h100_hbm3_3stack.yaml"
    memory = memory.replace(profile, str((source.parent / profile).resolve()))
    memory = memory.replace(
        "    planes_per_command_group: 32\n",
        f"    planes_per_command_group: {args.planes_per_command}\n",
    )
    memory = memory.replace(
        "    allow_program: false\n",
        "    allow_program: false\n    statistics_interval_ps: 1000000\n",
    )
    (inputs / "no-prefetch.yaml").write_text(memory)
    (inputs / "prefetch.yaml").write_text(
        memory.replace(
            "    allow_program: false\n",
            "    allow_program: false\n    enable_read_sram: true\n"
            "    read_sram_access_latency_ns: 20\n    read_sram_bandwidth_GBps_per_stack: 1536\n",
        )
    )
    residency = dict(
        stacks=2,
        dies_per_stack=8,
        planes_per_die=96,
        planes_per_group=args.planes_per_command,
        page_bytes=4096,
        plane_bytes=32768,
        stack_bytes=8388608,
        line_bytes=128,
        align_tensors_to_command_group=True,
    )
    (inputs / "prefetch.json").write_text(json.dumps(residency, indent=2) + "\n")
    for count in args.buffers:
        if count > 1:
            (inputs / f"prefetch-{count}buffer.json").write_text(
                json.dumps({**residency, "buffer_count": count}, indent=2) + "\n"
            )
    commands = {}
    for batch in (4, 8):
        params = parameters(batch, args.seed)
        file = inputs / f"b{batch}.json"
        file.write_text(json.dumps(params, indent=2) + "\n")
        counts = Counter(params["ROUTE_IDS"])
        trace = dict(
            seed=args.seed,
            generator="SHA-256 counter logits in [-4,4), top-8, selected-softmax normalization",
            layer=0,
            batch=batch,
            active_experts=len(counts),
            expert_counts={str(e): counts[e] for e in sorted(counts)},
            assignments=batch * params["TK"],
            ids=params["ROUTE_IDS"],
            probabilities=params["ROUTE_PROBS"],
            context_lengths=params["CONTEXTS"],
        )
        (inputs / f"b{batch}-routing.json").write_text(
            json.dumps(trace, indent=2) + "\n"
        )
        for mode in experiment_modes(args.buffers):
            name = f"b{batch}-{mode}"
            full = mode == "full-hbm"
            command = [
                sys.executable,
                "frontend/tools/run_qwen_simulations.py",
                "--params",
                str(file),
                "--runner",
                args.runner,
                "--architecture",
                "configs/v0.1.0/h100-sxm.yaml",
                "--source",
                str(
                    QWEN
                    / ("one_block.progir" if full else "flash-accel/one_block.progir")
                ),
                "--memory",
                "configs/ramulator/h100-sxm-ramulator.yaml"
                if full
                else str(
                    inputs
                    / (
                        "prefetch.yaml"
                        if mode.startswith("prefetch")
                        else "no-prefetch.yaml"
                    )
                ),
                "--out",
                str(args.out / name),
            ]
            if mode.startswith("prefetch"):
                command.extend(("--prefetch-config", str(inputs / f"{mode}.json")))
            commands[name] = command
    (args.out / "commands.sh").write_text(
        "#!/bin/sh\nset -e\nexport PYTHONPATH=frontend/src\n"
        + "\n".join(shlex.join(cmd) for cmd in commands.values())
        + "\n"
    )

    def run(item):
        name, command = item
        directory = args.out / name
        directory.mkdir(exist_ok=True)
        result = directory / "result.json"
        if result.exists() and json.loads(result.read_text()).get("returncode") == 0:
            return name, 0
        with (directory / "driver.log").open("w") as log:
            done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        return name, done.returncode

    with ThreadPoolExecutor(max_workers=args.jobs) as workers:
        completed = list(workers.map(run, commands.items()))
    failures = [name for name, code in completed if code]
    print(json.dumps(dict(completed=len(completed) - len(failures), failed=failures)))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
