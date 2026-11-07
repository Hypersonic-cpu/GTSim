"""Fixed B=4, shared A/B GEMM schedule search for FullHBM and double buffering."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from time import perf_counter

from run_flash_accel_batches import parameters
from run_tile_lens_sweep import GROUPS

LIB = Path("frontend/lib")


def candidates(hardware):
    result = []
    for bn, bk in product((32, 64, 128, 256, 512), (16, 32, 64, 128, 256)):
        shared = 4 * bk * (16 + bn) + 2 * 16 * bn
        registers = 16384 + 16 * bn * 4
        if (
            shared <= hardware["shared_memory_per_cta_bytes"]
            and registers <= hardware["register_file_per_sm_bytes"]
        ):
            result.append((16, bn, bk))
    assert len(result) == 22
    return result


def flow_variant(family, bn, bk):
    text = (LIB / f"{family}.flowir").read_text()
    text = text.replace(f"@{family}_flow", f"@{family}_flow_n{bn}_k{bk}")
    for old, new in (
        ("ceildiv(%N,64)", f"ceildiv(%N,{bn})"),
        ("%N divisible_by 64", f"%N divisible_by {bn}"),
        ("%K divisible_by 16", f"%K divisible_by {bk}"),
        ("tile(16x64x16)", f"tile(16x{bn}x{bk})"),
        ("ceildiv(%K,16)", f"ceildiv(%K,{bk})"),
        ("%k*16", f"%k*{bk}"),
        ("%COL*64", f"%COL*{bn}"),
        ("bytes_per_slot<7168>", f"bytes_per_slot<{4 * bk * (16 + bn) + 32 * bn}>"),
    ):
        text = text.replace(old, new)
    lines = []
    for line in text.splitlines():
        if "flow.load_async @load_A " in line:
            line = line.replace("VALIDx16", f"VALIDx{bk}")
        elif "flow.load_async @load_B " in line:
            line = line.replace("16x64", f"{bk}x{bn}")
        elif "flow.matmul " in line:
            line = line.replace("16x64", f"16x{bn}").replace("VALIDx64", f"VALIDx{bn}")
        elif "flow.store_async " in line:
            line = line.replace("VALIDx64", f"VALIDx{bn}")
        lines.append(line)
    return "\n".join(lines) + "\n"


def program_variant(source, bn, bk):
    for group in GROUPS[:2]:
        for launch in group:
            pattern = (
                rf"(prog\.launch @{launch}\s+callee<@(\w+)> implementation<@)\w+(>)"
            )
            source, count = re.subn(
                pattern, lambda m: m[1] + f"{m[2]}_flow_n{bn}_k{bk}" + m[3], source
            )
            assert count == 1, launch
    return source


def prepare(args):
    inputs = args.out / "inputs"
    lib = inputs / "lib"
    lib.mkdir(parents=True, exist_ok=True)
    hardware = json.loads(
        subprocess.check_output(
            [
                args.runner,
                "configs/v0.1.0/h100-sxm.yaml",
                "--memory-config",
                "configs/ramulator/h100-sxm-ramulator.yaml",
                "--describe-config",
            ],
            text=True,
        )
    )
    tiles = candidates(hardware)
    for path in LIB.glob("*ir"):
        link = lib / path.name
        if not link.exists():
            link.symlink_to(path.resolve())
    for _, bn, bk in tiles:
        for family in ("small_m_gemm", "grouped_gemm"):
            (lib / f"{family}_n{bn}_k{bk}.flowir").write_text(
                flow_variant(family, bn, bk)
            )
    params = parameters(4, args.seed)
    (inputs / "b4.json").write_text(json.dumps(params, indent=2) + "\n")
    residency = dict(
        stacks=2,
        dies_per_stack=8,
        planes_per_die=96,
        planes_per_group=32,
        page_bytes=4096,
        plane_bytes=32768,
        stack_bytes=8388608,
        line_bytes=128,
        align_tensors_to_command_group=True,
        buffer_count=2,
    )
    (inputs / "double-buffer.json").write_text(json.dumps(residency, indent=2) + "\n")
    original = Path("configs/memconf/2F3M_grouped_planes.yaml")
    memory = original.read_text().replace(
        "../ramulator/h100_hbm3_3stack.yaml",
        str((original.parent / "../ramulator/h100_hbm3_3stack.yaml").resolve()),
    )
    memory = memory.replace(
        "    allow_program: false\n",
        "    allow_program: false\n    statistics_interval_ps: 1000000\n    enable_read_sram: true\n    read_sram_access_latency_ns: 20\n    read_sram_bandwidth_GBps_per_stack: 1536\n",
    )
    (inputs / "double-buffer.yaml").write_text(memory)
    manifest = dict(
        batch=4,
        seed=args.seed,
        contexts=params["CONTEXTS"],
        shared_AB=True,
        fixed_C=[16, 64, 16],
        FA3=dict(TILE=params["TILE"], S=params["S"]),
        hardware=hardware,
        candidates=tiles,
        concurrency=2,
    )
    metadata = inputs / "experiment.json"
    if metadata.exists() and json.loads(metadata.read_text()) != json.loads(
        json.dumps(manifest)
    ):
        raise ValueError("use a new output directory for different search inputs")
    metadata.write_text(json.dumps(manifest, indent=2) + "\n")
    commands = {}
    for _, bn, bk in tiles:
        for case in ("full-hbm", "double-buffer"):
            source = Path("frontend/asm/qwen3") / (
                "one_block.progir"
                if case == "full-hbm"
                else "flash-accel/one_block.progir"
            )
            program = inputs / f"{case}-n{bn}-k{bk}.progir"
            program.write_text(program_variant(source.read_text(), bn, bk))
            name = f"{case}-n{bn}-k{bk}"
            command = [
                sys.executable,
                "frontend/tools/run_qwen_simulations.py",
                "--params",
                str(inputs / "b4.json"),
                "--source",
                str(program),
                "--lib",
                str(lib),
                "--runner",
                args.runner,
                "--architecture",
                "configs/v0.1.0/h100-sxm.yaml",
                "--memory",
                "configs/ramulator/h100-sxm-ramulator.yaml"
                if case == "full-hbm"
                else str(inputs / "double-buffer.yaml"),
                "--out",
                str(args.out / name),
            ]
            if case == "double-buffer":
                command.extend(
                    ("--prefetch-config", str(inputs / "double-buffer.json"))
                )
            commands[name] = command
    (args.out / "commands.sh").write_text(
        "#!/bin/sh\nset -e\nexport PYTHONPATH=frontend/src\n"
        + "\n".join(shlex.join(c) for c in commands.values())
        + "\n"
    )
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--runner", default="build-backends/gtsim_program")
    parser.add_argument("--seed", default=20261106, type=int)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run only the first candidate for each case",
    )
    args = parser.parse_args()
    commands = prepare(args)
    selected = list(commands.items())[:2] if args.pilot else list(commands.items())

    def run(item):
        name, command = item
        folder = args.out / name
        folder.mkdir(exist_ok=True)
        result = folder / "result.json"
        if result.exists() and json.loads(result.read_text()).get("returncode") == 0:
            return name, 0
        started = perf_counter()
        with (folder / "driver.log").open("w") as log:
            done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        (folder / "wall-time.json").write_text(
            json.dumps(
                dict(
                    elapsed_seconds=perf_counter() - started, returncode=done.returncode
                ),
                indent=2,
            )
            + "\n"
        )
        return name, done.returncode

    started = perf_counter()
    with ThreadPoolExecutor(max_workers=2) as workers:
        records = list(workers.map(run, selected))
    failed = [name for name, code in records if code]
    outcome = dict(
        selected=len(selected), failed=failed, elapsed_seconds=perf_counter() - started
    )
    (args.out / ("pilot.json" if args.pilot else "run.json")).write_text(
        json.dumps(outcome, indent=2) + "\n"
    )
    print(json.dumps(outcome), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
