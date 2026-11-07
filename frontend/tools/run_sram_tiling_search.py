"""B=4 streaming double-buffer sweep; reuse the frozen five-stack HBM controls."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys
from time import perf_counter


def prepare(args):
    original = args.baseline / "inputs"
    frozen = json.loads((original / "experiment.json").read_text())
    tiles = [t for t in frozen["candidates"] if t[1] != 512 and t[2] != 16]
    assert len(tiles) == 15
    commands = []
    for mib, divisor in ((8, 4), (4, 8)):
        root = args.out / f"sram-{mib}mib"
        inputs = root / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        for name in ("lib", "b4.json"):
            dest = inputs / name
            if not dest.exists():
                dest.symlink_to((original / name).resolve())
        config = json.loads((original / "double-buffer.json").read_text())
        config.update(
            plane_bytes=config["plane_bytes"] // divisor,
            stack_bytes=config["stack_bytes"] // divisor,
            streaming=True,
        )
        (inputs / "double-buffer.json").write_text(json.dumps(config, indent=2) + "\n")
        memory = (original / "double-buffer.yaml").read_text()
        memory = memory.replace(
            "sram_bytes_per_plane: 32768",
            f"sram_bytes_per_plane: {config['plane_bytes']}",
        )
        memory = memory.replace(
            "base_die_sram_bytes_per_stack: 8388608",
            f"base_die_sram_bytes_per_stack: {config['stack_bytes']}",
        )
        # Programming is disabled; retain only the minimum dormant write quota.
        memory = memory.replace(
            "    allow_program: false\n",
            "    allow_program: false\n    write_pages_per_channel: 1\n",
        )
        (inputs / "double-buffer.yaml").write_text(memory)
        manifest = dict(
            frozen,
            candidates=tiles,
            sram_mib_per_stack=mib,
            buffer_count=2,
            streaming=True,
            full_hbm_reused_from=str(args.baseline.resolve()),
        )
        (inputs / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for _, bn, bk in tiles:
            control = args.baseline / f"full-hbm-n{bn}-k{bk}"
            assert json.loads((control / "result.json").read_text())["returncode"] == 0
            link = root / control.name
            if not link.exists():
                link.symlink_to(control.resolve(), target_is_directory=True)
            name = f"double-buffer-n{bn}-k{bk}"
            source = original / f"{name}.progir"
            command = [
                sys.executable,
                "frontend/tools/run_qwen_simulations.py",
                "--params",
                str(inputs / "b4.json"),
                "--source",
                str(source),
                "--lib",
                str(inputs / "lib"),
                "--runner",
                args.runner,
                "--architecture",
                "configs/v0.1.0/h100-sxm.yaml",
                "--memory",
                str(inputs / "double-buffer.yaml"),
                "--prefetch-config",
                str(inputs / "double-buffer.json"),
                "--out",
                str(root / name),
            ]
            if args.max_cycles is not None:
                command.extend(("--max-cycles", str(args.max_cycles)))
            commands.append((root / name, command))
    (args.out / "commands.sh").write_text(
        "#!/bin/sh\nexport PYTHONPATH=frontend/src\n"
        + "\n".join(shlex.join(c) for _, c in commands)
        + "\n"
    )
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("validation/results/flash-accel/b4-shared-ab-tiling"),
    )
    parser.add_argument("--runner", default="build-backends/gtsim_program")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-cycles", type=int)
    args = parser.parse_args()
    commands = prepare(args)
    if args.prepare_only:
        print(
            f"Prepared {len(commands)} NEW DoubleBuffer simulations; 15 reused FullHBM controls."
        )
        return

    reused = sum(
        (folder / "result.json").exists()
        and json.loads((folder / "result.json").read_text())["returncode"] == 0
        for folder, _ in commands
    )

    def run(item):
        folder, command = item
        folder.mkdir(exist_ok=True)
        result = folder / "result.json"
        if result.exists() and json.loads(result.read_text()).get("returncode") == 0:
            return str(folder), 0
        start = perf_counter()
        with (folder / "driver.log").open("w") as log:
            done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        (folder / "wall-time.json").write_text(
            json.dumps(
                dict(
                    elapsed_seconds=perf_counter() - start, returncode=done.returncode
                ),
                indent=2,
            )
            + "\n"
        )
        return str(folder), done.returncode

    start = perf_counter()
    with ThreadPoolExecutor(max_workers=2) as workers:
        records = list(workers.map(run, commands))
    outcome = dict(
        new_simulations=len(commands) - reused,
        reused_double_buffer=reused,
        reused_full_hbm=15,
        failed=[name for name, code in records if code],
        elapsed_seconds=perf_counter() - start,
    )
    (args.out / "run.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome), flush=True)
    if outcome["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
