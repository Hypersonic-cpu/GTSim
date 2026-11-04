"""Exercise a late prefetch across two kernels through the public IR-05 runner."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def run(build, directory):
    directory = Path(directory)
    source = ROOT / "frontend/asm/gemm/reference/two_gemms.progir"
    program = directory / "two_gemms.progir"
    program.write_text(
        source.read_text().replace(
            "@W1 : tensor<256x512xfp16> storage<HBM>",
            "@W1 : tensor<256x512xfp16> storage<HBF>",
        )
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "compile",
            str(program),
            "--lib",
            "frontend/lib",
            "--param",
            "NUM_SMS=132",
            "-o",
            str(directory / "two_gemms.dag.json"),
            "--runtime-dir",
            str(directory),
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "frontend/src")},
        cwd=ROOT,
        check=True,
    )
    config = (ROOT / "configs/memconf/2F3M_grouped_planes.yaml").read_text()
    config = config.replace(
        "../ramulator/h100_hbm3_3stack.yaml",
        str(ROOT / "configs/ramulator/h100_hbm3_3stack.yaml"),
    )
    config = config.replace(
        "    allow_program: false",
        "    allow_program: false\n"
        "    enable_read_sram: true\n"
        "    read_sram_access_latency_ns: 20\n"
        "    read_sram_bandwidth_GBps_per_stack: 1536",
    )
    memory = directory / "memory.yaml"
    memory.write_text(config)
    manifest = directory / "program.gtsir"

    def simulate(path):
        stdout = subprocess.check_output(
            [
                str(build / "gtsim_program"),
                str(ROOT / "configs/v0.1.0/h100-sxm.yaml"),
                "--memory-config",
                str(memory),
                str(path),
            ],
            text=True,
        )
        return json.loads(stdout.split("Program timing JSON: ", 1)[1])

    baseline = simulate(manifest)
    mixed = json.loads(manifest.read_text())
    weight = next(t for t in mixed["tensors"] if t["name"] == "W1")
    mixed["prefetches"] = [
        dict(id="next_weight", tensor=weight["id"], offset=0, bytes=262144)
    ]
    mixed["commands"].insert(
        0, dict(record="Command", kind="Prefetch", target="next_weight")
    )
    mixed["commands"].insert(
        -1, dict(record="Command", kind="Release", target="next_weight")
    )
    mixed["launches"][1]["read_aliases"] = ["next_weight"]
    overlap = directory / "overlap.gtsir"
    overlap.write_text(json.dumps(mixed))
    result = simulate(overlap)
    first, second = result["per_kernel"]
    assert first["complete_cycle"] == baseline["per_kernel"][0]["complete_cycle"]
    assert first["background_at_completion"] == second["background_at_launch"] == 1
    assert second["launch_cycle"] == first["complete_cycle"]
    assert second["background_at_completion"] == 0
    assert first["memory"]["HBF"]["counters"]["prefetch_bytes"] == 262144
    hbf = second["memory"]["HBF"]
    counters = hbf["counters"]
    assert counters["fallback_read_bytes"] > 0 and counters["resident_read_bytes"] > 0
    assert (
        counters["fallback_read_bytes"] + counters["resident_read_bytes"]
        == hbf["read_bytes"]
    )
    assert hbf["read_bytes"] == baseline["per_kernel"][1]["memory"]["HBF"]["read_bytes"]
    assert hbf["write_bytes"] == 0 and counters["sensed_bytes"] == 0
    print(json.dumps({"baseline": baseline, "overlap": result}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="gtsim-prefetch-") as directory:
        run(args.build_dir.resolve(), directory)
