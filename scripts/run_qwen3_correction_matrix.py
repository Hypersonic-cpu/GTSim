"""Run one-layer gates before two-layer gates and audit actual stream results."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

CONTEXTS = {
    1: [10000],
    8: [10000, 12000, 16000, 20000, 24000, 28000, 30000, 32000],
    16: [
        10000,
        11000,
        12000,
        13000,
        14000,
        16000,
        18000,
        20000,
        22000,
        24000,
        26000,
        28000,
        30000,
        31000,
        31500,
        32000,
    ],
}


def audit(directory, batch, layers):
    data = json.loads((directory / "kernel-timings.json").read_text())
    assert data["outcome"] == "PASS"
    assert data["context_lengths"] == CONTEXTS[batch]
    assert data["effective_context_lengths"] == [n + 1 for n in CONTEXTS[batch]]
    assert len(data["kernels"]) == layers * (18 if batch == 1 else 19)
    previous = 0
    for kernel in data["kernels"]:
        assert previous <= kernel["launch_cycle"] <= kernel["start_cycle"]
        assert (
            kernel["start_cycle"]
            < kernel["device_complete_cycle"]
            <= kernel["complete_cycle"]
        )
        previous = kernel["complete_cycle"]
    assert previous == data["stream"]["end_cycle"] > 0
    for layer in data["layers_summary"]:
        assert layer["new_tokens"] == batch
        assert layer["layer_output_shape"] == [batch, 2048]
        assert layer["expert_assignment_count"] == batch * 8
        assert sum(layer["expert_token_counts"].values()) == batch * 8
    assert all(
        data["memory_statistics"][key]
        for key in (
            "persistent_gpu",
            "persistent_memory",
            "persistent_l2",
            "stable_allocations",
        )
    )
    assert (directory / "simulation.log").stat().st_size > 0
    manifest = json.loads((directory / "manifest.json").read_text())
    assert all((directory / item["path"]).is_file() for item in manifest["artifacts"])
    for item in manifest["artifacts"]:
        with (directory / item["path"]).open("rb") as artifact:
            assert (
                hashlib.file_digest(artifact, "sha256").hexdigest() == item["sha256"]
            ), item["path"]
    return {
        "batch": batch,
        "layers": layers,
        "cycles": previous,
        "kernels": len(data["kernels"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    results = []
    for layers in (1, 2):
        for batch, contexts in CONTEXTS.items():
            directory = args.out_dir / f"b{batch}-l{layers}"
            if not args.audit_only:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "gtsim_frontend.cli",
                        "decode",
                        "--batch-size",
                        str(batch),
                        "--layers",
                        str(layers),
                        "--context-lengths",
                        *map(str, contexts),
                        "--runner",
                        args.runner,
                        "--architecture",
                        "configs/v0.1.0/h100-sxm.yaml",
                        "--memory-config",
                        "configs/v0.1.0/memory_full_hbm-0F5M.yaml",
                        "--out-dir",
                        str(directory),
                    ],
                    check=True,
                )
            results.append(audit(directory, batch, layers))
            print(results[-1], flush=True)
    (args.out_dir / "matrix-audit.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
