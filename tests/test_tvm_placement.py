#!/usr/bin/env python3
"""Matched native-TVM placement checks, including simulator routing and latency."""
import argparse
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/tvm"))
from gemm import (ARCH, generate, hardware_config, inspect_graph,
                  memory_statistics, run_simulation, translate)
from h100_schedule import GemmSchedule, scheduled_gemm
from pinned_tvm import load_tvm
from lower_gemm import lower_gemm
from semantic_checks import check_semantics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    tvm = load_tvm()
    with tempfile.TemporaryDirectory() as tmp:
        for profile in ("colocated-3F2M.yaml", "colocated-2F3M.yaml"):
            memory = ROOT / "configs/memconf" / profile
            hardware = hardware_config(build, memory=memory)
            # A small two-stage case and a real two-wave case with an incomplete
            # final wave. The architecture/SM count is unchanged in each pair.
            for dimensions in ((128, 256, 128), (128, (hardware["num_sms"] + 1) * 256, 128)):
                graphs, stats, neutral_ir = [], [], []
                for placement, tiers in (("baseline", ("HBM",) * 3), ("hybrid", ("HBM", "HBF", "HBM"))):
                    directory = Path(tmp) / profile / str(dimensions[1]) / placement
                    path, evidence = generate(tvm, dimensions, GemmSchedule(), hardware, directory, tiers)
                    assert evidence["tensor_tiers"] == list(tiers)
                    neutral_ir.append(path.read_bytes().replace(b"tier=HBF", b"tier=HBM"))
                    func = tvm.ir.load_json((directory / "scheduled.json").read_text())["gemm"]
                    assert [p.scope() for p in func.params] == ["global." + t.lower() for t in tiers]
                    ir, _ = lower_gemm(tvm, func)
                    check_semantics(tvm, func, ir, selected_ctas=[0, len(ir.ctas) - 1])
                    graph = inspect_graph(build, memory, path)
                    graphs.append(graph)
                    assert [t["device"] for t in graph["tensors"]] == list(tiers)
                    for tensor in graph["tensors"]:
                        assert (tensor["base"] >= 1 << 60) == (tensor["device"] == "HBF")
                    run_simulation(build, ARCH, memory, path, directory / "simulation.log")
                    stats.append(memory_statistics((directory / "simulation.log").read_text()))
                assert neutral_ir[0] == neutral_ir[1]
                assert graphs[0]["nodes"] == graphs[1]["nodes"]
                assert graphs[0]["ctas"] == graphs[1]["ctas"]
                assert stats[0]["HBF"]["requests"] == 0
                assert stats[1]["HBF"]["read_bytes"] > 0 and stats[1]["HBF"]["write_bytes"] == 0
                assert stats[1]["HBF"]["latency"] >= 7920
                assert stats[0]["HBM"]["read_bytes"] == stats[1]["HBM"]["read_bytes"] + stats[1]["HBF"]["read_bytes"]
        func = scheduled_gemm(tvm, 128, 256, 128, tensor_tiers=("HBM", "HBF", "HBM"))
        full_hbm = hardware_config(build)
        try:
            translate(tvm, func, full_hbm, Path(tmp) / "unavailable")
        except ValueError:
            pass
        else:
            raise AssertionError("frontend silently remapped HBF on FullHBM")
    print("TVM device scopes, address apertures, graph equivalence, two-wave semantics and real HBF traffic passed")


if __name__ == "__main__":
    main()
