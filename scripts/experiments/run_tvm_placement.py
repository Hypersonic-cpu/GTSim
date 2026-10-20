#!/usr/bin/env python3
"""Reproduce matched TVM GEMM device-placement experiments; no schedule search."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tvm"))
from gemm import (ARCH, cycles, generate, hardware_config, inspect_graph,
                  memory_statistics, run_simulation, source_hashes)
from h100_schedule import GemmSchedule
from lower_gemm import lower_gemm
from pinned_tvm import load_tvm, TVM_COMMIT
from semantic_checks import check_semantics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "validation/results/tvm/placement")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    out, build = args.out_dir.resolve(), args.build_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tvm, records = load_tvm(), []
    for profile in ("colocated-3F2M.yaml", "colocated-2F3M.yaml"):
        memory = ROOT / "configs/memconf" / profile
        hardware = hardware_config(build, memory=memory)
        for dimensions in ((512, 512, 512), (2048, 4096, 512)):
            matched = []
            for placement, tiers in (("baseline", ("HBM",) * 3), ("hybrid", ("HBM", "HBF", "HBM"))):
                directory = out / memory.stem / "x".join(map(str, dimensions)) / placement
                path, evidence = generate(tvm, dimensions, GemmSchedule(), hardware, directory, tiers)
                func = tvm.ir.load_json((directory / "scheduled.json").read_text())["gemm"]
                ir, _ = lower_gemm(tvm, func)
                # Audit boundary CTAs including the incomplete final wave.
                audit = check_semantics(tvm, func, ir, selected_ctas=sorted({0, len(ir.ctas) - 1, 123} & set(range(len(ir.ctas)))))
                graph = inspect_graph(build, memory, path)
                (directory / "materialized-dag.json").write_text(json.dumps(graph, separators=(",", ":")) + "\n")
                graph_digest = hashlib.sha256(json.dumps({key: graph[key] for key in ("ctas", "nodes")},
                                                          sort_keys=True).encode()).hexdigest()
                commands, samples, stats_samples, links = [], [], [], []
                for repeat in range(args.repeats):
                    log = directory / f"simulation-{repeat + 1}.log"
                    commands.append(run_simulation(build, ARCH, memory, path, log))
                    text = log.read_text()
                    samples.append(cycles(text))
                    stats_samples.append(memory_statistics(text))
                    links.append([line.strip() for line in text.splitlines() if "transferred bytes:" in line])
                assert len(set(samples)) == 1, "nondeterministic cycles"
                assert all(stats == stats_samples[0] for stats in stats_samples), "nondeterministic traffic"
                logical = Counter()
                for node in graph["nodes"]:
                    if node["op"] in ("cp.dram2sram_tma", "cp.sram2dram_tma"):
                        logical[graph["tensors"][node["tensor"]]["name"]] += 2 * node["region"][2] * node["region"][3]
                record = {"profile": profile, "dimensions": dimensions, "placement": placement,
                          "tensor_tiers": tiers, "cycles": samples, "statistics": stats_samples[0],
                          "placement_neutral_ir_sha256": hashlib.sha256(path.read_bytes().replace(b"tier=HBF", b"tier=HBM")).hexdigest(),
                          "links": links[0], "tensors": graph["tensors"], "nodes": len(graph["nodes"]),
                          "op_counts": dict(Counter(n["op"] for n in graph["nodes"])),
                          "complete_edges": sum(len(n["after_complete"]) for n in graph["nodes"]),
                          "issue_edges": sum(len(n["after_issue"]) for n in graph["nodes"]),
                          "logical_tma_bytes": dict(logical), "compute_graph_sha256": graph_digest,
                          "semantic_audit": audit, "waves": evidence["waves"], "commands": commands,
                          "artifact_directory": str(directory.relative_to(out))}
                records.append(record)
                matched.append(record)
                print(profile, dimensions, placement, samples, flush=True)
            baseline, hybrid = matched
            assert baseline["compute_graph_sha256"] == hybrid["compute_graph_sha256"], "placement changed graph"
            assert baseline["placement_neutral_ir_sha256"] == hybrid["placement_neutral_ir_sha256"], "placement changed IR beyond device declarations"
            assert baseline["statistics"]["HBF"]["requests"] == 0
            assert hybrid["statistics"]["HBF"]["read_bytes"] > 0
            assert hybrid["statistics"]["HBF"]["write_bytes"] == 0
            assert hybrid["statistics"]["HBF"]["latency"] >= 7920
            for record in matched:
                record["baseline_speedup"] = baseline["cycles"][0] / record["cycles"][0]
    result = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "tvm_commit": TVM_COMMIT, "source_hashes": source_hashes(),
              "schedule": {"BM": 128, "BN": 256, "BK": 64, "stages": 2, "warps": 12, "sms": 132},
              "runs": records}
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["| Topology | M×N×K | Placement | Cycles | Baseline speedup | HBM read/write bytes | HBF read/write bytes | HBM/HBF avg latency |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in records:
        hbm, hbf = r["statistics"]["HBM"], r["statistics"]["HBF"]
        lines.append(f"| {'3F2M' if r['profile'] == 'colocated-3F2M.yaml' else '2F3M'} | {'×'.join(map(str,r['dimensions']))} | {r['placement']} | {r['cycles'][0]} | {r['baseline_speedup']:.4f}× | {int(hbm['read_bytes'])}/{int(hbm['write_bytes'])} | {int(hbf['read_bytes'])}/{int(hbf['write_bytes'])} | {hbm['latency']:.2f}/{hbf['latency']:.2f} |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print(out / "results.json")


if __name__ == "__main__":
    main()
