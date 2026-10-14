#!/usr/bin/env python3
"""Run reproducible GEMM and FA3 memory-topology baseline sweeps."""

from __future__ import annotations

import argparse
import csv
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARCH_CONFIG = ROOT / "configs/v0.1.0/h100-sxm.yaml"
MEMORY_CONFIGS = {
    "FullHBM": ROOT / "configs/v0.1.0/memory_full_hbm-0F5M.yaml",
    "FullHBF": ROOT / "configs/v0.1.0/memory_full_hbf-5F0M.yaml",
    "CoLocated": ROOT / "configs/v0.1.0/memory_co_located-3F2M.yaml",
    "Cascaded": ROOT / "configs/v0.1.0/memory_cascaded-3F2M.yaml",
}
TOPOLOGIES = tuple(MEMORY_CONFIGS)
TOTAL_CYCLES = re.compile(r"^Total cycles:\s*(\d+)\s*$", re.MULTILINE)


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def gemm_cases() -> list[dict[str, Any]]:
    cases = [
        {"name": "gemm_1024_square", "workload": "gemm", "parameters": {"M": 1024, "N": 1024, "K": 1024}},
        {"name": "gemm_2048_square", "workload": "gemm", "parameters": {"M": 2048, "N": 2048, "K": 2048}},
        {"name": "gemm_4096_square", "workload": "gemm", "parameters": {"M": 4096, "N": 4096, "K": 4096}},
    ]
    for k in (512, 1024, 2048, 4096):
        if k == 2048:  # Already included by the square-size sweep.
            continue
        cases.append({
            "name": f"gemm_2048x2048_k{k}",
            "workload": "gemm",
            "parameters": {"M": 2048, "N": 2048, "K": k},
        })
    cases.append({
        "name": "gemm_2048x1536_k8192",
        "workload": "gemm",
        "parameters": {"M": 2048, "N": 1536, "K": 8192},
        "purpose": "L2 capacity and HBF performance",
        "metadata": {
            "a_b_input_bytes": 58_720_256,
            "a_b_input_mib": 56,
            "configured_l2_mib": 50,
        },
    })
    return cases


def fa3_cases() -> list[dict[str, Any]]:
    dimensions = (
        (1, 512), (1, 1024), (1, 4096),
        (4, 512), (4, 1024),
        (8, 4096),
        (2, 1024),
        (1, 1536),
    )
    cases = []
    for batch, sequence in dimensions:
        case = {
            "name": f"fa3_b{batch}_s{sequence}",
            "workload": "fa3",
            "parameters": {
                "batch": batch,
                "heads_q": 28,
                "heads_kv": 4,
                "seq_q": sequence,
                "seq_kv": sequence,
                "head_dim": 128,
            },
        }
        if (batch, sequence) == (2, 1024):
            case["purpose"] = "Batch-size sensitivity"
        elif (batch, sequence) == (1, 1536):
            case["purpose"] = "Sequence-length sensitivity"
        cases.append(case)
    return cases


def estimate_nodes(case: dict[str, Any]) -> int:
    """Estimate materialized DAG size to guard the host from oversized cases."""
    params = case["parameters"]
    if case["workload"] == "gemm":
        # Current GEMM builder: 133 nodes per K tile and 132 per output tile.
        output_tiles = (params["M"] // 128) * (params["N"] // 256)
        k_tiles = params["K"] // 64
        return output_tiles * (133 * k_tiles + 132)

    # Calibrated against the current FA3 builder's 196,672 nodes for
    # (batch=1, heads=28, seq_q=seq_kv=512). This scales with Q blocks and
    # KV tiles, the builder's two dominant loop dimensions.
    q_blocks = ceil_div(params["seq_q"], 128)
    kv_tiles = ceil_div(params["seq_kv"], 128)
    return params["batch"] * params["heads_q"] * q_blocks * kv_tiles * 439


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    ).stdout.strip())
    return commit, dirty


def write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "run_id", "workload", "case_name", "topology", "memory_config",
        "parameters_json", "purpose", "metadata_json", "estimated_nodes", "status", "total_cycles",
        "slowdown_vs_full_hbm", "git_commit", "git_dirty", "duration_seconds",
        "return_code", "log_path", "command_json", "run_key",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in columns})
    temporary.replace(path)


def markdown_summary(records: list[dict[str, Any]], commit: str, dirty: bool) -> str:
    completed = sum(record["status"] == "completed" for record in records)
    guarded = sum(record["status"] == "guarded_node_limit" for record in records)
    failed = len(records) - completed - guarded
    lines = [
        "# Baseline Workload Sweep",
        "",
        f"- Git commit: `{commit}` (working tree dirty: `{str(dirty).lower()}`)",
        f"- Completed: {completed}; guarded by node limit: {guarded}; failed/timed out: {failed}",
        "- Slowdowns use completed FullHBM runs with matching workload parameters as the baseline.",
        "",
        "| Workload | Parameters | Estimated nodes | FullHBM cycles | FullHBM slowdown | FullHBF slowdown | CoLocated slowdown | Cascaded slowdown |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    cases: dict[str, dict[str, Any]] = {}
    for record in records:
        cases.setdefault(record["case_name"], record)
    topology_order = ("FullHBM", "FullHBF", "CoLocated", "Cascaded")
    ordered_cases = sorted(
        cases.items(),
        key=lambda item: ({"gemm": 0, "fa3": 1}.get(item[1]["workload"], 2), item[0]),
    )
    for case_name, representative in ordered_cases:
        matching = [record for record in records if record["case_name"] == case_name]
        by_topology = {record["topology"]: record for record in matching}
        baseline = by_topology.get("FullHBM", {})
        baseline_cycles = baseline.get("total_cycles") if baseline.get("status") == "completed" else None
        if baseline_cycles:
            base_cell = f"{baseline_cycles:,}"
            full_slowdown = "1.000×"
        else:
            base_cell = cell_status(baseline)
            full_slowdown = cell_status(baseline)
        slowdowns = []
        for topology in topology_order[1:]:
            record = by_topology.get(topology, {})
            if record.get("status") == "completed" and baseline_cycles:
                slowdowns.append(f"{record['total_cycles'] / baseline_cycles:.3f}×")
            else:
                slowdowns.append(cell_status(record))
        params = ", ".join(f"{key}={value}" for key, value in representative["parameters"].items())
        lines.append(
            f"| {case_name} | {params} | {representative['estimated_nodes']:,} | "
            f"{base_cell} | {full_slowdown} | {slowdowns[0]} | {slowdowns[1]} | {slowdowns[2]} |"
        )

    described_cases: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("purpose"):
            described_cases.setdefault(record["case_name"], record)
    if described_cases:
        lines.extend(["", "## Added workload purposes", ""])
        for case_name, record in sorted(described_cases.items()):
            details = record.get("metadata", {})
            extra = ""
            if "a_b_input_mib" in details:
                extra = (
                    f" Combined A+B input is {details['a_b_input_mib']} MiB "
                    f"against {details['configured_l2_mib']} MiB configured L2."
                )
            lines.append(f"- `{case_name}`: {record['purpose']}.{extra}")

    non_completed = [record for record in records if record["status"] != "completed"]
    if non_completed:
        lines.extend(["", "## Runs not completed", ""])
        for record in non_completed:
            detail = "node limit" if record["status"] == "guarded_node_limit" else record["status"]
            lines.append(
                f"- `{record['case_name']}` / `{record['topology']}`: {detail} "
                f"({record['estimated_nodes']:,} estimated DAG nodes)."
            )
    lines.extend([
        "",
        "Full per-run parameters, commands, commit IDs, cycle counts, and logs are in `results.csv`, `results.json`, and `logs/`.",
        "",
    ])
    return "\n".join(lines)


def cell_status(record: dict[str, Any]) -> str:
    status = record.get("status", "not run")
    if status == "guarded_node_limit":
        return "guarded"
    if status in ("failed", "timeout"):
        return status
    return "—"


def persist(output_dir: Path, payload: dict[str, Any], commit: str, dirty: bool) -> None:
    records = payload["runs"]
    write_json(output_dir / "results.json", payload)
    write_csv(output_dir / "results.csv", records)
    summary_tmp = output_dir / "summary.md.tmp"
    summary_tmp.write_text(markdown_summary(records, commit, dirty))
    summary_tmp.replace(output_dir / "summary.md")


def build_command(binary: Path, case: dict[str, Any], memory_config: Path) -> list[str]:
    params = case["parameters"]
    command = [str(binary), str(ARCH_CONFIG), "--memory-config", str(memory_config)]
    if case["workload"] == "gemm":
        command.extend(str(params[key]) for key in ("M", "N", "K"))
    else:
        command.extend(str(params[key]) for key in ("batch", "heads_q", "heads_kv", "seq_q", "seq_kv"))
    command.extend(params.get("kernel_args", []))
    return command


def execute_record(record: dict[str, Any], command: list[str], log_path: Path, timeout_seconds: int) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds or None,
                check=False,
            )
        record["duration_seconds"] = round(time.perf_counter() - start, 3)
        record["return_code"] = completed.returncode
        output = log_path.read_text(errors="replace")
        cycle_matches = TOTAL_CYCLES.findall(output)
        if completed.returncode == 0 and cycle_matches:
            record["status"] = "completed"
            record["total_cycles"] = int(cycle_matches[-1])
            record["slowdown_vs_full_hbm"] = "pending"
        else:
            record["status"] = "failed"
            record["error"] = "no total-cycle line" if not cycle_matches else f"exit status {completed.returncode}"
    except subprocess.TimeoutExpired:
        record["duration_seconds"] = round(time.perf_counter() - start, 3)
        record["status"] = "timeout"
        record["error"] = f"exceeded {timeout_seconds} seconds"

    if record["status"] == "failed" and record.get("return_code") == 2 and "infeasible kernel" in log_path.read_text(errors="replace"):
        record["status"] = "infeasible"
        message = re.search(r"infeasible kernel:.*", output)
        if message: record["error"] = message.group(0)
    output = log_path.read_text(errors="replace")
    match = re.search(r"Generated DAG with (\d+) nodes", output)
    if match: record["actual_nodes"] = int(match.group(1))
    match = re.search(r"shared_bytes=(\d+) register_bytes=(\d+) resident_CTA_limit=(\d+)", output)
    if match: record["resources"] = dict(zip(("shared_bytes", "register_bytes", "resident_cta_limit"), map(int, match.groups())))
    tiers = re.findall(r"^\s*(HBM|HBF) requests: (\d+) \(read (\d+), write (\d+)\), bytes: read (\d+), write (\d+), average latency: ([\d.]+) cycles", output, re.MULTILINE)
    record["memory_statistics"] = {
        tier: dict(requests=int(requests), read_requests=int(reads), write_requests=int(writes),
                   read_bytes=int(read_bytes), write_bytes=int(write_bytes), average_latency_cycles=float(latency))
        for tier,requests,reads,writes,read_bytes,write_bytes,latency in tiers
    }
    return record


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("all", "gemm", "fa3"), default="all")
    parser.add_argument("--topology", choices=("all", *TOPOLOGIES), default="all")
    all_cases = gemm_cases() + fa3_cases()
    parser.add_argument("--case", action="append", choices=tuple(case["name"] for case in all_cases),
                        help="Run only the named workload case; may be provided multiple times.")
    parser.add_argument("--output-dir", default="validation/results/gemm_fa3-param-sweep/results/baseline")
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--kernel-args", default="", help="Quoted kernel CLI options for this variant.")
    parser.add_argument("--max-estimated-nodes", type=int, default=2_000_000,
                        help="Skip larger estimated DAGs; use 0 to disable this safety guard (default: 2,000,000).")
    parser.add_argument("--timeout-seconds", type=int, default=600,
                        help="Per-run timeout; use 0 to disable (default: 600).")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected matrix without launching workloads or writing files.")
    parser.add_argument("--resume", action="store_true",
                        help="Compatibility flag; completed matching runs are reused by default.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Force selected cases to rerun and replace their matching result rows.")
    args = parser.parse_args()
    if args.parallelism < 1:
        parser.error("parallelism must be positive")
    if args.max_estimated_nodes < 0 or args.timeout_seconds < 0:
        parser.error("node limit and timeout must be non-negative")

    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = ROOT / build_dir
    binaries = {"gemm": build_dir / "gtsim_gemm", "fa3": build_dir / "gtsim_fa3"}
    selected_topologies = TOPOLOGIES if args.topology == "all" else (args.topology,)
    cases = all_cases
    if args.workload != "all":
        cases = [case for case in cases if case["workload"] == args.workload]
    if args.case:
        selected_names = set(args.case)
        cases = [case for case in cases if case["name"] in selected_names]
    if not cases:
        parser.error("the workload and case selectors match no cases")

    for path in (ARCH_CONFIG, *MEMORY_CONFIGS.values()):
        if not path.is_file():
            parser.error(f"configuration file not found: {path}")
    for workload in {case["workload"] for case in cases}:
        if not binaries[workload].is_file():
            parser.error(f"simulator binary not found: {binaries[workload]} (build the project first)")

    if args.kernel_args:
        for case in cases:
            case["parameters"] = dict(case["parameters"], kernel_args=shlex.split(args.kernel_args))
    commit, dirty = git_metadata()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    planned = []
    for case in cases:
        for topology in selected_topologies:
            planned.append((case, topology, estimate_nodes(case)))
    if args.dry_run:
        print(f"Git commit: {commit} (working tree dirty: {str(dirty).lower()})")
        for case, topology, nodes in planned:
            status = "GUARDED" if args.max_estimated_nodes and nodes > args.max_estimated_nodes else "RUN"
            command = build_command(binaries[case["workload"]], case, MEMORY_CONFIGS[topology])
            print(f"{status:7} {case['name']:24} {topology:10} ~{nodes:>12,} nodes :: {' '.join(command)}")
        print(f"Planned {len(planned)} runs; dry-run wrote no files.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    if (output_dir / "results.json").exists():
        payload = json.loads((output_dir / "results.json").read_text())
    else:
        payload = {"schema_version": 1, "runs": []}
    existing_identities = {record_identity(record) for record in payload["runs"]}

    executor = ThreadPoolExecutor(max_workers=args.parallelism)
    futures = {}
    for case, topology, nodes in planned:
        binary = binaries[case["workload"]]
        memory_config = MEMORY_CONFIGS[topology]
        command = build_command(binary, case, memory_config)
        fingerprint = {
            "git_commit": commit,
            "binary_sha256": sha256_file(binary),
            "architecture_sha256": sha256_file(ARCH_CONFIG),
            "memory_config_sha256": sha256_file(memory_config),
            "workload": case["workload"],
            "case_name": case["name"],
            "parameters": case["parameters"],
            "topology": topology,
        }
        run_key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        identity = case_identity(case, topology)
        run_id = f"{case['name']}__{topology}__{args.variant}"
        log_path = logs_dir / f"{run_id}.log"
        if not args.overwrite and (identity in existing_identities or log_path.exists()):
            print(f"SKIP    {case['name']} / {topology}: prior result or run log already exists")
            continue

        record: dict[str, Any] = {
            "run_id": run_id,
            "workload": case["workload"],
            "case_name": case["name"],
            "topology": topology,
            "memory_config": memory_config.name,
            "parameters": case["parameters"],
            "parameters_json": json.dumps(case["parameters"], sort_keys=True),
            "purpose": case.get("purpose", ""),
            "metadata": case.get("metadata", {}),
            "metadata_json": json.dumps(case.get("metadata", {}), sort_keys=True),
            "estimated_nodes": nodes,
            "status": "pending",
            "total_cycles": "",
            "slowdown_vs_full_hbm": "",
            "git_commit": commit,
            "git_dirty": dirty,
            "duration_seconds": "",
            "return_code": "",
            "log_path": str(log_path.relative_to(output_dir)),
            "command": command,
            "command_json": json.dumps(command),
            "run_key": run_key,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if args.max_estimated_nodes and nodes > args.max_estimated_nodes:
            record["status"] = "guarded_node_limit"
            payload["runs"] = [old for old in payload["runs"] if record_identity(old) != identity]
            payload["runs"].append(record)
            existing_identities.add(identity)
            print(f"GUARDED {run_id}: ~{nodes:,} estimated nodes exceeds {args.max_estimated_nodes:,}")
            persist(output_dir, payload, commit, dirty)
            continue

        future = executor.submit(execute_record, record, command, log_path, args.timeout_seconds)
        futures[future] = identity

    for future in as_completed(futures):
        record = future.result()
        identity = futures[future]
        payload["runs"] = [old for old in payload["runs"] if record_identity(old) != identity]
        payload["runs"].append(record)
        update_slowdowns(payload["runs"])
        print(f"{record['status'].upper():8} {record['run_id']}: {record.get('total_cycles', '—')} cycles", flush=True)
        persist(output_dir, payload, commit, dirty)
    executor.shutdown()

    update_slowdowns(payload["runs"])
    persist(output_dir, payload, commit, dirty)
    print(f"Results written to {output_dir}")
    return 0 if all(record["status"] in ("completed", "guarded_node_limit") for record in payload["runs"]) else 1


def case_identity(case: dict[str, Any], topology: str) -> tuple[str, str, str]:
    return (case["name"], topology, json.dumps(case["parameters"], sort_keys=True))


def record_identity(record: dict[str, Any]) -> tuple[str, str, str]:
    return (record["case_name"], record["topology"], json.dumps(record["parameters"], sort_keys=True))


def update_slowdowns(records: list[dict[str, Any]]) -> None:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_case.setdefault(record["case_name"], {})[record["topology"]] = record
    for topologies in by_case.values():
        baseline = topologies.get("FullHBM", {})
        baseline_cycles = baseline.get("total_cycles") if baseline.get("status") == "completed" else None
        for topology, record in topologies.items():
            if topology == "FullHBM":
                record["slowdown_vs_full_hbm"] = 1.0 if baseline_cycles else ""
            elif baseline_cycles and record.get("status") == "completed":
                record["slowdown_vs_full_hbm"] = round(record["total_cycles"] / baseline_cycles, 6)


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        print("Sweep interrupted; completed records are saved and will be skipped on the next run.", file=sys.stderr)
        raise SystemExit(130)
