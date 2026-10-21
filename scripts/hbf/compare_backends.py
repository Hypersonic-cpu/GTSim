#!/usr/bin/env python3
"""Compare declared HBF placements; at most two simulator children run at once.

Raw logs, commands, source hashes, device counters and resource utilization are
retained. Heterogeneous timings are predictions, never H100 hardware errors.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("0F5M_channels", "3F2M_naive", "3F2M_bank_plane", "3F2M_bank_plane_ocp",
            "1F4M_naive", "1F4M_bank_plane", "1F4M_bank_plane_ocp")
CASES = {"gemm": [(512, 512, 512), (1024, 1024, 1024), (2048, 4096, 512)],
         "fa3": [(1, 8, 8, 1536, 1536), (1, 8, 8, 512, 512)]}


def git(*arguments, cwd=ROOT):
    return subprocess.check_output(["git", *arguments], cwd=cwd, text=True).strip()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_case(build, output, workload, shape, profile):
    label = workload + "_" + "x".join(map(str, shape)) + "_" + profile
    command = [str(build / f"gtsim_{workload}"), str(ROOT / "configs/v0.1.0/h100-sxm.yaml"),
               "--memory-config", str(ROOT / "configs/memconf" / (profile + ".yaml"))]
    mixed = not profile.startswith("0F")
    if mixed:
        placement = "gemm_b_hbf" if workload == "gemm" else "fa3_q_hbf"
        command += ["--memory-config", str(ROOT / "configs/memconf" / ("placement_" + placement + ".yaml"))]
    command += list(map(str, shape))
    started = time.monotonic()
    with (output / (label + ".log")).open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        # Blocking wait, no status polling. wait4 reports this child's RSS only.
        _, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
    wall = time.monotonic() - started
    text = (output / (label + ".log")).read_text()
    if process.returncode or "Simulation completed at cycle" not in text or "Memory drained at cycle:" not in text:
        raise RuntimeError(f"{label} failed; see its raw log")
    cycles = int(re.search(r"^Total cycles: (\d+)$", text, re.M)[1])
    drained = int(re.search(r"Memory drained at cycle: (\d+)", text)[1])
    traffic = {}
    for tier, requests, reads, writes, read_bytes, write_bytes in re.findall(
            r"(HBM|HBF) requests: (\d+) \(read (\d+), write (\d+)\), bytes: read (\d+), write (\d+)", text):
        traffic[tier] = dict(zip(("requests", "reads", "writes", "read_bytes", "write_bytes"),
                                map(int, (requests, reads, writes, read_bytes, write_bytes))))
    hbf = traffic.get("HBF", {"read_bytes": 0, "write_bytes": 0})
    assert (hbf["read_bytes"] > 0) == mixed, label + ": intended HBF traffic missing"
    assert hbf["write_bytes"] == 0, label + ": unexpected HBF writes"
    detailed = "bank_plane" in profile
    match = re.search(r"^HBF detailed JSON: (.*)$", text, re.M)
    stats = json.loads(match[1]) if match else {}
    if detailed:
        assert stats["accepted_requests"] == stats["completed_requests"] == traffic["HBF"]["requests"]
        assert stats["failed_requests"] == stats["host_write_bytes"] == stats["programs"] == stats["erases"] == 0
        assert stats["host_read_bytes"] == sum(stats["stack_read_bytes"]) == sum(stats["channel_read_bytes"]) == hbf["read_bytes"]
        assert sum(stats["bank_reads"]) * 64 == sum(stats["plane_reads"]) * 64 == hbf["read_bytes"]
        stacks = int(profile[0]); banks = 256 if profile.endswith("ocp") else 512
        assert len(stats["channel_read_bytes"]) == stacks * 16
        assert len(stats["bank_reads"]) == stacks * banks and len(stats["plane_reads"]) == stacks * 512
    row = {"case": workload + "_" + "x".join(map(str, shape)), "configuration": profile,
           "placement": "all_hbm" if not mixed else ("B_hbf" if workload == "gemm" else "Q_hbf_synthetic"),
           "cycles": cycles, "drained_cycle": drained, "wall_seconds": round(wall, 6),
           "peak_rss_MiB": round(usage.ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024), 3),
           "hbm_read_bytes": traffic.get("HBM", {}).get("read_bytes", 0),
           "hbm_write_bytes": traffic.get("HBM", {}).get("write_bytes", 0),
           "hbf_read_bytes": hbf["read_bytes"], "hbf_write_bytes": hbf["write_bytes"],
           "hbf_achieved_GBps": round(hbf["read_bytes"] * 1980 / drained / 1000, 6),
           "sensed_bytes": stats.get("sensed_bytes", ""), "cache_hits_64B": stats.get("cache_hits", ""),
           "cache_misses_64B": stats.get("cache_misses", ""), "cache_evictions": stats.get("cache_evictions", ""),
           "queue_stalls": stats.get("queue_stalls", ""), "bank_wait_ps": stats.get("bank_wait_ps", ""),
           "transfer_wait_ps": stats.get("transfer_wait_ps", ""),
           "accepted_requests": stats.get("accepted_requests", ""), "completed_requests": stats.get("completed_requests", ""),
           "failed_requests": stats.get("failed_requests", ""), "programs": stats.get("programs", ""),
           "erases": stats.get("erases", "")}
    (output / (label + ".json")).write_text(json.dumps({"command": command, "row": row,
        "traffic": traffic, "hbf_statistics": stats, "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime}, indent=2) + "\n")
    print(f"completed {label}: {cycles} cycles, {wall:.2f}s", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "validation/results/hbf-controller-20261021/sweep")
    args = parser.parse_args()
    output = args.out_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    build = args.build_dir.resolve()
    files = [*ROOT.glob("configs/memconf/*.yaml"), ROOT / "configs/v0.1.0/h100-sxm.yaml",
             *ROOT.glob("3rdparty/hbfine/src/*.cpp"), *ROOT.glob("3rdparty/hbfine/include/hbfine/*.hpp"),
             ROOT / "3rdparty/hbfine/CMakeLists.txt", ROOT / "CMakeLists.txt",
             build / "gtsim_gemm", build / "gtsim_fa3"]
    for directory in ("src", "include", "workloads", "scripts/hbf", "cmake"):
        files.extend(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and path.suffix in (".cpp", ".h", ".hpp", ".py", ".cmake"))
    (output / "provenance.json").write_text(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat(),
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"), "status": git("status", "--short"),
        "hbfine_head": git("rev-parse", "HEAD", cwd=ROOT / "3rdparty/hbfine"),
        "hbfine_status": git("status", "--short", cwd=ROOT / "3rdparty/hbfine"),
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "max_concurrent_simulators": 2, "core_frequency_mhz": 1980}, indent=2) + "\n")

    def family(name):
        # One child per family: GEMM grows while FA3 shrinks, limiting peak memory.
        return [run_case(build, output, name, shape, profile)
                for shape in CASES[name] for profile in PROFILES]

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = [row for group in pool.map(family, CASES) for row in group]
    for row in rows:
        same = [r for r in rows if r["case"] == row["case"]]
        reference = next(r for r in same if r["configuration"] == "0F5M_channels")
        naive = next(r for r in same if r["configuration"] == row["configuration"][:4] + "_naive") if row["configuration"] != "0F5M_channels" else reference
        row["cycles_over_0F5M"] = round(row["cycles"] / reference["cycles"], 6)
        row["wall_over_matched_naive"] = round(row["wall_seconds"] / naive["wall_seconds"], 6)
    write_csv(output / "comparison.csv", rows)
    write_csv(output / "cycles.csv", [{"case": case, **{r["configuration"]: r["cycles"] for r in rows if r["case"] == case}}
                                      for case in dict.fromkeys(r["case"] for r in rows)])
    resources = []
    for path in sorted(output.glob("*.json")):
        data = json.loads(path.read_text())
        stats = data.get("hbf_statistics", {})
        if not stats:
            continue
        for key in ("stack_busy_ps", "channel_read_busy_ps", "channel_write_busy_ps", "bank_busy_ps",
                    "plane_sense_busy_ps", "plane_program_busy_ps", "plane_erase_busy_ps", "die_transfer_busy_ps", "ecc_busy_ps"):
            for index, busy in enumerate(stats[key]):
                # Stack busy is summed Bank occupancy: normalize by Bank count.
                engines = len(stats["bank_busy_ps"]) // len(stats["stack_busy_ps"]) if key == "stack_busy_ps" else 1
                resources.append({"case": data["row"]["case"], "configuration": data["row"]["configuration"],
                    "resource": key, "index": index, "busy_ps": busy,
                    "utilization": busy / stats["elapsed_ps"] / engines})
    write_csv(output / "resources.csv", resources)


if __name__ == "__main__":
    main()
