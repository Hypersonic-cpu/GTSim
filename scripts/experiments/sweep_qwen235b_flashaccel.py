#!/usr/bin/env python3
"""Run the Qwen3-235B FlashAccel experiment as one resumable queue.

The primary sweep is deliberately generated here rather than through per-case
JSON inputs.  Context length is passed to the compiler runner as an override,
and all dependent control/replay jobs share the same scheduler and resource
guard as the primary cases.
"""

import argparse
import asyncio
import csv
from dataclasses import dataclass, field
from itertools import product
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "frontend/asm/qwen3/flash-accel/one_block.progir"
PARAMS = REPOSITORY / "frontend/asm/qwen3/qwen3-235b-a22b-b1.json"
ARCHITECTURE = REPOSITORY / "configs/v0.1.0/h100-sxm.yaml"
FULL_HBM_MEMORY = REPOSITORY / "configs/memconf/0F5M_channels.yaml"
FLASHACCEL_MEMORY = REPOSITORY / "configs/memconf/4F1M_flashaccel_32m.yaml"
DEFAULT_OUTPUT = REPOSITORY / "validation/results/qwen235b-flashaccel-sweep"

CONTEXT_LENGTHS = (10_000, 100_000, 200_000)
BM_VALUES = (16, 32)
BN_VALUES = (64, 128, 256)
BK_VALUES = (16, 32, 64, 128, 256)
STAGES = (2,)
SPLIT_K = (1,)
TILINGS = tuple(product(BM_VALUES, BN_VALUES, BK_VALUES, STAGES))
MAIN_CONFIGS = (
    ("fullhbm", "none", "off"),
    ("4f1m", "pool", "on"),
    ("4f1m", "double", "on"),
)
MAX_CONCURRENCY = 150
PER_CASE_TIMEOUT_SECONDS = 1800
AGENT_PROGRESS_INTERVAL_SECONDS = 1800
ASSUMED_CASE_BYTES = 10 * 1024**3
MEMORY_SAFETY_FRACTION = 0.75
MEMORY_GUARD_FRACTION = 0.90
STATISTICS_INTERVAL_PS = 1_000_000


@dataclass
class Case:
    kind: str
    context: int
    hardware: str
    sram_policy: str
    prefetch: str
    tiling: tuple[int, int, int, int]
    output: Path
    fingerprint: str = ""
    bandwidth: bool = False
    attempt: int = 0

    @property
    def bm(self):
        return self.tiling[0]

    @property
    def bn(self):
        return self.tiling[1]

    @property
    def bk(self):
        return self.tiling[2]

    @property
    def stages(self):
        return self.tiling[3]

    @property
    def group(self):
        return (self.context, self.hardware, self.sram_policy)

    @property
    def label(self):
        return (
            f"{self.kind}-ctx{self.context}-" 
            f"{self.hardware}-{self.sram_policy}-"
            f"bm{self.bm}-bn{self.bn}-bk{self.bk}-s{self.stages}"
        )


@dataclass
class JobResult:
    case: Case
    status: str
    returncode: int | None
    elapsed_seconds: float
    error: str = ""
    payload: dict = field(default_factory=dict)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def available_memory_bytes():
    """Return container-visible available memory, never host-total memory."""

    available = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    except (FileNotFoundError, ValueError):
        pass
    limit_text = None
    try:
        limit_text = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    except FileNotFoundError:
        pass
    if limit_text and limit_text != "max":
        limit = _read_int("/sys/fs/cgroup/memory.max")
        current = _read_int("/sys/fs/cgroup/memory.current") or 0
        headroom = max(0, limit - current) if limit is not None else None
        if headroom is not None:
            available = headroom if available is None else min(available, headroom)
    return available or ASSUMED_CASE_BYTES


def cpu_capacity():
    try:
        capacity = len(os.sched_getaffinity(0))
    except AttributeError:
        capacity = os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            capacity = min(capacity, max(1, math.floor(int(quota) / int(period))))
    except (FileNotFoundError, ValueError):
        pass
    return max(1, capacity)


def effective_concurrency(requested):
    available = available_memory_bytes()
    memory_slots = max(1, math.floor(available * MEMORY_SAFETY_FRACTION / ASSUMED_CASE_BYTES))
    return min(requested, MAX_CONCURRENCY, cpu_capacity(), memory_slots)


def _rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ValueError):
        return 0
    return 0


def _children(pid):
    try:
        text = Path(f"/proc/{pid}/task/{pid}/children").read_text()
        return [int(value) for value in text.split()]
    except (FileNotFoundError, ValueError):
        return []


def process_tree_rss(pid):
    seen, todo, total = set(), [pid], 0
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        total += _rss(current)
        todo.extend(_children(current))
    return total


def _context_name(context):
    return f"ctx{context // 1000}k"


def _tiling_name(tiling):
    bm, bn, bk, stages = tiling
    return f"bm{bm}_bn{bn}_bk{bk}_s{stages}"


def _memory_for(hardware):
    return FULL_HBM_MEMORY if hardware == "fullhbm" else FLASHACCEL_MEMORY


def _bandwidth_memory(base, target):
    text = Path(base).read_text()
    if "  statistics_interval_ps:" not in text:
        text = text.replace(
            "hbm:\n", f"hbm:\n  statistics_interval_ps: {STATISTICS_INTERVAL_PS}\n", 1
        )
    if "  controller:\n" in text and "    statistics_interval_ps:" not in text:
        text = text.replace(
            "  controller:\n",
            f"  controller:\n    statistics_interval_ps: {STATISTICS_INTERVAL_PS}\n",
            1,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_text() != text:
        target.write_text(text)
    return target


def _case_path(root, case):
    tile = _tiling_name(case.tiling)
    if case.kind == "main":
        return root / _context_name(case.context) / case.hardware / case.sram_policy / tile
    if case.kind == "preflight":
        return root / "preflight" / _context_name(case.context) / case.hardware / case.sram_policy / tile
    return root / case.kind / _context_name(case.context) / case.hardware / case.sram_policy / tile


def _case_fingerprint(case, commit, runner_sha, memory_sha, params):
    identity = {
        "kind": case.kind,
        "git_commit": commit,
        "runner_sha256": runner_sha,
        "source_sha256": _sha(SOURCE),
        "architecture_sha256": _sha(ARCHITECTURE),
        "memory_sha256": memory_sha,
        "context_length": case.context,
        "hardware": case.hardware,
        "sram_policy": case.sram_policy,
        "prefetch": case.prefetch,
        "tiling": case.tiling,
        "bandwidth": case.bandwidth,
        "params": params,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def make_case(kind, context, hardware, sram_policy, prefetch, tiling, root, bandwidth=False):
    case = Case(
        kind,
        context,
        hardware,
        sram_policy,
        prefetch,
        tuple(tiling),
        Path("."),
        bandwidth=bandwidth,
    )
    case.output = _case_path(root, case)
    return case


def command_for(case, runner, root, compile_only=False):
    memory = _memory_for(case.hardware)
    if case.bandwidth:
        memory = _bandwidth_memory(
            memory, root / "inputs" / "bandwidth" / memory.name
        )
    command = [
        sys.executable,
        "frontend/tools/run_qwen_simulations.py",
        "--params",
        str(PARAMS),
        "--source",
        str(SOURCE),
        "--lib",
        str(REPOSITORY / "frontend/lib"),
        "--runner",
        str(runner),
        "--architecture",
        str(ARCHITECTURE),
        "--memory",
        str(memory),
        "--context-length",
        str(case.context),
        "--gemm-bm",
        str(case.bm),
        "--gemm-bn",
        str(case.bn),
        "--gemm-bk",
        str(case.bk),
        "--gemm-stages",
        str(case.stages),
        "--sram-policy",
        case.sram_policy,
        "--prefetch",
        case.prefetch,
        "--max-cycles",
        "1000000000",
        "--out",
        str(case.output),
    ]
    if compile_only:
        command.append("--compile-only")
    return command


def read_case_status(case):
    path = case.output / "case_status.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return value if value.get("fingerprint") == case.fingerprint else None


def _tail(path, lines=20):
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except FileNotFoundError:
        return ""


class Scheduler:
    def __init__(self, root, runner, requested_concurrency, retry_failed=False):
        self.root = root
        self.runner = runner
        self.retry_failed = retry_failed
        self.allowed_running = effective_concurrency(requested_concurrency)
        self.initial_available = available_memory_bytes()
        self.active = {}
        self.active_lock = asyncio.Lock()
        self.monitor_task = None
        self.records = {}
        self.scheduled_groups = set()

    async def start(self):
        self.monitor_task = asyncio.create_task(self._memory_monitor())

    async def stop(self):
        if self.monitor_task is not None:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

    def can_skip(self, case):
        previous = read_case_status(case)
        if previous is None:
            return False
        if previous.get("status") in ("SUCCESS", "INFEASIBLE"):
            return True
        return not self.retry_failed

    async def _memory_monitor(self):
        while True:
            await asyncio.sleep(5)
            available = available_memory_bytes()
            limit = available * MEMORY_GUARD_FRACTION
            async with self.active_lock:
                active = sorted(
                    self.active.values(), key=lambda item: item["started"], reverse=True
                )
            total = sum(process_tree_rss(item["process"].pid) for item in active)
            while active and total > limit:
                victim = active.pop(0)
                if victim["ram_guard"]:
                    continue
                victim["ram_guard"] = True
                await self._terminate(victim["process"])
                total = sum(process_tree_rss(item["process"].pid) for item in active)
                self.allowed_running = max(1, self.allowed_running - 1)

    async def _terminate(self, process):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), 10)
            return
        except asyncio.TimeoutError:
            pass
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()

    async def execute(self, case, compile_only=False):
        case.output.mkdir(parents=True, exist_ok=True)
        log_path = case.output / "driver.log"
        started = time.monotonic()
        log = log_path.open("w")
        process = await asyncio.create_subprocess_exec(
            *command_for(case, self.runner, self.root, compile_only),
            cwd=REPOSITORY,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        entry = {"process": process, "started": started, "ram_guard": False}
        async with self.active_lock:
            self.active[case.label] = entry
        status, error = "FAILED", ""
        returncode = None
        try:
            try:
                returncode = await asyncio.wait_for(
                    process.wait(), PER_CASE_TIMEOUT_SECONDS
                )
                if entry["ram_guard"]:
                    status = "DEFERRED_RAM"
                    error = "terminated by the 90% available-memory guard"
                elif returncode == 0:
                    status = "SUCCESS"
                else:
                    status = "INFEASIBLE" if compile_only else "FAILED"
                    error = _tail(log_path)
            except asyncio.TimeoutError:
                status, error = "TIMEOUT", "1800-second wall-clock timeout"
                await self._terminate(process)
            except asyncio.CancelledError:
                await self._terminate(process)
                raise
        finally:
            async with self.active_lock:
                self.active.pop(case.label, None)
            log.close()
        elapsed = time.monotonic() - started
        payload = {}
        result_path = case.output / ("compile_result.json" if compile_only else "result.json")
        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text())
            except json.JSONDecodeError:
                payload = {}
        if status == "SUCCESS" and not compile_only:
            timing = payload.get("timing", {})
            if not isinstance(timing.get("total_cycles"), int):
                status = "FAILED"
                error = "successful process had no total_cycles in result.json"
        record = {
            "fingerprint": case.fingerprint,
            "case_id": case.label,
            "kind": case.kind,
            "status": status,
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "attempt": case.attempt,
            "context_length": case.context,
            "hardware": case.hardware,
            "sram_policy": case.sram_policy,
            "prefetch": case.prefetch,
            "bm": case.bm,
            "bn": case.bn,
            "bk": case.bk,
            "stages": case.stages,
            "error": error,
            "result": str(result_path) if result_path.exists() else "",
        }
        case.output.joinpath("case_status.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        return JobResult(case, status, returncode, elapsed, error, payload)

    async def run_queue(self, cases, compile_only, on_result):
        queue = asyncio.Queue()
        for case in cases:
            await queue.put(case)
        worker_count = max(1, min(self.allowed_running, len(cases) or 1))

        async def worker():
            while True:
                case = await queue.get()
                try:
                    while True:
                        async with self.active_lock:
                            occupied = len(self.active)
                        if occupied < self.allowed_running:
                            break
                        await asyncio.sleep(5)
                    result = await self.execute(case, compile_only)
                    if result.status == "DEFERRED_RAM" and case.attempt < 20:
                        case.attempt += 1
                        await asyncio.sleep(min(300, 30 * case.attempt))
                        await queue.put(case)
                    else:
                        self.records[case.label] = result
                        await on_result(result, queue)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
        await queue.join()
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


def _base_params():
    return json.loads(PARAMS.read_text())


def _make_main_cases(root):
    return [
        make_case(
            "main",
            context,
            hardware,
            policy,
            prefetch,
            tiling,
            root,
        )
        for context, (hardware, policy, prefetch), tiling in product(
            CONTEXT_LENGTHS, MAIN_CONFIGS, TILINGS
        )
    ]


def _record_dict(result):
    return result if isinstance(result, dict) else result.__dict__


def _result_payload(case):
    result_path = case.output / "result.json"
    if not result_path.exists():
        return {}
    try:
        return json.loads(result_path.read_text())
    except json.JSONDecodeError:
        return {}


def _metric_record(case, status=None):
    payload = _result_payload(case)
    timing = payload.get("timing", {})
    traffic = timing.get("traffic_bytes", {})
    prefetch = timing.get("prefetch_stats", {})
    metadata = payload.get("compile_metadata", {})
    partition = metadata.get("partition", {})
    cycles = timing.get("total_cycles")
    latency_ms = cycles / (1980 * 1_000_000) * 1000 if isinstance(cycles, int) else None
    return {
        "context_length": case.context,
        "hardware": case.hardware,
        "sram_policy": case.sram_policy,
        "prefetch": case.prefetch,
        "kind": case.kind,
        "bm": case.bm,
        "bn": case.bn,
        "bk": case.bk,
        "stages": case.stages,
        "tiling": list(case.tiling),
        "partitions": partition.get("generated_slabs"),
        "cycles": cycles,
        "latency_ms": latency_ms,
        "tokens_per_s": (1000 / latency_ms) if latency_ms else None,
        "hbm_read_bytes": traffic.get("HBM_read"),
        "hbm_write_bytes": traffic.get("HBM_write"),
        "hbf_read_bytes": traffic.get("HBF_read"),
        "hbf_write_bytes": traffic.get("HBF_write"),
        "prefetch_queued": prefetch.get("queued"),
        "prefetch_admitted": prefetch.get("admitted"),
        "prefetch_completed": prefetch.get("completed"),
        "sram_resident_read_bytes": prefetch.get("resident_read_bytes"),
        "fallback_read_bytes": prefetch.get("fallback_read_bytes"),
        "peak_rss_kib": payload.get("peak_rss_kib"),
        "wall_seconds": payload.get("elapsed_seconds"),
        "status": status or "SUCCESS",
    }


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sample_rows(case):
    payload = _result_payload(case)
    samples = payload.get("timing", {}).get("memory_bandwidth", {})
    rows = []
    for tier, values in samples.items():
        for sample in values:
            start, end = sample.get("start_ps", 0), sample.get("end_ps", 0)
            width = max(1, end - start)
            bytes_by_name = sample.get("bytes", {})
            counts = sample.get("counts", {})
            row = {
                "context_length": case.context,
                "hardware": case.hardware,
                "sram_policy": case.sram_policy,
                "tier": tier,
                "start_ps": start,
                "end_ps": end,
                "elapsed_us": (start + end) / 2 / 1_000_000,
                "hbm_read_gbps": 0.0,
                "hbm_write_gbps": 0.0,
                "hbm_total_gbps": 0.0,
                "hbf_demand_read_gbps": 0.0,
                "hbf_prefetch_fill_gbps": 0.0,
                "hbf_media_read_gbps": 0.0,
                "hbf_resident_read_gbps": 0.0,
                "sense_commands": counts.get("sense_commands", 0),
            }
            for name, value in bytes_by_name.items():
                rate = value * 1000.0 / width
                if tier == "HBM" and name == "demand_read":
                    row["hbm_read_gbps"] = rate
                elif tier == "HBM" and name == "demand_write":
                    row["hbm_write_gbps"] = rate
                elif tier == "HBF" and name == "demand_read":
                    row["hbf_demand_read_gbps"] = rate
                elif tier == "HBF" and name == "prefetch_fill":
                    row["hbf_prefetch_fill_gbps"] = rate
                elif tier == "HBF" and name == "media_read":
                    row["hbf_media_read_gbps"] = rate
                elif tier == "HBF" and name == "resident_read":
                    row["hbf_resident_read_gbps"] = rate
            row["hbm_total_gbps"] = row["hbm_read_gbps"] + row["hbm_write_gbps"]
            rows.append(row)
    return rows


def _make_figures(root, bandwidth_cases):
    rows = [row for case in bandwidth_cases for row in _sample_rows(case)]
    full_rows = [row for row in rows if row["hardware"] == "fullhbm"]
    flash_rows = [row for row in rows if row["hardware"] == "4f1m"]
    _write_csv(root / "figures/bw_fullhbm.csv", full_rows)
    _write_csv(root / "figures/bw_4f1m_pool.csv", [r for r in flash_rows if r["sram_policy"] == "pool"])
    _write_csv(root / "figures/bw_4f1m_double.csv", [r for r in flash_rows if r["sram_policy"] == "double"])
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib unavailable; CSV bandwidth samples were written"

    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    contexts = CONTEXT_LENGTHS
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for axis, context in zip(axes, contexts):
        subset = [row for row in full_rows if row["context_length"] == context]
        x = [row["elapsed_us"] for row in subset]
        for key, label, color in (
            ("hbm_read_gbps", "HBM read", "tab:blue"),
            ("hbm_write_gbps", "HBM write", "tab:orange"),
            ("hbm_total_gbps", "HBM total", "tab:green"),
        ):
            axis.plot(x, [row[key] for row in subset], label=label, color=color)
        axis.axhline(3100, color="black", linestyle="--", linewidth=0.8, label="3100 GB/s")
        axis.set_title(_context_name(context))
        axis.set_xlabel("simulated elapsed time (µs)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("bandwidth (GB/s)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "bw_fullhbm.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for axis, context in zip(axes, contexts):
        for policy, linestyle in (("pool", "-"), ("double", "--")):
            subset = [
                row
                for row in flash_rows
                if row["context_length"] == context and row["sram_policy"] == policy
            ]
            x = [row["elapsed_us"] for row in subset]
            for key, label, color in (
                ("hbm_read_gbps", f"HBM read ({policy})", "tab:blue"),
                ("hbf_demand_read_gbps", f"HBF demand ({policy})", "tab:red"),
                ("hbf_prefetch_fill_gbps", f"HBF prefetch ({policy})", "tab:orange"),
                ("hbf_media_read_gbps", f"HBF sense ({policy})", "tab:purple"),
                ("hbf_resident_read_gbps", f"SRAM resident ({policy})", "tab:green"),
            ):
                axis.plot(x, [row[key] for row in subset], linestyle=linestyle, color=color, label=label)
        axis.axhline(620, color="tab:blue", linestyle=":", linewidth=0.8, label="620 GB/s HBM")
        axis.axhline(3072, color="tab:red", linestyle=":", linewidth=0.8, label="3072 GB/s HBF")
        axis.set_title(_context_name(context))
        axis.set_xlabel("simulated elapsed time (µs)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("bandwidth (GB/s)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "bw_4f1m.png", dpi=150)
    plt.close(fig)
    return "figures and CSV bandwidth samples written"


def _write_report(root, main_cases, controls, bandwidth_cases, statuses, best):
    rows = []
    for case in main_cases:
        status = statuses.get(case.label, {}).get("status", "NOT_RUN")
        if status == "SUCCESS":
            rows.append(_metric_record(case, status))
    _write_csv(root / "summary.csv", rows)
    manifest_rows = []
    for status in statuses.values():
        manifest_rows.append(dict(status))
    _write_csv(root / "manifest.csv", manifest_rows)
    root.joinpath("best_configs.json").write_text(json.dumps(best, indent=2) + "\n")

    lines = [
        "# Qwen3-235B-A22B FlashAccel sweep",
        "",
        "Measured one-block B=1 results. Missing values are left unfilled.",
        "",
        "| Context | Configuration | Best BM/BN/BK | Partitions | Cycles | Latency (ms) | Speedup vs HBM |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for context in CONTEXT_LENGTHS:
        hbm = best.get(f"{context}:fullhbm:none", {})
        hbm_cycles = hbm.get("cycles")
        for hardware, policy, title in (
            ("fullhbm", "none", "FullHBM"),
            ("4f1m", "pool", "4F1M Pool"),
            ("4f1m", "double", "4F1M Double"),
        ):
            winner = best.get(f"{context}:{hardware}:{policy}", {})
            cycles = winner.get("cycles")
            speedup = hbm_cycles / cycles if hbm_cycles and cycles else None
            tile = winner.get("tiling")
            tile_text = "/".join(map(str, tile[:3])) if tile else ""
            lines.append(
                f"| {_context_name(context)} | {title} | {tile_text} | "
                f"{winner.get('partitions', '')} | {cycles or ''} | "
                f"{winner.get('latency_ms', '')} | "
                f"{f'{speedup:.3f}×' if speedup else ''} |"
            )
    lines.extend(["", "## Prefetch-OFF controls", ""])
    for case in controls:
        status = statuses.get(case.label, {}).get("status", "NOT_RUN")
        metric = _metric_record(case, status) if status == "SUCCESS" else {}
        lines.append(
            f"- {_context_name(case.context)} {case.sram_policy}: status={status}, "
            f"cycles={metric.get('cycles', '')}, tiling=BM{case.bm}/BN{case.bn}/BK{case.bk}"
        )
    lines.extend(["", "## Measured-data questions", ""])
    lines.append("- The best tiling by context is reported in the table; no unmeasured trend is inferred.")
    lines.append("- BK and resource effects are reported only where successful candidate data exist.")
    lines.append("- N-slab counts come from the compiler's generated partition metadata.")
    lines.append("- Prefetch overlap is compared only between matching winner/control tilings.")
    lines.append("- Bandwidth figures use sampled runtime bins; HBM/HBF domains are kept separate.")
    lines.append("- Results are measured single-block simulations, not full-model deployment estimates.")
    successful = sum(1 for value in statuses.values() if value.get("status") == "SUCCESS")
    infeasible = sum(1 for value in statuses.values() if value.get("status") == "INFEASIBLE")
    timed_out = sum(1 for value in statuses.values() if value.get("status") == "TIMEOUT")
    failed = sum(1 for value in statuses.values() if value.get("status") == "FAILED")
    lines.append("")
    lines.append(
        f"Run status counts: success={successful}, infeasible={infeasible}, "
        f"timeout={timed_out}, failed={failed}."
    )
    root.joinpath("report.md").write_text("\n".join(lines) + "\n")


async def main_async(args):
    os.chdir(REPOSITORY)
    frontend_path = str(REPOSITORY / "frontend/src")
    os.environ["PYTHONPATH"] = (
        frontend_path
        if not os.environ.get("PYTHONPATH")
        else frontend_path + os.pathsep + os.environ["PYTHONPATH"]
    )
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    runner = Path(args.runner).resolve()
    if not runner.exists():
        raise SystemExit(f"runner binary does not exist: {runner}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    runner_sha = _sha(runner)
    params = _base_params()
    scheduler = Scheduler(root, runner, args.max_concurrency, args.retry_failed)
    for context in CONTEXT_LENGTHS:
        params["CONTEXTS"] = [context]
        break
    base_params = _base_params()

    def write_progress(statuses, phase):
        counts = {}
        for value in statuses.values():
            status = value.get("status", "UNKNOWN")
            counts[status] = counts.get(status, 0) + 1
        root.joinpath("progress.json").write_text(
            json.dumps(
                {
                    "phase": phase,
                    "updated_unix_seconds": time.time(),
                    "counts": counts,
                    "available_memory_bytes": available_memory_bytes(),
                    "active_jobs": len(scheduler.active),
                    "allowed_running": scheduler.allowed_running,
                },
                indent=2,
            )
            + "\n"
        )

    def prepare_cases(cases):
        for case in cases:
            params_for_case = dict(base_params)
            params_for_case["CONTEXTS"] = [case.context]
            case.fingerprint = _case_fingerprint(
                case,
                commit,
                runner_sha,
                _sha(_memory_for(case.hardware)),
                params_for_case,
            )
        return cases

    main_cases = prepare_cases(_make_main_cases(root))
    pilot = next(
        case
        for case in main_cases
        if case.context == 10_000
        and case.hardware == ("fullhbm" if args.pilot_config == "fullhbm" else "4f1m")
        and case.sram_policy == ("none" if args.pilot_config == "fullhbm" else "double")
        and case.tiling == (16, 64, 16, 2)
    )
    await scheduler.start()
    try:
        if not scheduler.can_skip(pilot):
            pilot_result = await scheduler.execute(pilot)
            if pilot_result.status != "SUCCESS":
                raise SystemExit(
                    "T0 pilot failed; inspect " + str(pilot.output / "driver.log")
                )
            scheduler.records[pilot.label] = pilot_result

        preflight_cases = []
        for case in main_cases:
            preflight = make_case(
                "preflight",
                case.context,
                case.hardware,
                case.sram_policy,
                case.prefetch,
                case.tiling,
                root,
            )
            preflight.fingerprint = _case_fingerprint(
                preflight,
                commit,
                runner_sha,
                _sha(_memory_for(case.hardware)),
                dict(base_params, CONTEXTS=[case.context]),
            )
            if not scheduler.can_skip(preflight):
                preflight_cases.append(preflight)

        async def preflight_done(result, _queue):
            scheduler.records[result.case.label] = result
            write_progress({}, "preflight")

        await scheduler.run_queue(preflight_cases, True, preflight_done)

        legal = []
        statuses = {}
        for case in main_cases:
            preflight = make_case(
                "preflight",
                case.context,
                case.hardware,
                case.sram_policy,
                case.prefetch,
                case.tiling,
                root,
            )
            preflight.fingerprint = _case_fingerprint(
                preflight,
                commit,
                runner_sha,
                _sha(_memory_for(case.hardware)),
                dict(base_params, CONTEXTS=[case.context]),
            )
            previous = read_case_status(preflight)
            if previous and previous.get("status") == "SUCCESS":
                legal.append(case)
            else:
                status = previous or {"status": "INFEASIBLE", "error": "no successful preflight"}
                statuses[case.label] = dict(status, case_id=case.label, kind="main")

        group_expected = {}
        group_done = {}
        group_results = {}
        for case in legal:
            group_expected[case.group] = group_expected.get(case.group, 0) + 1
            group_done.setdefault(case.group, 0)
            group_results.setdefault(case.group, [])

        dependent_cases = []
        best = {}
        controls = []
        bandwidth_cases = []
        all_jobs = []

        def winner_for(group):
            results = group_results[group]
            successes = [item for item in results if item.get("status") == "SUCCESS"]
            if not successes:
                return None
            return min(successes, key=lambda item: item.get("cycles", 10**30))

        async def enqueue_dependents(group, queue):
            if group in scheduler.scheduled_groups:
                return
            scheduler.scheduled_groups.add(group)
            winner = winner_for(group)
            if winner is None:
                return
            context, hardware, policy = group
            tile = tuple(winner["tiling"])
            prefetch = "off" if hardware == "fullhbm" else "on"
            replay = make_case("bandwidth", context, hardware, policy, prefetch, tile, root, True)
            replay.fingerprint = _case_fingerprint(
                replay, commit, runner_sha, _sha(_memory_for(hardware)), dict(base_params, CONTEXTS=[context])
            )
            bandwidth_cases.append(replay)
            previous = read_case_status(replay)
            if previous and scheduler.can_skip(replay):
                statuses[replay.label] = dict(previous)
            else:
                await queue.put(replay)
            if hardware == "4f1m":
                control = make_case("controls", context, hardware, policy, "off", tile, root)
                control.fingerprint = _case_fingerprint(
                    control, commit, runner_sha, _sha(_memory_for(hardware)), dict(base_params, CONTEXTS=[context])
                )
                controls.append(control)
                previous = read_case_status(control)
                if previous and scheduler.can_skip(control):
                    statuses[control.label] = dict(previous)
                else:
                    await queue.put(control)

        async def main_done(result, queue):
            case = result.case
            metric = _metric_record(case, result.status) if result.status == "SUCCESS" else {}
            statuses[case.label] = {
                "case_id": case.label,
                "kind": case.kind,
                "status": result.status,
                "returncode": result.returncode,
                "error": result.error,
                **metric,
            }
            write_progress(statuses, "main_and_dependents")
            group_done[case.group] += 1
            group_results[case.group].append(statuses[case.label])
            if group_done[case.group] == group_expected[case.group]:
                winner = winner_for(case.group)
                if winner:
                    context, hardware, policy = case.group
                    best[f"{context}:{hardware}:{policy}"] = winner
                await enqueue_dependents(case.group, queue)

        # Seed completed main cases for resume, then enqueue any dependents
        # whose winner group was already complete.
        pending_main = []
        for case in legal:
            previous = read_case_status(case)
            if previous and previous.get("status") == "SUCCESS" and not args.retry_failed:
                metric = _metric_record(case, previous.get("status"))
                statuses[case.label] = dict(previous, **metric)
                group_done[case.group] += 1
                group_results[case.group].append(statuses[case.label])
            elif not scheduler.can_skip(case):
                pending_main.append(case)
            else:
                previous = previous or {"status": "FAILED"}
                statuses[case.label] = dict(previous, case_id=case.label, kind="main")
                group_done[case.group] += 1
                group_results[case.group].append(statuses[case.label])
        for group, expected in group_expected.items():
            if group_done[group] == expected:
                winner = winner_for(group)
                if winner:
                    context, hardware, policy = group
                    best[f"{context}:{hardware}:{policy}"] = winner

        async def existing_dependents(queue):
            for group, expected in group_expected.items():
                if group_done[group] == expected:
                    await enqueue_dependents(group, queue)

        async def run_main_and_dependents():
            queue = asyncio.Queue()
            for case in pending_main:
                await queue.put(case)
            await existing_dependents(queue)
            # Use the scheduler's queue machinery for the main jobs, but
            # allow completion callbacks to append dependent jobs to it.
            worker_count = max(1, min(scheduler.allowed_running, max(1, len(pending_main) + len(dependent_cases))))

            async def worker():
                while True:
                    case = await queue.get()
                    try:
                        while len(scheduler.active) >= scheduler.allowed_running:
                            await asyncio.sleep(5)
                        result = await scheduler.execute(case)
                        if result.status == "DEFERRED_RAM" and case.attempt < 20:
                            case.attempt += 1
                            await asyncio.sleep(min(300, 30 * case.attempt))
                            await queue.put(case)
                        else:
                            scheduler.records[case.label] = result
                            await main_done(result, queue)
                    finally:
                        queue.task_done()

            workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
            await queue.join()
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        await run_main_and_dependents()

        for case in bandwidth_cases + controls:
            previous = read_case_status(case)
            if previous:
                statuses[case.label] = dict(previous)
        for group, winner in best.items():
            if winner.get("status") == "SUCCESS":
                winner_case = next(
                    (case for case in legal if case.label == winner.get("case_id")), None
                )
                if winner_case:
                    best[group] = _metric_record(winner_case)
                    best[group]["tiling"] = list(winner_case.tiling)
        _write_report(root, main_cases, controls, bandwidth_cases, statuses, best)
        figure_message = _make_figures(root, [case for case in bandwidth_cases if read_case_status(case)])
        root.joinpath("figure_status.txt").write_text(figure_message + "\n")
        root.joinpath("scheduler_summary.json").write_text(
            json.dumps(
                {
                    "initial_available_memory_bytes": scheduler.initial_available,
                    "effective_concurrency": effective_concurrency(args.max_concurrency),
                    "assumed_case_bytes": ASSUMED_CASE_BYTES,
                    "memory_safety_fraction": MEMORY_SAFETY_FRACTION,
                    "memory_guard_fraction": MEMORY_GUARD_FRACTION,
                    "polling_note": "agent inspection interval is at least 1800 seconds; safety monitor is internal",
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        await scheduler.stop()


def main():
    global FLASHACCEL_MEMORY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=REPOSITORY / "build/gtsim_program")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--flashaccel-memory", type=Path, default=FLASHACCEL_MEMORY)
    parser.add_argument("--pilot-config", choices=("double", "fullhbm"), default="double")
    args = parser.parse_args()
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    FLASHACCEL_MEMORY = args.flashaccel_memory.resolve()
    if not FLASHACCEL_MEMORY.is_file():
        parser.error(f"memory configuration does not exist: {FLASHACCEL_MEMORY}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
