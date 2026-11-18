#!/usr/bin/env python3
"""Run the Qwen3-235B FlashAccel experiment as one resumable queue.

The primary sweep is deliberately generated here rather than through per-case
JSON inputs.  Context length is passed to the compiler runner as an override,
and all dependent control/replay jobs share the same scheduler and resource
guard as the primary cases.
"""

import argparse
import asyncio
from ast import literal_eval
import csv
from dataclasses import dataclass, field, replace
from itertools import product
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import textwrap
import time

if __package__:
    from .common.sweep_scheduler import memory_constraints, process_tree_memory
else:
    from common.sweep_scheduler import memory_constraints, process_tree_memory


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
STATUS_INTERVAL_SECONDS = 300
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

    def status_fields(self):
        return dict(context_length=self.context, hardware=self.hardware,
                    sram_policy=self.sram_policy, prefetch=self.prefetch,
                    bm=self.bm, bn=self.bn, bk=self.bk, stages=self.stages)


@dataclass
class ProgramCase:
    kind: str
    label: str
    output: Path
    fingerprint: str
    parameters: dict
    invocation: tuple[str, ...]
    attempt: int = 0

    def status_fields(self):
        return {"parameters": self.parameters}


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


def available_memory_bytes():
    """Return container-visible available memory, never host-total memory."""
    return min(max(0, item.capacity - item.used) for item in memory_constraints())


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


def effective_concurrency(requested, case_bytes=ASSUMED_CASE_BYTES,
                          safety_fraction=MEMORY_SAFETY_FRACTION,
                          ram_admission=True):
    capacity = min(requested, MAX_CONCURRENCY, cpu_capacity())
    if not ram_admission:
        return capacity
    available = available_memory_bytes()
    memory_slots = max(1, math.floor(available * safety_fraction / case_bytes))
    return min(capacity, memory_slots)


def process_tree_rss(pid):
    return process_tree_memory(pid)[0]


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
    if isinstance(case, ProgramCase):
        return [*case.invocation, "--out", str(case.output),
                *(("--compile-only",) if compile_only else ())]
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
    def __init__(self, root, runner, requested_concurrency, retry_failed=False,
                 ram_guard=True, case_bytes=ASSUMED_CASE_BYTES,
                 safety_fraction=MEMORY_SAFETY_FRACTION, ram_admission=True):
        self.root = root
        self.runner = runner
        self.retry_failed = retry_failed
        self.ram_guard = ram_guard
        self.allowed_running = effective_concurrency(
            requested_concurrency, case_bytes, safety_fraction, ram_admission)
        self.initial_available = available_memory_bytes()
        self.active = {}
        self.active_lock = asyncio.Lock()
        self.monitor_task = None
        self.records = {}
        self.scheduled_groups = set()
        self.status_task = None
        self.phase = "initializing"
        self.job_states = {}
        self.root.mkdir(parents=True, exist_ok=True)

    def log(self, event, **fields):
        record = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "event": event, "phase": self.phase, **fields}
        with (self.root / "scheduler_events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        # The event file retains every transition. The console reports routine
        # work in periodic summaries, so fast compiler checks do not flood it.
        if event in ("JOB_STARTED", "JOB_RESUMED", "DEPENDENT_JOB_ENQUEUED"):
            return
        if event == "JOB_FINISHED" and fields.get("status") in ("SUCCESS", "INFEASIBLE"):
            return
        stamp = time.strftime("%H:%M:%S", time.gmtime())
        if event == "STATUS":
            lines = [
                f"[{stamp} UTC] {self.phase}",
                f"run={fields['running']}  wait={fields['waiting']}  "
                f"ok={fields['completed_success']}  fail={fields['completed_failed']}  "
                f"infeasible={fields['infeasible']}",
                f"limit={fields['concurrency_limit']}  "
                f"free RAM={fields['available_ram_gib']:.1f} GiB",
            ]
        else:
            lines = [f"[{stamp} UTC] {event.lower().replace('_', ' ')}"]
            for key, value in fields.items():
                if key == "error":
                    # Tracebacks remain in scheduler_events.jsonl and job logs.
                    value = str(value).strip().splitlines()[-1:] or [""]
                    value = value[0]
                lines.append(f"{key.replace('_', ' ')}: {value}")
        width = max(32, min(88, shutil.get_terminal_size((80, 24)).columns))
        for line in lines:
            print(textwrap.fill(line, width=width, subsequent_indent="  "), flush=True)

    def begin_phase(self, phase, cases=(), states=None):
        self.phase = phase
        self.job_states = {case.label: "WAITING" for case in cases}
        self.job_states.update(states or {})
        self.report_status()

    def report_status(self):
        counts = {}
        for state in self.job_states.values():
            counts[state] = counts.get(state, 0) + 1
        record = {
            "running": len(self.active),
            "waiting": counts.get("WAITING", 0) + counts.get("DEFERRED_RAM", 0),
            "completed_success": counts.get("SUCCESS", 0),
            "completed_failed": sum(counts.get(s, 0) for s in ("FAILED", "TIMEOUT", "OOM")),
            "infeasible": counts.get("INFEASIBLE", 0),
            "known_jobs": len(self.job_states),
            "concurrency_limit": self.allowed_running,
            "available_ram_gib": round(available_memory_bytes() / 1024**3, 2),
        }
        self.log("STATUS", **record)
        (self.root / "scheduler_status.json").write_text(
            json.dumps({"phase": self.phase, **record}, indent=2) + "\n")

    async def _status_monitor(self):
        while True:
            await asyncio.sleep(STATUS_INTERVAL_SECONDS)
            self.report_status()

    async def requeue(self, case, queue):
        case.attempt += 1
        delay = min(300, 30 * case.attempt)
        self.job_states[case.label] = "WAITING"
        self.log("RAM_RETRY_SCHEDULED", case=case.label, attempt=case.attempt,
                 delay_seconds=delay, concurrency_limit=self.allowed_running)
        await asyncio.sleep(delay)
        await queue.put(case)
        self.log("RAM_REQUEUED", case=case.label, attempt=case.attempt)

    async def start(self):
        self.status_task = asyncio.create_task(self._status_monitor())
        self.log("SCHEDULER_STARTED", concurrency_limit=self.allowed_running,
                 cpu_capacity=cpu_capacity(), ram_guard_enabled=self.ram_guard,
                 status_interval_seconds=STATUS_INTERVAL_SECONDS)
        if self.ram_guard:
            self.monitor_task = asyncio.create_task(self._memory_monitor())

    async def stop(self):
        for task in (self.monitor_task, self.status_task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.report_status()
        self.log("SCHEDULER_STOPPED")

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
                self.log("RAM_GUARD_TRIGGERED", case=victim["case"],
                         total_rss_gib=round(total / 1024**3, 3),
                         threshold_gib=round(limit / 1024**3, 3),
                         action="Terminate newest job and defer it for retry")
                self.allowed_running = max(1, self.allowed_running - 1)
                self.log("CONCURRENCY_REDUCED", concurrency_limit=self.allowed_running)
                await self._terminate(victim["process"])
                total = sum(process_tree_rss(item["process"].pid) for item in active)
                self.log("RAM_GUARD_RECHECK", remaining_rss_gib=round(total / 1024**3, 3),
                         threshold_gib=round(limit / 1024**3, 3))

    async def _terminate(self, process):
        self.log("PROCESS_TERMINATION_REQUESTED", pid=process.pid, signal="SIGTERM")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            self.log("PROCESS_ALREADY_EXITED", pid=process.pid)
            return
        try:
            await asyncio.wait_for(process.wait(), 10)
            self.log("PROCESS_TERMINATED", pid=process.pid, returncode=process.returncode)
            return
        except asyncio.TimeoutError:
            pass
        try:
            self.log("PROCESS_TERMINATION_ESCALATED", pid=process.pid, signal="SIGKILL")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
        self.log("PROCESS_TERMINATED", pid=process.pid, returncode=process.returncode)

    async def execute(self, case, compile_only=False):
        self.job_states[case.label] = "RUNNING"
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
        entry = {"process": process, "started": started, "ram_guard": False,
                 "case": case.label}
        async with self.active_lock:
            self.active[case.label] = entry
        self.log("JOB_STARTED", case=case.label, pid=process.pid,
                 attempt=case.attempt, compile_only=compile_only)
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
                    memory_error_text = error + "\n" + _tail(case.output / "simulation.log")
                    oom_reported = any(marker in memory_error_text.lower() for marker in (
                        "memoryerror", "std::bad_alloc", "cannot allocate memory", "out of memory"))
                    if oom_reported:
                        self.log("OUT_OF_MEMORY", case=case.label, pid=process.pid,
                                 returncode=returncode, error=error)
                        status = "DEFERRED_RAM"
                        self.allowed_running = max(1, self.allowed_running - 1)
                        self.log("CONCURRENCY_REDUCED", concurrency_limit=self.allowed_running)
                    elif returncode == -signal.SIGKILL:
                        self.log("JOB_KILLED", case=case.label, pid=process.pid,
                                 reason="SIGKILL; OOM is possible but unconfirmed")
            except asyncio.TimeoutError:
                status, error = "TIMEOUT", "1800-second wall-clock timeout"
                self.log("JOB_TIMEOUT", case=case.label, pid=process.pid)
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
        if status == "DEFERRED_RAM" and case.attempt >= 20:
            status = "OOM"
            self.log("RAM_RETRY_LIMIT_REACHED", case=case.label, attempts=case.attempt + 1)
        record = {
            "fingerprint": case.fingerprint,
            "case_id": case.label,
            "kind": case.kind,
            "status": status,
            "returncode": returncode,
            "elapsed_seconds": elapsed,
            "attempt": case.attempt,
            **case.status_fields(),
            "error": error,
            "result": str(result_path) if result_path.exists() else "",
        }
        case.output.joinpath("case_status.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        self.job_states[case.label] = status
        self.log("JOB_FINISHED", case=case.label, status=status,
                 returncode=returncode, elapsed_seconds=round(elapsed, 2),
                 error=error if status not in ("SUCCESS", "INFEASIBLE") else "")
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
                        await self.requeue(case, queue)
                    else:
                        self.records[case.label] = result
                        await on_result(result, queue)
                finally:
                    queue.task_done()

        async with asyncio.TaskGroup() as group:
            workers = [group.create_task(worker()) for _ in range(worker_count)]
            await queue.join()
            for task in workers:
                task.cancel()


def parameter_sets(specification):
    """A list names explicit cases, while a dict names Cartesian axes."""
    if isinstance(specification, dict):
        choices = [values if isinstance(values, (tuple, list)) else (values,)
                   for values in specification.values()]
        specification = [dict(zip(specification, values)) for values in product(*choices)]
    if not isinstance(specification, list) or not specification:
        raise ValueError("sweep requires a nonempty dict of axes or list of case dicts")
    for case in specification:
        if not isinstance(case, dict) or any(
            not isinstance(name, str) or not name.isidentifier() for name in case
        ):
            raise ValueError("each case must map parameter identifiers to Python literals")
    return specification


async def run_program_sweep(args, specification):
    root = args.output.resolve()
    runner, source, library, architecture, memory = (
        value.resolve() for value in
        (args.runner, args.program, args.lib, args.architecture, args.memory)
    )
    driver = REPOSITORY / "frontend/tools/run_qwen_simulations.py"
    invocation = (
        sys.executable, str(driver), "--source", str(source), "--lib", str(library),
        "--runner", str(runner), "--architecture", str(architecture), "--memory",
        str(memory), "--max-cycles", "1000000000",
    )
    inputs = [runner, source, driver, architecture, memory, *library.glob("*ir"),
              *(REPOSITORY / "frontend/src").rglob("*.py")]
    if args.params:
        inputs.append(args.params.resolve())
        invocation += ("--params", str(args.params.resolve()))
    input_hashes = {str(path): _sha(path) for path in sorted(inputs)}
    scheduler = Scheduler(
        root, runner, args.max_concurrency, args.retry_failed,
        ram_guard=args.ram_guard == "on", case_bytes=args.memory_per_case_gib * 1024**3,
        safety_fraction=args.memory_budget_fraction, ram_admission=args.ram_admission == "on",
    )
    cases = {}
    for parameters in parameter_sets(specification):
        # repr preserves list/tuple distinctions and accepts native Python literals.
        identity = repr((input_hashes, sorted(parameters.items())))
        fingerprint = hashlib.sha256(identity.encode()).hexdigest()
        label = fingerprint[:16]
        overrides = tuple(token for name, value in sorted(parameters.items())
                          for token in ("--param", f"{name}={value!r}"))
        cases[label] = ProgramCase("simulation", label, root / "cases" / label,
                                   fingerprint, parameters, invocation + overrides)
    checks = [replace(case, kind="preflight", output=root / "preflight" / case.label)
              for case in cases.values()]

    def phase(name, jobs):
        pending, saved = [], {}
        for case in jobs:
            previous = read_case_status(case)
            if previous is not None and scheduler.can_skip(case):
                result = case.output / ("compile_result.json" if name == "preflight" else "result.json")
                if previous["status"] != "SUCCESS" or result.is_file():
                    saved[case.label] = previous["status"]
                    continue
            pending.append(case)
        scheduler.begin_phase(name, pending, saved)
        return pending

    async def finished(result, queue):
        pass

    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(REPOSITORY / "frontend/src"), os.environ.get("PYTHONPATH"),
    )))
    await scheduler.start()
    try:
        scheduler.log("PILOT_SKIPPED", reason="Program sweep uses compiler preflight")
        await scheduler.run_queue(phase("preflight", checks), True, finished)
        legal = [cases[check.label] for check in checks
                 if (read_case_status(check) or {}).get("status") == "SUCCESS"]
        await scheduler.run_queue(phase("simulation", legal), False, finished)
    finally:
        await scheduler.stop()
    manifest = []
    for check in checks:
        case = cases[check.label]
        record = read_case_status(case) or read_case_status(check) or {}
        result = _result_payload(case) if record.get("status") == "SUCCESS" else {}
        manifest.append(dict(case=case.label, parameters=repr(case.parameters),
                             status=record.get("status", "WAITING"),
                             cycles=result.get("timing", {}).get("total_cycles"),
                             error=record.get("error", "")))
    _write_csv(root / "manifest.csv", manifest)


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
    scheduler = Scheduler(root, runner, args.max_concurrency, args.retry_failed,
                          ram_guard=args.ram_guard == "on",
                          case_bytes=args.memory_per_case_gib * 1024**3,
                          safety_fraction=args.memory_budget_fraction,
                          ram_admission=args.ram_admission == "on")
    scheduler.log("RESOURCE_POLICY", requested_concurrency=args.max_concurrency,
                  effective_concurrency=scheduler.allowed_running,
                  memory_per_case_gib=args.memory_per_case_gib,
                  memory_budget_fraction=args.memory_budget_fraction,
                  ram_admission=args.ram_admission, ram_guard=args.ram_guard)
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
        scheduler.begin_phase("pilot", () if args.skip_pilot else (pilot,))
        if args.skip_pilot:
            scheduler.log("PILOT_SKIPPED", reason="--skip-pilot was specified")
        elif scheduler.can_skip(pilot):
            previous = read_case_status(pilot)
            scheduler.job_states[pilot.label] = previous["status"]
            scheduler.log("PILOT_SKIPPED", case=pilot.label,
                          reason="A matching saved status exists", status=previous["status"])
        if not args.skip_pilot and not scheduler.can_skip(pilot):
            scheduler.log("PILOT_STARTED", case=pilot.label,
                          reason="Serial smoke check before candidate compilation and concurrent simulation")
            pilot_result = await scheduler.execute(pilot)
            scheduler.log("PILOT_FINISHED", case=pilot.label, status=pilot_result.status)
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

        scheduler.begin_phase("preflight", preflight_cases)
        await scheduler.run_queue(preflight_cases, True, preflight_done)
        scheduler.report_status()

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
                scheduler.job_states[replay.label] = previous["status"]
                scheduler.log("JOB_RESUMED", case=replay.label, status=previous["status"])
            else:
                scheduler.job_states[replay.label] = "WAITING"
                await queue.put(replay)
                scheduler.log("DEPENDENT_JOB_ENQUEUED", case=replay.label)
            if hardware == "4f1m":
                control = make_case("controls", context, hardware, policy, "off", tile, root)
                control.fingerprint = _case_fingerprint(
                    control, commit, runner_sha, _sha(_memory_for(hardware)), dict(base_params, CONTEXTS=[context])
                )
                controls.append(control)
                previous = read_case_status(control)
                if previous and scheduler.can_skip(control):
                    statuses[control.label] = dict(previous)
                    scheduler.job_states[control.label] = previous["status"]
                    scheduler.log("JOB_RESUMED", case=control.label, status=previous["status"])
                else:
                    scheduler.job_states[control.label] = "WAITING"
                    await queue.put(control)
                    scheduler.log("DEPENDENT_JOB_ENQUEUED", case=control.label)

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
            scheduler.begin_phase("main_and_dependents", pending_main,
                {name: value["status"] for name, value in statuses.items()})
            queue = asyncio.Queue()
            for case in pending_main:
                await queue.put(case)
            await existing_dependents(queue)
            # Use the scheduler's queue machinery for the main jobs, but
            # allow completion callbacks to append dependent jobs to it.
            worker_count = max(1, min(scheduler.allowed_running, max(1, queue.qsize())))

            async def worker():
                while True:
                    case = await queue.get()
                    try:
                        while len(scheduler.active) >= scheduler.allowed_running:
                            await asyncio.sleep(5)
                        result = await scheduler.execute(case)
                        if result.status == "DEFERRED_RAM" and case.attempt < 20:
                            await scheduler.requeue(case, queue)
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
        scheduler.report_status()

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
                    "effective_concurrency": scheduler.allowed_running,
                    "assumed_case_bytes": args.memory_per_case_gib * 1024**3,
                    "memory_safety_fraction": args.memory_budget_fraction,
                    "ram_admission_enabled": args.ram_admission == "on",
                    "pilot_skipped": args.skip_pilot,
                    "memory_guard_fraction": MEMORY_GUARD_FRACTION,
                    "ram_guard_enabled": scheduler.ram_guard,
                    "polling_note": "agent inspection interval is at least 1800 seconds; safety monitor is internal",
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        await scheduler.stop()


def main(specification=None, argv=None):
    global FLASHACCEL_MEMORY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=REPOSITORY / "build/gtsim_program")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--program", type=Path, help="Run any ProgIR with a Python sweep specification")
    parser.add_argument("--params", type=Path, help="Optional base parameter file for --program")
    parser.add_argument("--lib", type=Path, default=REPOSITORY / "frontend/lib")
    parser.add_argument("--architecture", type=Path, default=ARCHITECTURE)
    parser.add_argument("--memory", type=Path, default=FULL_HBM_MEMORY)
    sweep = parser.add_mutually_exclusive_group()
    sweep.add_argument("--sweep", help="Python dict of axes or list of case dicts")
    sweep.add_argument("--sweep-file", type=Path, help="Read the Python literal from a file")
    parser.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--ram-guard", choices=("on", "off"), default="on",
                        help="Enable/disable RSS-based termination and requeue; admission RAM limits remain active")
    parser.add_argument("--ram-admission", choices=("on", "off"), default="on",
                        help="Enable/disable the RAM-derived concurrency cap")
    parser.add_argument("--memory-per-case-gib", type=float, default=10,
                        help="Estimated GiB per simulation for RAM admission (default: 10)")
    parser.add_argument("--memory-budget-fraction", type=float, default=0.75,
                        help="Fraction of available RAM used for admission (default: 0.75)")
    parser.add_argument("--skip-pilot", action="store_true",
                        help="Skip the initial serial smoke run; compiler feasibility checks still run")
    parser.add_argument("--flashaccel-memory", type=Path, default=FLASHACCEL_MEMORY)
    parser.add_argument("--pilot-config", choices=("double", "fullhbm"), default="double")
    args = parser.parse_args(argv)
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    if not math.isfinite(args.memory_per_case_gib) or args.memory_per_case_gib <= 0:
        parser.error("--memory-per-case-gib must be finite and positive")
    if not 0 < args.memory_budget_fraction <= 1:
        parser.error("--memory-budget-fraction must be in (0, 1]")
    if args.program:
        if args.output == DEFAULT_OUTPUT:
            parser.error("--program requires a separate --output directory")
        try:
            if args.sweep_file is not None:
                specification = literal_eval(args.sweep_file.read_text())
            elif args.sweep is not None:
                specification = literal_eval(args.sweep)
            specification = parameter_sets(specification)
        except (OSError, SyntaxError, ValueError) as error:
            parser.error(str(error))
        asyncio.run(run_program_sweep(args, specification))
        return
    if specification is not None or args.sweep is not None or args.sweep_file is not None:
        parser.error("sweep parameters require --program")
    FLASHACCEL_MEMORY = args.flashaccel_memory.resolve()
    if not FLASHACCEL_MEMORY.is_file():
        parser.error(f"memory configuration does not exist: {FLASHACCEL_MEMORY}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
