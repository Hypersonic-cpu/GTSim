"""Event-driven execution of independent, fingerprinted simulation jobs."""

import asyncio
from collections import Counter, deque
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time


CPU_BUDGET = 180
RAM_HARD_LIMIT = .95
RAM_ADMISSION_TARGET = .85
RAM_RECOVERY_TARGET = .90
PILOT_RAM_MULTIPLIER = 2
DEFAULT_JOB_RAM = 10 * 1024**3
TIMEOUT_SECONDS = 1800
STARTUP_CONCURRENCY = 8
COMPLETION_INCREMENT = 4


def use_pidfd_child_watcher():
    """Use the Linux event loop's process watcher without one thread per job."""
    if sys.version_info < (3, 14):
        asyncio.get_event_loop_policy().set_child_watcher(asyncio.PidfdChildWatcher())


def fingerprint(inputs):
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def cgroup_paths():
    """Return this process's cgroup and applicable ancestors within the mount."""
    member = next((line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines()
                   if line.startswith("0::")), None)
    if member is None:
        return ()
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        if fields[fields.index("-") + 1] != "cgroup2":
            continue
        root, mount = (Path(value.replace(r"\040", " ")) for value in fields[3:5])
        relative = Path(member).relative_to(root) if Path(member).is_relative_to(root) else Path()
        current = mount / relative
        if not (current / "cgroup.procs").is_file():
            continue
        return tuple(path for path in (current, *current.parents)
                     if path.is_relative_to(mount))
    return ()


def cpu_capacity():
    bounds = [float(len(os.sched_getaffinity(0)))]
    for path in cgroup_paths():
        quota_file = path / "cpu.max"
        if quota_file.is_file():
            quota, period = quota_file.read_text().split()
            if quota != "max":
                bounds.append(int(quota) / int(period))
    return min(bounds)


@dataclass(frozen=True)
class Memory:
    name: str
    capacity: int
    used: int


def memory_constraints():
    values = dict((line.split()[0], int(line.split()[1]) * 1024)
                  for line in Path("/proc/meminfo").read_text().splitlines()
                  if line.startswith(("MemTotal:", "MemAvailable:")))
    constraints = [Memory("host", values["MemTotal:"],
                          values["MemTotal:"] - values["MemAvailable:"])]
    for path in cgroup_paths():
        if not (path / "memory.max").is_file():
            continue
        limit = (path / "memory.max").read_text().strip()
        if limit != "max":
            constraints.append(Memory(str(path), int(limit), read_int(path / "memory.current")))
    return constraints


def read_int(path):
    return int(path.read_text().strip())


def process_tree_memory(pid):
    """Sample simultaneous RSS and anonymous private resident bytes."""
    seen, pending, rss, anonymous = set(), [pid], 0, 0
    while pending:
        child = pending.pop()
        if child in seen:
            continue
        seen.add(child)
        proc = Path(f"/proc/{child}")
        try:
            fields = dict(line.split(":", 1) for line in
                          (proc / "status").read_text().splitlines()
                          if line.startswith(("VmRSS:", "RssAnon:")))
            rss += int(fields.get("VmRSS", "0").split()[0]) * 1024
            anonymous += int(fields.get("RssAnon", "0").split()[0]) * 1024
            for thread in (proc / "task").iterdir():
                try:
                    pending.extend(map(int, (thread / "children").read_text().split()))
                except (FileNotFoundError, ProcessLookupError):
                    continue
        except (FileNotFoundError, ProcessLookupError):
            continue
    return rss, anonymous


async def terminate_group(process, grace=2):
    # start_new_session makes the original pid the group id even after the leader exits.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), grace)
    except asyncio.TimeoutError:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


@dataclass(frozen=True)
class Job:
    id: str
    fingerprint: str
    command: tuple[str, ...]
    output: Path
    cpu_slots: int | None = None
    timeout: float = TIMEOUT_SECONDS


@dataclass
class Running:
    job: Job
    process: asyncio.subprocess.Process
    sequence: int
    reservation: int
    cpu_slots: int
    cgroup: Path | None = None
    rss: int = 0
    anonymous: int = 0
    peak: int = 0
    deferred: bool = False

    def sample(self):
        self.rss, self.anonymous = process_tree_memory(self.process.pid)
        self.peak = max(self.peak, self.rss)


class SweepScheduler:
    def __init__(self, root, cwd, *, cpu_budget=CPU_BUDGET, job_ram=DEFAULT_JOB_RAM,
                 memory=memory_constraints, sample_seconds=1, status_seconds=300):
        self.root, self.cwd = Path(root), Path(cwd)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".scheduler.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        self.cpu_budget = min(CPU_BUDGET, cpu_budget, cpu_capacity())
        self.limit = math.floor(self.cpu_budget)
        self.job_ram = job_ram
        self.default_cpu_slots = 1
        self.memory = memory
        self.sample_seconds, self.status_seconds = sample_seconds, status_seconds
        self.active = {}
        self.emergency = False
        self.error = None
        self.peak_utilization = 0
        self.sequence = 0
        self.cgroup = None
        self.last_status = 0
        self.target = STARTUP_CONCURRENCY
        self.mode = "STARTUP"
        self._kernel_ceiling()

    def _kernel_ceiling(self):
        paths = cgroup_paths()
        group = paths[0] / f"gtsim-sweep-{os.getpid()}" if paths else None
        try:
            if group is None:
                raise OSError("cgroup v2 unavailable")
            constraints = self.memory()
            ceiling = min(int(item.capacity * RAM_HARD_LIMIT) - item.used for item in constraints)
            if ceiling <= 0:
                raise OSError("no memory headroom for sweep cgroup")
            group.mkdir()
            try:
                (group / "memory.max").write_text(str(ceiling))
                (group / "cgroup.subtree_control").write_text("+memory")
            except OSError:
                group.rmdir()
                raise
            self.cgroup = group
            self.event("KERNEL_MEMORY_CEILING", bytes=ceiling)
        except OSError as error:
            self.event("MEMORY_MONITORED", reason=str(error))
            print("RAM limit at 95% is monitored. Kernel enforcement unavailable.", flush=True)

    def close(self):
        try:
            if self.cgroup is not None:
                self.cgroup.rmdir()
        finally:
            self.lock.close()

    def event(self, event, **fields):
        with (self.root / "scheduler_events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(time=time.time(), event=event, **fields)) + "\n")
        if event in {"RAM_GUARD", "RAM_RECOVERED", "RAM_RETRY", "OOM", "TIMEOUT",
                     "SCHEDULER_FAILED", "ADMISSION_MODE"}:
            print(f"{event.replace('_', ' ')} {fields}", flush=True)

    def _status(self, phase, pending, results, running, durations, force=False):
        now = time.monotonic()
        if not force and now - self.last_status < self.status_seconds:
            return
        self.last_status = now
        counts = Counter(result["status"] for result in results.values())
        slots = min(self.limit, self.target)
        eta = (len(pending) + running) * sum(durations) / len(durations) / slots if (
            durations and slots) else None
        status = dict(phase=phase, completed=len(results), total=len(results) + len(pending) + running,
                      running=running, queued=len(pending), statuses=counts, concurrency=slots,
                      peak_ram_utilization=self.peak_utilization, estimated_seconds_remaining=eta,
                      last_scheduler_error=self.error, job_reservation_bytes=self.job_ram,
                      cpu_budget=self.cpu_budget, admission_mode=self.mode,
                      ram_kernel_enforced=self.cgroup is not None)
        write_json(self.root / "scheduler_status.json", status)
        print(f"{phase} {len(results)}/{status['total']}  run {running}  queue {len(pending)}"
              f"  ok {counts['SUCCESS']}  failed {counts['FAILED'] + counts['OOM'] + counts['TIMEOUT']}"
              f"  infeasible {counts['INFEASIBLE']}  limit {slots}", flush=True)

    def admission_limit(self, job, memory):
        slots = job.cpu_slots or self.default_cpu_slots
        cpu_free = self.cpu_budget - sum(item.cpu_slots for item in self.active.values())
        # Credit anonymous resident memory already included in system usage.
        # File-backed RSS may be shared or reclaimable, so do not credit it.
        reserved = sum(max(0, max(self.job_ram, item.reservation) - item.anonymous)
                       for item in self.active.values())
        memory_free = min(int(item.capacity * RAM_ADMISSION_TARGET) - item.used - reserved
                          for item in memory)
        additional = max(0, min(math.floor(cpu_free / slots), memory_free // self.job_ram))
        return min(self.limit, len(self.active) + additional)

    def admits(self, job, memory):
        return not self.emergency and len(self.active) < min(
            self.target, self.admission_limit(job, memory))

    def _set_target(self, requested, job, memory, *, completed=None):
        safe = self.admission_limit(job, memory)
        target = min(requested, safe)
        mode = "STEADY" if target >= safe else "STARTUP"
        if mode != self.mode:
            self.event("ADMISSION_MODE", mode=mode, target=target, admission_limit=safe)
        if target > self.target or completed is None and target != self.target:
            self.event("CONCURRENCY_TARGET", previous=self.target, target=target,
                       admission_limit=safe, completion=completed)
        self.target, self.mode = target, mode

    async def _monitor(self, sample):
        for item in self.active.values():
            item.sample()
            self._raise_reservation(item.peak)
        usage = max(item.used / item.capacity for item in sample)
        self.peak_utilization = max(self.peak_utilization, usage)
        if usage >= RAM_HARD_LIMIT:
            self.emergency = True
        if not self.emergency:
            return
        while usage >= RAM_RECOVERY_TARGET:
            victim = next((item for item in reversed(self.active.values())
                           if not item.deferred and item.process.returncode is None), None)
            if victim is None:
                if usage >= RAM_HARD_LIMIT:
                    raise RuntimeError("RAM emergency cannot be resolved using experiment jobs")
                return
            victim.deferred = True
            self._reduce_concurrency()
            self.event("RAM_GUARD", case=victim.job.id, utilization=usage, limit=self.limit)
            await terminate_group(victim.process)
            sample = self.memory()
            usage = max(item.used / item.capacity for item in sample)
        self.emergency = False
        self.event("RAM_RECOVERED", utilization=usage)

    def _raise_reservation(self, peak):
        if peak > self.job_ram:
            self.job_ram = peak
            self.event("RAM_RESERVATION_INCREASED", bytes=peak)

    def _reduce_concurrency(self):
        self.limit = max(1, min(self.limit, len(self.active)) - 1)
        self.target = min(self.target, self.limit)
        self.mode = "STEADY"

    async def _launch(self, job, attempt):
        job.output.mkdir(parents=True, exist_ok=True)
        attempt_dir = job.output / f"attempt-{attempt}"
        attempt_dir.mkdir(exist_ok=True)
        for name in ("driver.log", "simulation.log", "resource_usage.json", "result.json"):
            source = job.output / name
            if source.exists():
                source.replace(attempt_dir / name)
        resource_path = job.output / "resource_usage.json"
        group = None
        if self.cgroup is not None:
            group = self.cgroup / f"job-{self.sequence}"
            group.mkdir()
        command = (sys.executable, str(Path(__file__).with_name("sweep_child.py")),
                   str(resource_path), str(group or ""), *job.command)
        started = time.monotonic()
        try:
            with (job.output / "driver.log").open("w") as log:
                process = await asyncio.create_subprocess_exec(
                    *command, cwd=self.cwd, stdout=log, stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True)
        except BaseException:
            if group is not None:
                group.rmdir()
            raise
        self.sequence += 1
        entry = Running(job, process, self.sequence, self.job_ram,
                        job.cpu_slots or self.default_cpu_slots, group)
        self.active[job.id] = entry
        self.event("JOB_STARTED", case=job.id, pid=process.pid, attempt=attempt)
        return asyncio.create_task(self._finish(entry, started, attempt))

    async def _finish(self, entry, started, attempt):
        job, process = entry.job, entry.process
        error, cgroup_peak, cgroup_oom = "", None, False
        try:
            entry.sample()
            try:
                code = await asyncio.wait_for(process.wait(), job.timeout)
                status = "DEFERRED_RAM" if entry.deferred else (
                    "SUCCESS" if code == 0 else "INFEASIBLE" if code == 78 else "FAILED")
            except asyncio.TimeoutError:
                status, error = "TIMEOUT", f"exceeded {job.timeout}s timeout"
                self.event("TIMEOUT", case=job.id, seconds=job.timeout)
                await terminate_group(process)
                code = process.returncode
            if status == "FAILED":
                logs = []
                for name in ("driver.log", "simulation.log"):
                    path = job.output / name
                    if path.exists():
                        with path.open("rb") as stream:
                            stream.seek(max(0, path.stat().st_size - 16384))
                            logs.append(stream.read().decode(errors="replace"))
                error = f"exit code {code}"
                if any(marker in "\n".join(logs).lower() for marker in (
                        "memoryerror", "std::bad_alloc", "cannot allocate memory", "out of memory")):
                    status = "OOM"
                    self._reduce_concurrency()
                    self.event("OOM", case=job.id, limit=self.limit)
        finally:
            # A failed or canceled leader can still own live descendants.
            await terminate_group(process)
            self.active.pop(job.id, None)
            if entry.cgroup is not None:
                if (entry.cgroup / "memory.peak").is_file():
                    cgroup_peak = read_int(entry.cgroup / "memory.peak")
                    events = dict(line.split() for line in
                                  (entry.cgroup / "memory.events").read_text().splitlines())
                    cgroup_oom = int(events.get("oom_kill", 0)) > 0
                entry.cgroup.rmdir()
        usage_path = job.output / "resource_usage.json"
        usage = read_json(usage_path)
        # ru_maxrss is an individual-process peak, only a lower bound on tree RSS.
        peak = max(entry.peak, usage.get("ru_maxrss_bytes", 0))
        usage.update(sampled_peak_tree_rss_bytes=entry.peak, peak_ram_bytes=peak,
                     sample_interval_seconds=self.sample_seconds, cgroup_peak_bytes=cgroup_peak)
        if cgroup_oom and status == "FAILED":
            status = "OOM"
            self._reduce_concurrency()
            self.event("OOM", case=job.id, limit=self.limit, source="cgroup")
        write_json(usage_path, usage)
        self._raise_reservation(peak)
        record = dict(id=job.id, fingerprint=job.fingerprint, status=status, returncode=code,
                      attempt=attempt, elapsed_seconds=time.monotonic() - started,
                      error=error, resources=usage)
        write_json(job.output / "case_status.json", record)
        self.event("JOB_FINISHED", **record)
        return record

    async def run(self, phase, jobs, *, ramp=True, validate_resume=lambda job: True):
        pending, tasks, results = deque(), {}, {}
        seen, directories, ids = set(), set(), set()
        for job in jobs:
            if job.fingerprint in seen:
                self.event("DEDUPLICATED", case=job.id, fingerprint=job.fingerprint)
                continue
            if job.id in ids or job.output.resolve() in directories:
                raise ValueError("distinct jobs require independent ids and directories")
            if not 0 < (job.cpu_slots or self.default_cpu_slots) <= self.cpu_budget:
                raise ValueError(f"{job.id} exceeds CPU quota")
            seen.add(job.fingerprint)
            ids.add(job.id)
            directories.add(job.output.resolve())
            saved = read_json(job.output / "case_status.json")
            if (saved.get("fingerprint") == job.fingerprint and saved.get("status") == "SUCCESS"
                    and validate_resume(job)):
                results[job.id] = saved
                self._raise_reservation(saved.get("resources", {}).get("peak_ram_bytes", 0))
                self.event("JOB_RESUMED", case=job.id)
            else:
                pending.append((job, 0))
        self.target = STARTUP_CONCURRENCY if ramp else min(self.limit, len(pending) or 1)
        self.mode = "STARTUP" if ramp else "FIXED"
        if ramp and pending:
            self._set_target(self.target, pending[0][0], self.memory())
            self.event("SWEEP_STARTUP", target=self.target)
        durations = []
        self._status(phase, pending, results, 0, durations, force=True)
        next_sample = 0
        idle_since = time.monotonic()
        try:
            while pending or tasks:
                now = time.monotonic()
                sample = self.memory()
                if now >= next_sample:
                    await self._monitor(sample)
                    next_sample = now + self.sample_seconds
                    sample = self.memory()
                if ramp and pending and not self.emergency and self.target == 0:
                    self._set_target(STARTUP_CONCURRENCY, pending[0][0], sample)
                while pending and self.admits(pending[0][0], sample):
                    job, attempt = pending.popleft()
                    tasks[await self._launch(job, attempt)] = job
                    if time.monotonic() >= next_sample:
                        break
                self._status(phase, pending, results, len(tasks), durations)
                delay = max(0, next_sample - time.monotonic())
                if not tasks:
                    if time.monotonic() - idle_since >= TIMEOUT_SECONDS:
                        raise RuntimeError("RAM admission remained blocked for 1800 seconds")
                    await asyncio.sleep(delay)
                    continue
                idle_since = time.monotonic()
                done, _ = await asyncio.wait(tasks, timeout=delay, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    job = tasks.pop(task)
                    record = task.result()
                    if record["status"] == "DEFERRED_RAM" and record["attempt"] < 3:
                        pending.append((job, record["attempt"] + 1))
                        self.event("RAM_RETRY", case=job.id, attempt=record["attempt"] + 1,
                                   limit=self.limit)
                    else:
                        results[job.id] = record
                        durations.append(record["elapsed_seconds"])
                    if (ramp and pending and not self.emergency and
                            record["status"] not in {"DEFERRED_RAM", "OOM"}):
                        self._set_target(self.target + COMPLETION_INCREMENT,
                                         pending[0][0], self.memory(), completed=job.id)
        except BaseException as error:
            self.error = str(error)
            self.event("SCHEDULER_FAILED", error=self.error)
            raise
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._status(phase, pending, results, 0, durations, force=True)
        return results

    async def pilot(self, job, *, validate_resume=lambda job: True):
        result = (await self.run("pilot", [job], ramp=False,
                                 validate_resume=validate_resume))[job.id]
        if result["status"] != "SUCCESS":
            raise RuntimeError(f"pilot {result['status']}: {result['error']}")
        usage = result["resources"]
        if usage["peak_ram_bytes"] <= 0:
            raise RuntimeError("pilot provided no usable RAM measurement")
        self.job_ram = PILOT_RAM_MULTIPLIER * max(
            usage["peak_ram_bytes"], usage.get("cgroup_peak_bytes") or 0)
        # A small measurement margin avoids rounding one fully busy core to two.
        self.default_cpu_slots = max(1, math.ceil(usage.get("mean_cores", 1) - .05))
        self.event("PILOT_ESTIMATE", job_ram_bytes=self.job_ram, cpu_slots=self.default_cpu_slots)
        return result
