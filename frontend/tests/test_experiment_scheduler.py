"""Exercise sweep resource handling with small synthetic process trees."""

import asyncio
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.common import sweep_scheduler as sweep


class ExperimentSchedulerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.memory = [sweep.Memory("synthetic", 1024**3, 0)]
        cpu = patch.object(sweep, "cpu_capacity", return_value=2)
        cgroup = patch.object(sweep, "cgroup_paths", return_value=())
        for mock in (cpu, cgroup):
            mock.start()
            self.addCleanup(mock.stop)
        self.scheduler = sweep.SweepScheduler(
            self.root, self.root, job_ram=64 * 1024**2,
            memory=lambda: self.memory, sample_seconds=.02)
        self.addCleanup(self.scheduler.close)

    def job(self, name, code, *, key=None, timeout=4, slots=None):
        return sweep.Job(name, key or name, (sys.executable, "-c", code),
                         self.root / name, cpu_slots=slots, timeout=timeout)

    def test_cpu_and_memory_admission(self):
        job = self.job("one", "", slots=2)
        self.scheduler.target = 2
        self.assertTrue(self.scheduler.admits(job, self.memory))
        active = sweep.Running(job, None, 0, self.scheduler.job_ram, 1)
        self.scheduler.active[job.id] = active
        self.assertFalse(self.scheduler.admits(job, self.memory))
        other = self.job("other", "", slots=1)
        self.memory = [sweep.Memory("host", 1024**3, 760 * 1024**2),
                       sweep.Memory("cgroup", 512 * 1024**2, 350 * 1024**2)]
        self.assertFalse(self.scheduler.admits(other, self.memory))
        active.anonymous = 48 * 1024**2
        self.assertTrue(self.scheduler.admits(other, self.memory))

    async def test_completion_order_resume_and_deduplication(self):
        first = self.job("first", "import time; time.sleep(.4)")
        second = self.job("second", "import time; time.sleep(.04)")
        duplicate = self.job("duplicate", "raise AssertionError", key=second.fingerprint)
        jobs = [first, second, duplicate]
        values = await self.scheduler.run("test", jobs, ramp=False)
        self.assertEqual(list(values), ["second", "first"])
        self.assertTrue(all(result["status"] == "SUCCESS" for result in values.values()))
        resumed = await self.scheduler.run("test", jobs, ramp=False)
        self.assertEqual(resumed, values)
        self.assertFalse(duplicate.output.exists())
        changed = self.job("first", "raise RuntimeError", key="new")
        results = await self.scheduler.run("test", [changed])
        self.assertEqual(results["first"]["status"], "FAILED")

    async def test_timeout_kills_descendants_that_ignore_term(self):
        marker = self.root / "survived"
        descendant = ("import signal,time; from pathlib import Path; "
                      "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                      f"time.sleep(.8); Path({str(marker)!r}).touch()")
        job = self.job("timeout", f"import subprocess,sys,time; "
                       f"subprocess.Popen([sys.executable, '-c', {descendant!r}]); "
                       "time.sleep(5)", timeout=.3)
        values = await self.scheduler.run("test", [job])
        self.assertEqual(values[job.id]["status"], "TIMEOUT")
        await asyncio.sleep(.8)
        self.assertFalse(marker.exists())

    async def test_memory_emergency_retries_newest_and_drains_queue(self):
        jobs = [self.job(str(i), f"import time; time.sleep({.35 if i == 0 else .25})")
                for i in range(3)]

        def memory():
            # Make the second job cross the threshold until termination starts.
            newest = self.scheduler.active.get("1")
            pressured = newest and not newest.deferred and self.scheduler.limit == 2
            return [sweep.Memory("host", 1024**3, int(.96 * 1024**3) if pressured else 0)]

        self.scheduler.memory = memory
        results = await asyncio.wait_for(self.scheduler.run("test", jobs, ramp=False), 4)
        self.assertEqual(self.scheduler.limit, 1)
        self.assertEqual(results["1"]["attempt"], 1)
        self.assertEqual(results["0"]["attempt"], 0)
        self.assertTrue(all(record["status"] == "SUCCESS" for record in results.values()))
        self.assertTrue((jobs[1].output / "attempt-1" / "driver.log").exists())

    async def test_pilot_measures_simultaneous_child_memory(self):
        size = 24 * 1024**2
        descendant = f"import time; x=bytearray({size}); time.sleep(.3)"
        job = self.job("pilot", f"import subprocess,sys; a=bytearray({size}); "
                       f"subprocess.check_call([sys.executable, '-c', {descendant!r}])")
        result = await self.scheduler.pilot(job)
        usage = result["resources"]
        self.assertGreater(usage["sampled_peak_tree_rss_bytes"], 2 * size)
        self.assertGreater(usage["sampled_peak_tree_rss_bytes"], usage["ru_maxrss_bytes"] + size)
        self.assertEqual(self.scheduler.job_ram, 2 * usage["peak_ram_bytes"])
