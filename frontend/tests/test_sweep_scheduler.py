"""Check scheduler reporting and RAM retries without launching simulations."""

import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments import sweep_qwen235b_flashaccel as sweep


class SchedulerReportingTests(unittest.IsolatedAsyncioTestCase):
    async def test_console_is_compact_and_routine_events_stay_in_file(self):
        with tempfile.TemporaryDirectory() as folder:
            output = io.StringIO()
            scheduler = sweep.Scheduler(Path(folder), Path("runner"), 2, ram_guard=False)
            with redirect_stdout(output):
                scheduler.log("JOB_STARTED", case="preflight-case", pid=42)
                scheduler.log("JOB_FINISHED", case="preflight-case", status="INFEASIBLE")
                self.assertEqual(output.getvalue(), "")
                with patch.object(sweep.shutil, "get_terminal_size",
                                  return_value=SimpleNamespace(columns=40)):
                    scheduler.report_status()
                    scheduler.log("PILOT_SKIPPED", reason="--skip-pilot was specified")
            lines = output.getvalue().splitlines()
            self.assertTrue(all(len(line) <= 40 for line in lines))
            self.assertTrue(any("run=0" in line for line in lines))
            self.assertIn("pilot skipped", output.getvalue())
            self.assertNotIn('"event"', output.getvalue())
            events = [json.loads(line)["event"] for line in
                      (Path(folder) / "scheduler_events.jsonl").read_text().splitlines()]
            self.assertIn("JOB_STARTED", events)
            self.assertIn("JOB_FINISHED", events)

    async def test_status_counts_and_periodic_reports(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            scheduler = sweep.Scheduler(Path(folder), Path("runner"), 2, ram_guard=False)
            scheduler.job_states = {"a": "SUCCESS", "b": "FAILED", "c": "WAITING",
                                    "d": "DEFERRED_RAM", "e": "INFEASIBLE"}
            scheduler.report_status()
            status = json.loads((Path(folder) / "scheduler_status.json").read_text())
            self.assertEqual(status["completed_success"], 1)
            self.assertEqual(status["completed_failed"], 1)
            self.assertEqual(status["waiting"], 2)
            self.assertEqual(status["infeasible"], 1)
            with patch.object(sweep, "STATUS_INTERVAL_SECONDS", 0.01):
                await scheduler.start()
                await asyncio.sleep(0.035)
                await scheduler.stop()
            events = [json.loads(line) for line in
                      (Path(folder) / "scheduler_events.jsonl").read_text().splitlines()]
            self.assertGreaterEqual(sum(e["event"] == "STATUS" for e in events), 3)

    async def test_guard_terminates_newest_first_and_logs_actions(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            scheduler = sweep.Scheduler(Path(folder), Path("runner"), 2)
            scheduler.allowed_running = 2
            scheduler.active = {
                "old": {"process": SimpleNamespace(pid=1), "started": 1,
                        "ram_guard": False, "case": "old"},
                "new": {"process": SimpleNamespace(pid=2), "started": 2,
                        "ram_guard": False, "case": "new"},
            }
            scheduler._terminate = AsyncMock()
            with patch.object(sweep.asyncio, "sleep", AsyncMock(
                    side_effect=[None, asyncio.CancelledError()])), \
                 patch.object(sweep, "available_memory_bytes", return_value=100), \
                 patch.object(sweep, "process_tree_rss", return_value=100):
                with self.assertRaises(asyncio.CancelledError):
                    await scheduler._memory_monitor()
            self.assertEqual([c.args[0].pid for c in
                              scheduler._terminate.await_args_list], [2, 1])
            events = [json.loads(line) for line in
                      (Path(folder) / "scheduler_events.jsonl").read_text().splitlines()]
            self.assertEqual(sum(e["event"] == "RAM_GUARD_TRIGGERED" for e in events), 2)
            self.assertTrue(any(e["event"] == "CONCURRENCY_REDUCED" for e in events))

    async def test_oom_is_logged_and_requeued(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            scheduler = sweep.Scheduler(root, Path("runner"), 2, ram_guard=False)
            scheduler.allowed_running = 2
            case = sweep.make_case("main", 10000, "fullhbm", "none", "off",
                                   (16, 64, 16, 2), root)
            process = SimpleNamespace(pid=42, wait=AsyncMock(return_value=1))

            async def spawn(*args, **kwargs):
                kwargs["stdout"].write("MemoryError: out of memory\n")
                kwargs["stdout"].flush()
                return process

            with patch.object(sweep.asyncio, "create_subprocess_exec", side_effect=spawn):
                result = await scheduler.execute(case)
            self.assertEqual(result.status, "DEFERRED_RAM")
            self.assertEqual(scheduler.allowed_running, 1)
            queue = asyncio.Queue()
            with patch.object(sweep.asyncio, "sleep", AsyncMock()):
                await scheduler.requeue(case, queue)
            self.assertIs(queue.get_nowait(), case)
            self.assertEqual(case.attempt, 1)
            events = [json.loads(line)["event"] for line in
                      (root / "scheduler_events.jsonl").read_text().splitlines()]
            for expected in ("OUT_OF_MEMORY", "RAM_RETRY_SCHEDULED", "RAM_REQUEUED"):
                self.assertIn(expected, events)


if __name__ == "__main__":
    unittest.main()
