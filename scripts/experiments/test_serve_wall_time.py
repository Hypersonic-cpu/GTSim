"""Regression checks for wall-time accounting and incomplete client captures."""
import json
import tempfile
import unittest
from pathlib import Path

from audit_serve_wall_time import interval_union_ns, read_client


class WallTimeAuditTests(unittest.TestCase):
    def test_overlapping_activity_is_not_double_counted(self):
        # A copy can overlap a kernel; summing their durations would incorrectly
        # charge 25 ns against a 20 ns wall window.
        self.assertEqual(interval_union_ns([(0, 10), (5, 20)]), 20)
        self.assertEqual(interval_union_ns([(0, 10), (15, 20)]), 15)
        self.assertEqual(interval_union_ns([(20, 25), (0, 20)]), 25)

    def test_end_to_end_keeps_waits_and_rejects_missing_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # 255 output intervals of 400 ms, even if kernels occupy only a
            # small fraction of each interval. No GPU-duration input is used.
            times = [i * 400_000_000 for i in range(256)]
            metrics = dict(success=True, done=True,
                           usage=dict(completion_tokens=256, prompt_tokens=32768),
                           first_text_monotonic_ns=times[0],
                           last_text_monotonic_ns=times[-1], tpot_ms=400, ttft_s=1)
            (root / "trace-metrics.json").write_text(json.dumps(metrics))
            (root / "trace-request.json").write_text(json.dumps(dict(prompt=[1] * 32768, max_tokens=256)))
            events = [dict(monotonic_ns=t, event=dict(choices=[dict(text="token")])) for t in times]
            path = root / "trace-events.jsonl"
            path.write_text("\n".join(json.dumps(e) for e in events))
            client, _ = read_client(root)
            self.assertEqual(client["tpot_ms"], 400)
            self.assertEqual(client["intervals"], 255)
            path.write_text("\n".join(json.dumps(e) for e in events[:-1]))
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                read_client(root)


if __name__ == "__main__":
    unittest.main()
