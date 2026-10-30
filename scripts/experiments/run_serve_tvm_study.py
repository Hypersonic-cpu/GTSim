#!/usr/bin/env python3
"""Run corrected baselines and the native serving search without overlapping simulations."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'validation/results/serve-3m2f/study-a100-fp32'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    phases = [
        ('baseline', [sys.executable, str(ROOT/'scripts/experiments/run_serve_3m2f.py'),
                      '--out-dir', str(OUT/'baseline')]),
        ('search', [sys.executable, str(ROOT/'scripts/experiments/run_serve_searches.py'),
                    '--out-dir', str(OUT/'search')]),
    ]
    records = []
    for name, command in phases:
        record = dict(phase=name, status='running', command=command)
        records.append(record)
        (OUT/'study-status.json').write_text(json.dumps(records, indent=2)+'\n')
        print('STUDY START', name, flush=True)
        started = time.monotonic()
        process = subprocess.run(command, cwd=ROOT, check=False)
        record.update(status='complete' if process.returncode == 0 else 'failed',
                      returncode=process.returncode, wall_seconds=time.monotonic()-started)
        (OUT/'study-status.json').write_text(json.dumps(records, indent=2)+'\n')
        print('STUDY END', name, record, flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
