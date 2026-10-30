#!/usr/bin/env python3
"""Run the real Qwen serving search matrix serially, retaining every artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'validation/results/serve-3m2f/tvm-search')
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = [
        ('fa2', ['--kind', 'fa2', '--trials', '12']),
        ('gdn_qkvz', ['--kind', 'projection', '--n', '12288', '--k', '4096', '--trials', '8']),
        ('gqa_qkvgate', ['--kind', 'projection', '--n', '10240', '--k', '4096', '--trials', '8']),
        ('output_proj', ['--kind', 'projection', '--n', '4096', '--k', '4096', '--trials', '8']),
        ('mlp_gate_up', ['--kind', 'projection', '--n', '24576', '--k', '4096', '--trials', '8']),
        ('mlp_down', ['--kind', 'projection', '--n', '4096', '--k', '12288', '--trials', '8']),
        ('gdn_full', ['--kind', 'gdn', '--trials', '16']),
    ]
    sources = [*sorted((ROOT / 'scripts/tvm').glob('serve_*.py')),
               ROOT / 'workloads/serve/semantics.py', ROOT / 'build/gtsim_serve_ir',
               ROOT / 'configs/v0.1.0/a100-pcie40-3m2f-grade1-simple.yaml']
    spec = dict(workers=1, previous_simulation_peak_bytes=3772419264,
                policy='One worker because measured peak exceeds 3GB', cases=cases,
                source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    spec_path = out / 'run-spec.json'
    if spec_path.exists() and json.loads(spec_path.read_text()) != json.loads(json.dumps(spec)):
        raise RuntimeError('Sources changed; retain this run and choose a new output directory')
    spec_path.write_text(json.dumps(spec, indent=2) + '\n')
    rows = []
    for name, options in cases:
        target = out / name
        if (target / 'results.json').exists():
            result = json.loads((target / 'results.json').read_text())
            rows.append(dict(name=name, status='complete', cycles=result['best']['cycles'], reused=True))
            continue
        if target.exists():
            raise RuntimeError('Incomplete search retained at ' + str(target) + '; use a new output root')
        command = [sys.executable, str(ROOT / 'scripts/tvm/serve_search.py'), *options, '--out-dir', str(target)]
        print('START', name, flush=True)
        started = time.monotonic()
        rows.append(dict(name=name, status='running', command=command))
        (out / 'batch-status.json').write_text(json.dumps(rows, indent=2) + '\n')
        with (out / (name + '.console.log')).open('w') as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        rows[-1].update(status='complete' if process.returncode == 0 else 'failed',
                        returncode=process.returncode, wall_seconds=time.monotonic() - started)
        if process.returncode == 0:
            result = json.loads((target / 'results.json').read_text())
            rows[-1].update(cycles=result['best']['cycles'], us=result['best']['us'],
                            successful=result['successful_candidates'])
        (out / 'batch-status.json').write_text(json.dumps(rows, indent=2) + '\n')
        print('END', name, rows[-1], flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
