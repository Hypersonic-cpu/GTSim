"""Run frontend GEMM IR controls for the grouped FlashAccel hardware profile.

Other native backends join the final matrix after their grouped command paths
pass the native geometry contract. This script only runs FullHBM and HBFine.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'validation/results/1112-flashaccel/ir-gemm-controls'
OUT.mkdir(parents=True, exist_ok=True)


def run(job):
    backend, prefetch, size = job
    stem = f'{backend}-pf{prefetch}-{size}'
    config = ROOT / ('configs/memconf/0F5M_channels.yaml' if backend == 'hbm'
                     else 'configs/memconf/4F1M_grouped_planes.yaml')
    binary = ROOT / 'build/gtsim_program'
    params=OUT/f'{stem}-params.json'
    params.write_text(json.dumps(dict(M=size,N=size,K=size,NUM_SMS=132),indent=2))
    placements=OUT/f'{stem}-placements.json'
    placements.write_text(json.dumps({'B':'HBM' if backend=='hbm' else 'HBF'}))
    residency=OUT/f'{stem}-double-buffer.json'
    if not residency.exists():
        residency.write_text(json.dumps(dict(stacks=4,dies_per_stack=8,planes_per_die=96,
            planes_per_group=32,page_bytes=4096,plane_bytes=8192,stack_bytes=2097152,
            line_bytes=128,buffer_count=2,streaming=True),indent=2))
    command = [sys.executable,'frontend/tools/run_qwen_simulations.py',
               '--source','frontend/asm/gemm/reference/example.progir',
               '--params',str(params),'--runner',str(binary),
               '--architecture','configs/v0.1.0/h100-sxm.yaml','--memory',str(config),
               '--tensor-placements',str(placements),'--out',str(OUT/stem)]
    if prefetch:command.extend(['--prefetch-config',str(residency)])
    record = dict(backend=backend, prefetch=bool(prefetch), dimensions=[size]*3,
                  command=command, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
    (OUT / f'{stem}-config.yaml').write_text(config.read_text())
    start = time.perf_counter()
    with (OUT / f'{stem}.log').open('w') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONPATH":"frontend/src"})
        record['pid'] = process.pid
        (OUT / f'{stem}.json').write_text(json.dumps(record, indent=2)+'\n')
        record['exit_code'] = process.wait()
    record['wall_seconds'] = time.perf_counter()-start
    (OUT / f'{stem}.json').write_text(json.dumps(record, indent=2)+'\n')
    return record['exit_code']


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=os.cpu_count() or 1)
    args=parser.parse_args()
    if args.jobs<1:parser.error('--jobs must be positive')
    JOBS=args.jobs
    jobs = [(backend, prefetch, size) for size in (512, 1024, 2048)
            for backend, prefetch in [('hbm', 0), ('hbfine', 0), ('hbfine', 1)]]
    with concurrent.futures.ThreadPoolExecutor(JOBS) as pool:
        codes = list(pool.map(run, jobs))
    raise SystemExit(0 if all(code == 0 for code in codes) else 1)
