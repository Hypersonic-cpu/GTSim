"""Run the row-major frontend IR matrix on a single selectable-backend simulator.

FullHBM is a shared control: no Flash tensor is eligible for prefetch. Each HBF
case uses the same 4F1M geometry/SRAM profile with prefetch enabled or disabled.
Sources in frontend/lib are read without modification.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORKLOADS = ('gemm512', 'gemm1024', 'gemm2048', 'qwen30b', 'qwen235b')
MODES = (('hbm', False), *((b, p) for b in ('hbfine', 'eventlab', 'flashlab') for p in (False, True)))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(out, workload, backend, prefetch, streaming=True, base_die_sram_mib=2):
    name = f'{workload}-{backend}-pf{int(prefetch)}'
    case = out / name
    case.mkdir(exist_ok=False)
    if workload.startswith('gemm'):
        size = int(workload[4:])
        source = ROOT / 'frontend/asm/gemm/reference/example.progir'
        params = dict(M=size, N=size, K=size, NUM_SMS=132)
        weights = ['B']
    else:
        source = ROOT / 'frontend/asm/qwen3/one_block.progir'
        param_file = 'b1.json' if workload == 'qwen30b' else 'qwen3-235b-a22b-b1.json'
        params = json.loads((source.parent / param_file).read_text())
        weights = re.findall(r'prog.tensor @(W_\w+)\s*:', source.read_text())
    inputs = case / 'inputs'
    inputs.mkdir()
    (inputs / 'params.json').write_text(json.dumps(params, indent=2)+'\n')
    (inputs / 'placements.json').write_text(json.dumps({w: 'HBM' if backend == 'hbm' else 'HBF' for w in weights}, indent=2)+'\n')
    residency = dict(stacks=4, dies_per_stack=8, planes_per_die=96, planes_per_group=32,
                     page_bytes=4096, plane_bytes=8192, stack_bytes=base_die_sram_mib*1024*1024,
                     line_bytes=128, buffer_count=2, streaming=streaming)
    (inputs / 'residency.json').write_text(json.dumps(residency, indent=2)+'\n')
    if backend == 'hbm':
        config = (ROOT / 'configs/memconf/0F5M_channels.yaml').read_text()
    else:
        config = (ROOT / 'configs/memconf/4F1M_grouped_planes.yaml').read_text()
        config = config.replace('backend: hbfine', f'backend: {backend}')
        config = config.replace('base_die_sram_bytes_per_stack: 2097152',
                                f'base_die_sram_bytes_per_stack: {base_die_sram_mib*1024*1024}')
        if backend in ('eventlab', 'flashlab'):
            # Native transport owns offchip timing; avoid counting it twice.
            config = config.replace('bandwidth_GBps: 6144\n    latency_cycles: 10',
                                    'bandwidth_GBps: 1000000000000\n    latency_cycles: 0')
    memory = inputs / 'memory.yaml'
    memory.write_text(config)
    binary = ROOT / 'build/gtsim_program'
    command = [sys.executable, 'frontend/tools/run_qwen_simulations.py',
               '--source', str(source), '--params', str(inputs/'params.json'),
               '--tensor-placements', str(inputs/'placements.json'),
               '--runner', str(binary), '--architecture', 'configs/v0.1.0/h100-sxm.yaml',
               '--memory', str(memory), '--out', str(case/'run'), '--max-cycles', '1000000000']
    if prefetch:
        command.extend(['--prefetch-config', str(inputs/'residency.json')])
    record = dict(workload=workload, backend=backend, prefetch=prefetch, streaming=streaming, base_die_sram_mib=base_die_sram_mib, command=command,
                  binary_sha256=digest(binary), config_sha256=digest(memory),
                  source_sha256=digest(source), state='running')
    start = time.perf_counter()
    with (case/'launcher.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'PYTHONPATH':'frontend/src'})
        record['pid'] = process.pid
        (case/'job.json').write_text(json.dumps(record, indent=2)+'\n')
        record['returncode'] = process.wait()
    record['wall_seconds'] = time.perf_counter()-start
    record['state'] = 'completed' if record['returncode'] == 0 else 'failed'
    (case/'job.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--streaming', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--prefetch-only', action='store_true')
    parser.add_argument('--base-die-sram-mib',type=int,default=2)
    parser.add_argument('--workloads', nargs='+', choices=WORKLOADS, default=WORKLOADS)
    parser.add_argument('--backends', nargs='+', choices=('hbm','hbfine','eventlab','flashlab'),
                        default=('hbm','hbfine','eventlab','flashlab'))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if args.base_die_sram_mib<1 or args.base_die_sram_mib & (args.base_die_sram_mib-1):
        parser.error('--base-die-sram-mib must be a positive power of two')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = [*ROOT.glob('frontend/lib/**/*'), *ROOT.glob('frontend/src/**/*.py'),
             *ROOT.glob('src/memory/*'), *ROOT.glob('include/memory/*'),
             *ROOT.glob('cmake/patches/*')]
    (out/'source-sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):digest(p) for p in sorted(files) if p.is_file()}, indent=2)+'\n')
    (out/'submodules.txt').write_text(subprocess.check_output(['git','submodule','status'],cwd=ROOT,text=True))
    jobs = [(w,b,p) for w in args.workloads for b,p in MODES if b in args.backends and (not args.prefetch_only or p)]
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(run_case,out,*job,args.streaming,args.base_die_sram_mib) for job in jobs]
        records = [future.result() for future in concurrent.futures.as_completed(futures)]
    (out/'jobs.json').write_text(json.dumps(records, indent=2)+'\n')
    return int(any(record['returncode'] for record in records))


if __name__ == '__main__':
    raise SystemExit(main())
