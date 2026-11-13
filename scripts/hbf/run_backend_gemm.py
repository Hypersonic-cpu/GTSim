#!/usr/bin/env python3
"""Run only the four native C++ backend GEMM comparisons, retaining evidence."""
import argparse
import concurrent.futures
import csv
import json
from pathlib import Path
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', default='build')
    parser.add_argument('--backend', choices=('baseline', 'hbfine', 'eventlab', 'flashlab'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = root / 'validation/results/1112-hbf-backend-gemm'
    for directory in ('configs', 'logs'):
        (out / directory).mkdir(parents=True, exist_ok=True)
    cases = {'baseline': '0F5M_channels', 'hbfine': '4F1M_hbfine',
             'eventlab': '4F1M_eventlab', 'flashlab': '4F1M_flashlab'}
    architecture = root / 'configs/v0.1.0/h100-sxm.yaml'
    (out / 'configs/h100-sxm.yaml').write_text(architecture.read_text())

    def run(item):
        name, config = item
        source = root / f'configs/memconf/{config}.yaml'
        (out / 'configs' / source.name).write_text(source.read_text())
        command = [str(root / args.build_dir / 'gtsim_gemm'), str(architecture),
                   '--memory-config', str(source), '128', '256', '64']
        start = time.perf_counter()
        try:
            result = subprocess.run(command, cwd=root, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, timeout=180)
            log, code = result.stdout, result.returncode
        except subprocess.TimeoutExpired as error:
            log, code = (error.stdout or b'').decode(), 124
        elapsed = time.perf_counter() - start
        (out / 'logs' / f'{name}.log').write_text(log)
        row = dict(backend=name, config=str(source.relative_to(root)), exit_code=code,
                   wall_seconds=elapsed, status='completed' if code == 0 and 'Simulation completed' in log else 'failed')
        cycles = re.search(r'Total cycles: (\d+)', log)
        row['gpu_cycles'] = int(cycles[1]) if cycles else 'N/A'
        for tier in ('HBM', 'HBF'):
            match = re.search(tier + r' requests: (\d+) \(read (\d+), write (\d+)\), bytes: read (\d+), write (\d+), average latency: ([\d.]+)', log)
            for index, key in enumerate(('requests', 'read_requests', 'write_requests', 'read_bytes', 'write_bytes', 'average_latency_cycles'), 1):
                row[f'{tier.lower()}_{key}'] = (float(match[index]) if index == 6 else int(match[index])) if match else 'N/A'
        counters = dict(re.findall(r'Backend counter HBF (\w+)=(\d+)', log))
        for key in ('nand_senses', 'page_buffer_hits', 'setup_bytes', 'setup_ps'):
            row[key] = int(counters[key]) if key in counters else 'N/A'
        detailed = re.search(r'HBF detailed JSON: (\{[^\n]+\})', log)
        if detailed:
            stats = json.loads(detailed[1])
            row['nand_senses'] = stats.get('senses', 'N/A')
            row['page_buffer_hits'] = stats.get('cache_hits', 'N/A')
        row['command'] = command
        return row

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        selected = {name: config for name, config in cases.items() if not args.backend or name == args.backend}
        fresh_rows = list(pool.map(run, selected.items()))
    rows = fresh_rows
    if args.backend and (out / 'results.json').exists():
        previous = json.loads((out / 'results.json').read_text())['results']
        merged = {row['backend']: row for row in previous}
        merged.update({row['backend']: row for row in fresh_rows})
        rows = [merged[name] for name in cases if name in merged]
    metadata = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                    submodules=subprocess.check_output(['git', 'submodule', 'status'], cwd=root, text=True).splitlines(),
                    dimensions=dict(M=128, N=256, K=64), results=rows)
    (out / 'results.json').write_text(json.dumps(metadata, indent=2) + '\n')
    fields = [key for key in rows[0] if key != 'command']
    with (out / 'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    table = '| Backend | Status | GPU cycles | CPU wall seconds |\n|---|---|---:|---:|\n'
    table += ''.join(f"| {r['backend']} | {r['status']} | {r['gpu_cycles']} | {r['wall_seconds']:.4f} |\n" for r in rows)
    notes = '''
All runs use H100, M=128/N=256/K=64; A/C (including PartialC) stay in HBM.
HBM uses the same channelized naive implementation, scaled by stack count.
EventLab uses four sessions with 4 KiB stack striping, sixteen channels per stack,
32 banks/channel, 4 us sense and 75 us program. Legal setup writes are drained;
setup_ps and setup_bytes are separate from measured GPU traffic. No statistics sink
or fault callback is installed. Sessions retain native online backpressure.
FlashLab uses HbfStatic reserved blocks and RawPhysical mapping, avoiding host HBM
metadata and GC. Empty frontier batches preserve arrival times; completion receipts
are held until the GPU reaches their timestamps. A contract probe compares grouped
and incremental concurrent arrivals. This is a smoke verification, not proof for every
native scheduler policy. FlashLab limits planes/die to 16; it uses 16 instead of 32,
with doubled blocks/plane to preserve capacity. Read-buffer coalescing is enabled;
read MSHRs and ingress tags retain their disabled defaults. It retains native ECC,
TSV and HBIO costs.
For EventLab/FlashLab the GTSim HBF link is configured as zero latency, effectively
unlimited bandwidth, since the native backend owns external interface service.
HBFine retains its historical GTSim D2D configuration and internal service law.
Interface grades are not exactly equivalent (EventLab Grade2 94 GB/s/channel;
FlashLab native Grade2 96 GB/s/channel); results are comparisons of these models, not calibrated
performance equivalence. HBFine models preload as initial state; FlashLab installs
static blocks on construction. Neither generates setup workload IO.
N/A denotes unavailable metrics. CPU wall time includes construction/preload;
GPU cycles and HBF read/write traffic exclude setup. Third-party source and SHAs
are unchanged. Only the simulator C++ path and necessary memory tests were run.
'''
    if args.backend:
        notes += '\nOnly ' + args.backend + ' was rerun; the other rows retain their earlier measurements.\n'
    (out / 'summary.md').write_text(table + notes)
    print(table)
    return 0 if all(row['status'] == 'completed' for row in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
