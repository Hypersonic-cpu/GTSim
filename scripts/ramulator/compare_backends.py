#!/usr/bin/env python3
"""Run a restricted measured H100 subset with at most two simulator processes.

GEMM runs small to large; FA3 runs large to small. Logs and per-run provenance
are retained before deriving a CSV and a Markdown accuracy/runtime table.
"""
import argparse
import concurrent.futures
import csv
import json
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import time


CASES = {
    'gemm': [(512, 512, 512), (512, 512, 1024), (1024, 1024, 512), (1024, 1024, 1024)],
    'fa3': [(1, 8, 8, n, n) for n in (1536, 1024, 512)],
}
FIELDS = {'gemm': ['M', 'N', 'K'], 'fa3': ['batch', 'heads_q', 'heads_kv', 'seq_q', 'seq_kv']}


def run_case(root, executable, output, workload, shape, backend):
    case = workload + '_' + '_'.join(map(str, shape))
    command = [str(executable), str(root / 'configs/v0.1.0/h100-sxm.yaml')]
    if backend == 'ramulator':
        command += ['--memory-config', str(root / 'configs/ramulator/h100-sxm-ramulator.yaml')]
    command += list(map(str, shape))
    start = time.perf_counter()
    usage = None
    with (output / f'{case}_{backend}.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        if hasattr(os, 'wait4'):
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        else:
            process.wait()
    elapsed = time.perf_counter() - start
    log = (output / f'{case}_{backend}.log').read_text()
    cycles = re.search(r'^Total cycles:\s*(\d+)', log, re.M)
    finished = re.search(r'^Finished:\s*(\d+)/(\d+)', log, re.M)
    completed = (process.returncode == 0 and '✅ Simulation completed' in log and cycles and
                 finished and finished[1] == finished[2])
    conservation = re.search(r'Ramulator HBM3 parents: (\d+)/(\d+) accepted/completed; '
                             r'bursts: (\d+)/(\d+); bytes: (\d+); admission retries: (\d+); '
                             r'DRAM ticks: (\d+); channels used: (\d+)/(\d+)', log)
    if backend == 'ramulator':
        completed = (completed and 'Memory drained at cycle:' in log and conservation and
                     conservation[1] == conservation[2] and conservation[3] == conservation[4])
    record = dict(case=case, workload=workload, args=list(shape), command=command,
                  executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                  returncode=process.returncode, completed=bool(completed),
                  cycles=int(cycles[1]) if cycles else None, wall_seconds=elapsed,
                  peak_rss_bytes=(usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
                                  if usage else None),
                  user_seconds=usage.ru_utime if usage else None,
                  system_seconds=usage.ru_stime if usage else None)
    drained = re.search(r'^Memory drained at cycle:\s*(\d+)', log, re.M)
    record['memory_drained_cycle'] = int(drained[1]) if drained else None
    if conservation:
        record['ramulator'] = dict(zip(
            ['accepted_parents', 'completed_parents', 'submitted_bursts', 'completed_bursts',
             'transaction_bytes', 'admission_retries', 'dram_ticks', 'channels_used', 'channels'],
            map(int, conservation.groups())))
    controller = re.search(r'Ramulator controllers: read/write accepted (\d+)/(\d+), served '
                           r'(\d+)/(\d+); row hit/miss/conflict (\d+)/(\d+)/(\d+); '
                           r'channel burst min/max (\d+)/(\d+)', log)
    if controller:
        record['controller'] = dict(zip(
            ['reads', 'writes', 'served_reads', 'served_writes', 'row_hits', 'row_misses',
             'row_conflicts', 'min_channel_bursts', 'max_channel_bursts'],
            map(int, controller.groups())))
        counters = record['controller']
        bursts = record['ramulator']['completed_bursts']
        record['completed'] &= (counters['reads'] + counters['writes'] == bursts and
                                counters['served_reads'] + counters['served_writes'] == bursts)
    elif backend == 'ramulator':
        record['completed'] = False
    (output / f'{case}_{backend}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f"{case} {backend}: completed={record['completed']} cycles={record['cycles']} "
          f"wall={elapsed:.3f}s", flush=True)
    return record


def report(root, output):
    measured = {}
    for workload, fields in FIELDS.items():
        with (root / f'validation/data/h100_{workload}.csv').open() as file:
            for row in csv.DictReader(file):
                key = workload + '_' + '_'.join(row[field] for field in fields)
                measured[key] = int(row['measured_cycles'])
    rows = []
    incomplete = []
    for workload, shapes in CASES.items():
        for shape in shapes:
            case = workload + '_' + '_'.join(map(str, shape))
            pair = {}
            for backend in ('naive', 'ramulator'):
                path = output / f'{case}_{backend}.json'
                pair[backend] = json.loads(path.read_text()) if path.exists() else None
            if not all(run and run['completed'] for run in pair.values()):
                incomplete.append({case: pair})
                continue
            naive, ram = pair['naive'], pair['ramulator']
            target = measured[case]
            old_error = (naive['cycles'] / target - 1) * 100
            new_error = (ram['cycles'] / target - 1) * 100
            rows.append(dict(case=case, measured_cycles=target, naive_cycles=naive['cycles'],
                             ramulator_cycles=ram['cycles'], naive_error_percent=old_error,
                             ramulator_error_percent=new_error,
                             absolute_error_change_pp=abs(new_error) - abs(old_error),
                             naive_wall_seconds=naive['wall_seconds'],
                             ramulator_wall_seconds=ram['wall_seconds'],
                             wall_ratio=ram['wall_seconds'] / naive['wall_seconds'],
                             naive_peak_rss_bytes=naive['peak_rss_bytes'],
                             ramulator_peak_rss_bytes=ram['peak_rss_bytes']))
    if rows:
        with (output / 'comparison.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['| Case | Measured | Naive | Ramulator | Naive error | Ram error | Δ abs error (pp) | Naive wall (s) | Ram wall (s) | Ram/naive wall |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| {case} | {measured_cycles} | {naive_cycles} | {ramulator_cycles} | '
                     '{naive_error_percent:+.2f}% | {ramulator_error_percent:+.2f}% | '
                     '{absolute_error_change_pp:+.2f} | {naive_wall_seconds:.3f} | '
                     '{ramulator_wall_seconds:.3f} | {wall_ratio:.2f}× |'.format(**row))
    summary = {}
    for workload in ('gemm', 'fa3', 'all'):
        subset = [row for row in rows if workload == 'all' or row['case'].startswith(workload)]
        if subset:
            summary[workload] = {
                'paired_cases': len(subset),
                **{backend + '_mape_percent': sum(abs(row[backend + '_error_percent']) for row in subset) / len(subset)
                   for backend in ('naive', 'ramulator')},
            }
    lines += ['', 'MAPE over the same completed cases:', '```json', json.dumps(summary, indent=2), '```',
              '', f'Failed or unrun selected cases: {len(incomplete)}. See summary.json for statuses.']
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    (output / 'summary.json').write_text(json.dumps(dict(mape=summary, incomplete=incomplete), indent=2) + '\n')


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=root / 'build')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--naive-build-dir', type=Path,
                        help='optional frozen original executables for naive runs')
    parser.add_argument('--phase', choices=['naive', 'ramulator', 'all', 'report'], default='all')
    args = parser.parse_args()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.phase != 'report':
        def git(*command):
            return subprocess.check_output(['git', *command], cwd=root, text=True).strip()
        profiles = ['configs/v0.1.0/h100-sxm.yaml', 'configs/ramulator/h100-sxm-ramulator.yaml',
                    'configs/ramulator/h100_hbm3.yaml']
        provenance = dict(branch=git('branch', '--show-current'), head=git('rev-parse', 'HEAD'),
                          submodule=git('submodule', 'status', '3rdparty/ramulator2'),
                          build_dir=str(args.build_dir.resolve()),
                          naive_build_dir=str(args.naive_build_dir.resolve()) if args.naive_build_dir else None,
                          configuration_sha256={file: hashlib.sha256((root / file).read_bytes()).hexdigest()
                                                for file in profiles})
        source_files = git('ls-files', '--cached', '--others', '--exclude-standard').splitlines()
        provenance['source_sha256'] = {
            file: hashlib.sha256((root / file).read_bytes()).hexdigest()
            for file in sorted(set(source_files))
            if (root / file).is_file() and
            (file.startswith(('include/', 'src/', 'workloads/', 'cmake/', 'scripts/ramulator/')) or
             file == 'CMakeLists.txt')
        }
        cache = args.build_dir / 'CMakeCache.txt'
        if cache.is_file():
            provenance['build_settings'] = [line for line in cache.read_text().splitlines()
                if line.startswith(('CMAKE_CXX_COMPILER:', 'CMAKE_BUILD_TYPE:', 'CMAKE_CXX_FLAGS:',
                                    'GTSIM_ENABLE_RAMULATOR2:', 'FETCHCONTENT_SOURCE_DIR_'))]
        (output / f'provenance_{args.phase}.json').write_text(json.dumps(provenance, indent=2) + '\n')
        (output / f'source_{args.phase}.diff').write_text(git('diff', '--binary') + '\n')
    backends = ['naive', 'ramulator'] if args.phase == 'all' else [args.phase]
    def sweep(workload):
        runs = []
        for shape in CASES[workload]:
            for backend in backends:
                build = args.naive_build_dir if backend == 'naive' and args.naive_build_dir else args.build_dir
                runs.append(run_case(root, (build / f'gtsim_{workload}').resolve(), output, workload, shape, backend))
        return runs
    runs = []
    if args.phase != 'report':
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            runs = [run for family in executor.map(sweep, CASES) for run in family]
    report(root, output)
    if any(not run['completed'] for run in runs):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
