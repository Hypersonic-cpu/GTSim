#!/usr/bin/env python3
"""Replay a recorded native schedule, check its values, and simulate it again."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pinned_tvm import ROOT, load_tvm


def replay(search_dir, build_dir, out_dir, candidate=None):
    tvm = load_tvm()
    from gemm import ARCH, MEMORY, cycles, hardware_config, run_simulation, source_hashes
    from hopper_compiler import lower_hopper
    from tier_memory import lower_module

    search_dir, out_dir = Path(search_dir), Path(out_dir)
    results = json.loads((search_dir / 'results.json').read_text())
    memory=Path(results.get('memory_config',MEMORY))
    history = json.loads((search_dir / 'search-history.json').read_text())
    row = results['best'] if candidate is None else next(
        row for row in history if row['candidate'] == candidate)
    if row['status'] != 'complete':
        raise ValueError('the selected candidate did not complete simulation')
    if results['source_sha256'] != source_hashes():
        raise ValueError('frontend sources differ from the recorded search')
    for path, key in ((ARCH, 'architecture_sha256'), (memory, 'memory_sha256')):
        if hashlib.sha256(path.read_bytes()).hexdigest() != results[key]:
            raise ValueError(f'simulation configuration changed: {path}')
    directory = search_dir / f"candidate-{row['candidate']:04d}"
    computation = tvm.ir.load_json((search_dir / 'computation.json').read_text())
    schedule = tvm.s_tir.Schedule(computation, debug_mask='all', seed=1)
    # Tensor intrinsics are registered compiler extensions, not workload code.
    from hopper_intrinsics import register_wgmma
    register_wgmma(tvm)
    tvm.s_tir.schedule.Trace.apply_json_to_schedule(
        json.loads((directory / 'trace.json').read_text()), schedule)
    tvm.ir.assert_structural_equal(
        schedule.mod, tvm.ir.load_json((directory / 'scheduled.json').read_text()))
    hardware = hardware_config(build_dir,memory=memory)
    if hardware != results['hardware']:
        raise ValueError('hardware description differs from the recorded search')
    native, _, _ = lower_hopper(tvm, schedule.mod, hardware)
    tvm.ir.assert_structural_equal(
        native, tvm.ir.load_json((directory / 'native.json').read_text()))
    m, n, k = results['dimensions']
    rng = np.random.default_rng(0)
    a = rng.integers(-2, 3, (m, k)).astype('float16')
    b = rng.integers(-2, 3, (k, n)).astype('float16')
    ir, evidence = lower_module(tvm, native, hardware, inputs=(a, b))
    golden = (a.astype('int32') @ b.astype('int32')).astype('float16')
    actual = evidence['numerical_output']
    if not np.isfinite(actual).all():
        raise ValueError('nonfinite interpreted output')
    np.testing.assert_array_equal(actual, golden)
    if hashlib.sha256(ir.text().encode()).hexdigest() != row['ir_sha256']:
        raise ValueError('replayed Tile IR differs from the search artifact')
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'kernel.gtsir'
    ir.write(path)
    command = run_simulation(build_dir, ARCH, memory, path, out_dir / 'simulation.log')
    count = cycles((out_dir / 'simulation.log').read_text())
    if count != row['cycles']:
        raise ValueError('replayed simulation cycles differ')
    result = dict(candidate=row['candidate'], dimensions=[m, n, k], cycles=count,
                  max_absolute_error=float(np.max(np.abs(actual.astype('float32') - golden))),
                  mismatched_elements=int(np.count_nonzero(actual != golden)), input_seed=0,
                  oracle='exact int32 GEMM for small integer FP16 inputs, then FP16 cast',
                  scope='software interpretation of lowered instructions; no CUDA execution',
                  command=command)
    (out_dir / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-dir', type=Path, required=True)
    parser.add_argument('--candidate', type=int)
    parser.add_argument('--all', action='store_true', help='verify every completed search candidate')
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build')
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    if args.all:
        if args.candidate is not None:
            parser.error('--all and --candidate are mutually exclusive')
        history = json.loads((args.search_dir / 'search-history.json').read_text())
        out = args.out_dir or args.search_dir / 'replay-all'
        reports = []
        for row in history:
            if row['status'] == 'complete':
                report = replay(args.search_dir, args.build_dir,
                                out / f"candidate-{row['candidate']:04d}", row['candidate'])
                reports.append(report)
                print(f"PASS candidate {row['candidate']}: {report['cycles']} cycles, "
                      f"max error {report['max_absolute_error']}", flush=True)
        (out / 'verification.json').write_text(json.dumps(reports, indent=2) + '\n')
        return
    out = args.out_dir or args.search_dir / ('replay-best' if args.candidate is None
                                           else f'replay-{args.candidate:04d}')
    print(json.dumps(replay(args.search_dir, args.build_dir, out, args.candidate), indent=2))


if __name__ == '__main__':
    main()
