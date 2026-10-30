#!/usr/bin/env python3
"""Publish measured search artifacts and consistently calibrated serving plots."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-root', type=Path, required=True)
    parser.add_argument('--baseline-operators', type=Path, required=True)
    parser.add_argument('--wall-audit', type=Path, default=ROOT / 'docs/results/serve-3m2f/wall-time-audit.json')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'docs/results/serve-3m2f/tvm')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = ['fa2', 'gdn_qkvz', 'gqa_qkvgate', 'output_proj', 'mlp_gate_up', 'mlp_down', 'gdn_full']
    hardware = json.loads(subprocess.check_output([str(ROOT / 'build/gtsim_serve_ir'), str(ROOT / 'configs/v0.1.0/a100-pcie40-3m2f-grade1-simple.yaml'), '--describe-config'], text=True))
    experiments = {}
    for name in names:
        folder = args.search_root / name
        result = json.loads((folder / 'results.json').read_text())
        if not result['best_replay_confirmed']:
            raise ValueError('Best replay not confirmed: ' + name)
        history = json.loads((folder / 'search-history.json').read_text())
        best = result['best']
        record = dict(cycles=best['cycles'], us=best['us'], choices=best['choices'],
                      successful=result['successful_candidates'], attempted=result['attempted_candidates'],
                      feedback_updates=result['feedback_updates'],
                      peak_memory_bytes=max(row.get('maximum_process_memory_bytes', 0) for row in history),
                      best_replay_confirmed=True, tensor_ir_sha256=best['tir_sha256'],
                      tile_ir_sha256=best['ir_sha256'], memory_statistics=best['memory_statistics'])
        feasible = [row for row in history if row['status'] == 'complete']
        first = feasible[0]
        device = 'HBM' if name == 'fa2' else 'HBF'
        read_bytes = best['memory_statistics'][device]['read_bytes']
        peak = hardware[device.lower() + '_read_bw_GBps']
        floor = read_bytes / (peak * 1000)
        if best['us'] + .001 < floor:
            raise ValueError('Result violates measured-traffic / peak-bandwidth lower bound: ' + name)
        record['bandwidth_check'] = dict(device=device, measured_read_bytes=read_bytes,
                                        peak_read_GBps=peak, read_time_lower_bound_us=floor,
                                        effective_read_GBps=read_bytes / (best['us'] * 1000),
                                        below_bound=False)
        record['first_feasible_us'] = first['us']
        record['search_speedup_over_first_feasible'] = first['cycles'] / best['cycles']
        record['fusion_coverage'] = {str(level): sum(row.get('choices', {}).get('inline') == level for row in feasible)
                                     for level in [0, 1, 2]} if name == 'gdn_full' else None
        experiments[name] = record
        target = args.out_dir / 'best-ir' / name
        target.mkdir(parents=True, exist_ok=True)
        for filename in ['scheduled.tir', 'scheduled.json', 'native.tir', 'native.json', 'trace.json',
                         'lowering.json', 'simulation.log', 'resources.txt']:
            if (folder / 'best' / filename).exists():
                shutil.copyfile(folder / 'best' / filename, target / filename)
                if filename == 'simulation.log':
                    published = target / filename
                    published.write_text('\n'.join(line.rstrip() for line in published.read_text().splitlines()) + '\n')
        (target / 'selection.json').write_text(json.dumps(record, indent=2) + '\n')
        portable = [{key: value for key, value in row.items() if key not in ('directory', 'command')}
                    for row in history]
        (args.out_dir / (name + '-history.json')).write_text(json.dumps(portable, indent=2) + '\n')
        for source in ['database_tuning_record.json', 'database_workload.json', 'cost-model.json', 'space.json',
                       'computation.tir']:
            if (folder / source).exists():
                shutil.copyfile(folder / source, args.out_dir / (name + '-' + source))
    shutil.copyfile(args.baseline_operators, args.out_dir / 'baseline-operators.json')
    if (args.baseline_operators.parent / 'run-spec.json').exists():
        shutil.copyfile(args.baseline_operators.parent / 'run-spec.json', args.out_dir / 'baseline-run-spec.json')
    baseline_rows = json.loads(args.baseline_operators.read_text())
    baseline = {arch: {row['operator']: row['us'] for row in baseline_rows if row['arch'] == arch}
                for arch in ('normal', 'hbf')}
    covered = {}
    for arch, operators in [*baseline.items(), ('optimized', {name: experiments[name]['us'] for name in names})]:
        common = operators['output_proj'] + operators['mlp_gate_up'] + operators['mlp_down']
        covered[arch] = dict(GQA_us=operators['gqa_qkvgate'] + operators['fa2'] + common,
                             GDN_us=operators['gdn_qkvz'] + common)
        covered[arch]['step_ms'] = (8 * covered[arch]['GQA_us'] + 24 * covered[arch]['GDN_us']) / 1000
    wall = json.loads(args.wall_audit.read_text())
    host_tpot = wall['client_measurements']['hostram']['tpot_ms']
    normal_tpot = wall['client_measurements']['normal']['tpot_ms']
    factor = normal_tpot / covered['normal']['step_ms']
    calibrated = {arch: dict(GQA_ms=factor * values['GQA_us'] / 1000,
                             GDN_ms=factor * values['GDN_us'] / 1000,
                             TPOT_ms=factor * values['step_ms']) for arch, values in covered.items()}
    host_gqa = wall['hostram_layer_wall_audit']['GQA']['wall_ms']['mean']
    host_gdn = wall['hostram_layer_wall_audit']['GDN']['wall_ms']['mean']
    tail = host_tpot - (8 * host_gqa + 24 * host_gdn)
    if tail < 0:
        raise ValueError('Negative serving tail; inspect timing attribution')
    calibrated['hostram'] = dict(
        GQA_ms=host_gqa + tail * covered['normal']['GQA_us'] / (1000 * covered['normal']['step_ms']),
        GDN_ms=host_gdn + tail * covered['normal']['GDN_us'] / (1000 * covered['normal']['step_ms']),
        TPOT_ms=host_tpot)
    for arch, values in calibrated.items():
        if abs(8 * values['GQA_ms'] + 24 * values['GDN_ms'] - values['TPOT_ms']) > 1e-8:
            raise ValueError('Amortized layer contributions do not close: ' + arch)
    result = dict(experiments=experiments, covered_components=covered,
                  calibration_factor=factor, calibrated_ms=calibrated,
                  hostram_outside_layers_ms=tail,
                  definitions=dict(
                      TPOT='Normal measured TPOT times matched simulated component ratio',
                      layers='Amortized TPOT contributions per logical GQA/GDN layer, including allocated serving overhead; not kernel latency',
                      hostram_layers='Measured complete layer entry/compute/wait window plus unassigned serving tail allocated by 5M component share',
                      uncertainty='Auxiliary kernels, LM head and host scheduling are not explicitly included in the common component subset; whole-step proportional scaling is assumed',
                      full_gdn='Complete functional GDN search is separately simulated and reported; excluded from the matched-component calibration to preserve baseline coverage'),
                  speedups=dict(hostram_over_optimized=host_tpot / calibrated['optimized']['TPOT_ms'],
                                unoptimized_3m2f_over_optimized=calibrated['hbf']['TPOT_ms'] / calibrated['optimized']['TPOT_ms']),
                  workers=1,
                  provenance=dict(
                      search_spec_sha256=hashlib.sha256((args.search_root / 'run-spec.json').read_bytes()).hexdigest(),
                      baseline_operators_sha256=hashlib.sha256(args.baseline_operators.read_bytes()).hexdigest(),
                      wall_audit_sha256=hashlib.sha256(args.wall_audit.read_bytes()).hexdigest()))
    (args.out_dir / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    shutil.copyfile(args.search_root / 'run-spec.json', args.out_dir / 'run-spec.json')
    print(json.dumps(dict(calibrated_ms=calibrated, speedups=result['speedups']), indent=2))


if __name__ == '__main__':
    main()
