#!/usr/bin/env python3
"""Replay saved serving winners under matched Simple and Bank/Plane HBF.

No TVM import, compilation or tuning is performed. The small b/a projection is
isolated from the saved full-GDN graph at its existing completed stage boundary.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    ('fa2', 'GQA FA2 main+combine', None),
    ('gqa_qkvgate', 'GQA QKV+gate', [1, 10240, 4096]),
    ('gdn_qkvz', 'GDN QKVZ', [1, 12288, 4096]),
    ('gdn_ba', 'GDN b/a', [1, 64, 4096]),
    ('output_proj', 'GQA/GDN output projection', [1, 4096, 4096]),
    ('mlp_gate_up', 'GQA/GDN MLP gate/up', [1, 24576, 4096]),
    ('mlp_down', 'GQA/GDN MLP down', [1, 4096, 12288]),
    ('gdn_prediction', 'GDN recurrent prediction', [1, 128, 128]),
    ('gdn_readout', 'GDN recurrent readout+cast', [1, 128, 128]),
    ('gdn_state_update', 'GDN rank-1 state update', [128, 128, 1]),
]
EXTRACTED = dict(gdn_ba=['ba_acc', 'ba'], gdn_prediction=['prediction'],
                 gdn_readout=['attn_acc', 'attn'], gdn_state_update=['NextState'])
STATE = {'gdn_prediction', 'gdn_readout', 'gdn_state_update'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def isolate_stages(source, target, selected_stages):
    tensors, ctas, operations = [], {}, []
    with source.open() as handle:
        for line in handle:
            if line.startswith('tensor '):
                tensors.append(line)
            elif line.startswith('cta '):
                ctas[int(re.search(r'id=(\d+)', line)[1])] = line
            elif line.startswith('%'):
                stage = re.search(r'name=(\w+)__', line)[1]
                if stage in selected_stages:
                    operations.append(line)
    ids = {int(re.match(r'%(\d+)', line)[1]): i for i, line in enumerate(operations)}
    used = sorted({int(re.search(r'cta=(\d+)', line)[1]) for line in operations})
    cta_ids = {old: i for i, old in enumerate(used)}
    assert operations and len(used) >= len(selected_stages)
    removed = 0
    with target.open('w') as handle:
        handle.write('gtsim.tile_ir 2\n')
        handle.writelines(tensors)  # retain original allocation offsets
        for old in used:
            handle.write(re.sub(r'^cta id=\d+', 'cta id=' + str(cta_ids[old]), ctas[old]))
        for old, line in zip(ids, operations):
            line = re.sub(r'^%\d+', '%' + str(ids[old]), line)
            line = re.sub(r'cta=(\d+)', lambda m: 'cta=' + str(cta_ids[int(m[1])]), line)
            for field in ('after_issue', 'after_complete'):
                match = re.search(field + r'=\[([^]]*)\]', line)
                parents = [int(x) for x in match[1].split(',') if x]
                removed += sum(parent not in ids for parent in parents)
                line = line[:match.start()] + field + '=[' + ','.join(str(ids[p]) for p in parents if p in ids) + ']' + line[match.end():]
            handle.write(line)
    return dict(source_sha256=sha(source), retained_stages=selected_stages, operations=len(ids),
                ctas=len(used), external_completed_dependencies_removed=removed,
                scope='Unchanged selected stages from full-GDN winner; external activation/state inputs are resident HBM')


def parse(log):
    text = log.read_text()
    cycles = int(re.search(r'^Total cycles: (\d+)$', text, re.M)[1])
    drain = int(re.search(r'^Memory drained at cycle: (\d+)$', text, re.M)[1])
    memory = {}
    for match in re.finditer(r'^  (HBM|HBF) requests: (\d+) \(read (\d+), write (\d+)\), bytes: read (\d+), write (\d+), average latency: ([\d.]+) cycles', text, re.M):
        memory[match[1]] = dict(requests=int(match[2]), reads=int(match[3]), writes=int(match[4]),
                                read_bytes=int(match[5]), write_bytes=int(match[6]), average_latency_cycles=float(match[7]))
    detailed = re.search(r'^HBF detailed JSON: (.*)$', text, re.M)
    phases = {m[1]: dict(first=int(m[2]), complete=int(m[3]), cycles=int(m[3])-int(m[2])+1)
              for m in re.finditer(r'^Stage timing: (\S+) first (\d+) complete (\d+)$', text, re.M)}
    return dict(cycles=cycles, drained_cycles=drain, memory=memory,
                hbfine=json.loads(detailed[1]) if detailed else None, stages=phases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel-root', type=Path, default=ROOT/'validation/results/serve-3m2f/study-a100-fp32/search')
    parser.add_argument('--build-dir', type=Path, default=ROOT/'build-backends')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--only-state', action='store_true', help='Replay only recurrent prediction/readout/rank-1 updates')
    args = parser.parse_args()
    cases = [case for case in CASES if not args.only_state or case[0] in STATE]
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    binary = args.build_dir.resolve()/'gtsim_serve_ir'
    configurations = {kind: ROOT/'configs/v0.1.0'/('a100-pcie40-3m2f-grade1-'+suffix+'.yaml')
                      for kind, suffix in [('simple', 'simple-matched'), ('hbfine', 'hbfine')]}
    hardware = {kind: json.loads(subprocess.check_output([str(binary), str(config), '--describe-config'], text=True))
                for kind, config in configurations.items()}
    a, b = hardware['simple'].copy(), hardware['hbfine'].copy()
    assert a.pop('hbf_backend') == 'naive' and b.pop('hbf_backend') == 'bank_plane' and a == b
    assert a['hbm_backend'] == 'naive' and a['hbf_dies_per_stack'] == 8
    inputs = {}
    for name, label, shape in cases:
        key = 'gdn_full' if name in EXTRACTED else name
        result = json.loads((args.kernel_root/key/'results.json').read_text())
        selected = result['best']
        source = args.kernel_root/key/f"candidate-{selected['candidate']:04d}"/'kernel.gtsir'
        assert sha(source) == selected['ir_sha256'], name
        provenance = dict(selected_candidate=selected['candidate'], selection=selected['choices'],
                          selection_result_sha256=sha(args.kernel_root/key/'results.json'))
        if name in EXTRACTED:
            target = args.out_dir/(name+'-winner.gtsir')
            provenance['extraction'] = isolate_stages(source, target, EXTRACTED[name])
            source = target
        inputs[name] = dict(path=str(source.resolve()), sha256=sha(source), label=label, MNK=shape,
                            expected_weight_bytes=2*shape[1]*shape[2] if shape and name not in STATE else 0,
                            batch=32 if name in STATE else 1,
                            original_best_cycles=selected['cycles'] if name not in EXTRACTED else None,
                            provenance=provenance)
    sources = [Path(__file__).resolve(), *sorted((ROOT/'src').glob('*.cpp')), *sorted((ROOT/'include').glob('*.h')),
               *sorted((ROOT/'3rdparty/hbfine/src').glob('*.cpp')), *sorted((ROOT/'3rdparty/hbfine/include/hbfine').glob('*.hpp')),
               ROOT/'workloads/serve/serve_ir.cpp']
    spec = dict(workers=1, tuning=False, binary_sha256=sha(binary),
                source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
                configs={kind: dict(path=str(p.relative_to(ROOT)), sha256=sha(p), resolved=hardware[kind]) for kind, p in configurations.items()},
                inputs=inputs, measurement='DAG completion cycles / 1410 MHz; memory drain recorded separately')
    write(args.out_dir/'run-spec.json', spec)
    rows = []
    for name, label, shape in cases:
        for kind, config in configurations.items():
            folder = args.out_dir/(name+'-'+kind)
            folder.mkdir()
            command = [str(binary), str(config), inputs[name]['path']]
            timed = ['/usr/bin/time', '-l', '-o', str(folder/'resources.txt'), *command]
            started = time.monotonic()
            with (folder/'simulation.log').open('w') as log:
                result = subprocess.run(timed, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('Simulation failed: '+str(folder))
            row = dict(name=name, label=label, MNK=shape, backend=kind, command=command,
                       simulation_wall_seconds=time.monotonic()-started, **parse(folder/'simulation.log'))
            row['batch'] = inputs[name]['batch']
            row['us'] = row['cycles']/hardware[kind]['core_frequency_mhz']
            row['drained_us'] = row['drained_cycles']/hardware[kind]['core_frequency_mhz']
            resource_text = (folder/'resources.txt').read_text()
            row['maximum_process_memory_bytes'] = max(map(int,re.findall(r'(\d+)\s+(?:maximum resident set size|peak memory footprint)',resource_text)))
            expected = inputs[name]['expected_weight_bytes']
            assert row['memory']['HBF']['read_bytes'] == expected, (name, kind, row['memory'])
            assert row['memory']['HBF']['write_bytes'] == 0
            if kind == 'hbfine':
                stats = row['hbfine']
                assert stats is not None and stats['accepted_requests'] == stats['completed_requests']
                assert stats['host_read_bytes'] == expected
                assert stats['host_write_bytes'] == stats['programs'] == stats['erases'] == stats['failed_requests'] == 0
                assert len(stats['plane_reads']) == len(stats['bank_reads']) == 512
                row['read_amplification'] = stats['sensed_bytes']/expected if expected else 0
                row['achieved_HBF_GBps'] = expected/(row['us']*1000)
            rows.append(row)
            write(args.out_dir/'results.json', rows)
            print(name, kind, f"{row['us']:.6f} us", flush=True)
    fa = [row for row in rows if row['name'] == 'fa2']
    if fa:
        assert fa[0]['cycles'] == fa[1]['cycles'], 'FA2 does not use HBF; changing HBF changed its timing'
    for name, _, _ in cases:
        pair = [row for row in rows if row['name'] == name]
        if inputs[name]['expected_weight_bytes'] == 0:
            assert pair[0]['cycles'] == pair[1]['cycles'], 'HBM-only control changed: '+name
        assert pair[0]['memory'] == pair[1]['memory'] or all(pair[0]['memory'][tier][key] == pair[1]['memory'][tier][key]
            for tier in ('HBM','HBF') for key in ('read_bytes','write_bytes'))
    with (args.out_dir/'comparison.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['operator','M','N','K','simple_us','hbfine_us','hbfine_over_simple','HBF_host_bytes','HBF_sensed_bytes','read_amplification'])
        for name, label, shape in cases:
            pair = [row for row in rows if row['name'] == name]
            writer.writerow([name,*(shape or ['','','']),pair[0]['us'],pair[1]['us'],pair[1]['cycles']/pair[0]['cycles'],
                             pair[1]['memory']['HBF']['read_bytes'],pair[1]['hbfine']['sensed_bytes'],pair[1]['read_amplification']])
    write(args.out_dir/'status.json', dict(status='complete',simulations=len(rows),workers=1,tuning=False,FA2_same_cycles=True if fa else None))


if __name__ == '__main__':
    main()
