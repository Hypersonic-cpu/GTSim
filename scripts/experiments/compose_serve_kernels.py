#!/usr/bin/env python3
"""Compose independently searched projection layouts into the searched GDN fusion family.

This is a separately measured composition of native MetaSchedule winners, not a
claim that the composed trace is a single MetaSchedule database record.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/tvm'))
sys.path.insert(0, str(ROOT/'workloads/serve'))
from pinned_tvm import load_tvm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    tvm = load_tvm()
    import semantics
    from auto_ir import trace_json
    from serve_memory import annotate
    from serve_rules import INLINE
    from serve_lower import lower_vector
    from gemm import cycles, memory_statistics
    architecture = ROOT/'configs/v0.1.0/a100-pcie40-3m2f-grade1-simple.yaml'
    hardware = json.loads(subprocess.check_output([str(ROOT/'build/gtsim_serve_ir'), str(architecture), '--describe-config'], text=True))
    names = ['gdn_full', 'gdn_qkvz', 'output_proj', 'mlp_gate_up', 'mlp_down']
    inputs = {name: json.loads((args.search_root/name/'results.json').read_text()) for name in names}
    base = inputs['gdn_full']['best']['choices']
    projection = dict(qkvz='gdn_qkvz', out_proj='output_proj', gate_up='mlp_gate_up', down='mlp_down')
    original = annotate(tvm, tvm.IRModule({'main': tvm.te.create_prim_func(semantics.gdn())}), hardware['topology'])
    sch = tvm.s_tir.Schedule(original, debug_mask='all', seed=1)
    sch.annotate(sch.get_sblock('root'), 'gtsim.composed_native_winners', 1)
    for name in INLINE.get(base['inline'], ()):
        sch.compute_inline(sch.get_sblock(name))
    selected = {}
    for handle in sch.get_child_blocks(sch.get_sblock('root')):
        block = sch.get(handle)
        group = next((key for key in projection if block.name_hint in (key, key+'_acc')), None)
        choice = inputs[projection[group]]['best']['choices'] if group else base
        selected[block.name_hint] = choice
        # Force the previously measured factors into an ordinary replayable
        # TVM trace; all split/bind transformations are represented in the IR.
        factors = {key: sch.sample_categorical([choice[key]], [1.0], decision=0)
                   for key in ('output_tile', 'reduction_tile', 'warps', 'stages')}
        loops = list(sch.get_loops(handle))
        spatial = sum(int(var.iter_type) == 0 for var in block.iter_vars)
        reduction = [var for var in block.iter_vars if int(var.iter_type) == 2]
        output = sch.fuse(*loops[:spatial])
        outer, inner = sch.split(output, factors=[None, factors['output_tile']])
        warp, vector, lane = sch.split(inner, factors=[factors['warps'], None, 32])
        if reduction:
            red_outer, red_inner = sch.split(loops[-1], factors=[None, factors['reduction_tile']])
            sch.reorder(outer, red_outer, warp, vector, lane, red_inner)
            sch.annotate(red_inner, 'gtsim.reduction_tile', 1)
        sch.bind(outer, 'blockIdx.x')
        sch.bind(warp, 'threadIdx.y')
        sch.bind(lane, 'threadIdx.x')
        sch.annotate(vector, 'gtsim.output_vector', 1)
        sch.annotate(handle, 'gtsim.register_stages', factors['stages'])
    replay = tvm.s_tir.Schedule(original, debug_mask='all')
    tvm.s_tir.schedule.Trace.apply_json_to_schedule(trace_json(sch.trace), replay)
    tvm.ir.assert_structural_equal(sch.mod, replay.mod)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    serialized = tvm.ir.save_json(replay.mod)
    (args.out_dir/'scheduled.json').write_text(serialized)
    (args.out_dir/'scheduled.tir').write_text(replay.mod.script())
    (args.out_dir/'trace.json').write_text(json.dumps(trace_json(sch.trace), indent=2)+'\n')
    graph, evidence = lower_vector(tvm, tvm.ir.load_json(serialized), hardware)
    graph.write(args.out_dir/'kernel.gtsir')
    (args.out_dir/'lowering.json').write_text(json.dumps(evidence, indent=2)+'\n')
    del graph
    command = [str(ROOT/'build/gtsim_serve_ir'), str(architecture), str((args.out_dir/'kernel.gtsir').resolve())]
    with (args.out_dir/'simulation.log').open('w') as output, (args.out_dir/'resources.txt').open('w') as resource:
        result = subprocess.run(['/usr/bin/time', '-l', *command], stdout=output, stderr=resource)
    if result.returncode:
        raise RuntimeError('Composed GDN simulation failed')
    log = (args.out_dir/'simulation.log').read_text()
    count = cycles(log)
    stats = memory_statistics(log)
    if stats['HBF']['write_bytes']:
        raise RuntimeError('State or activation output reached HBF')
    record = dict(kind='composition_of_independent_native_metaschedule_winners', cycles=count,
                  us=count/hardware['core_frequency_mhz'], base_full_search_cycles=inputs['gdn_full']['best']['cycles'],
                  full_layer_speedup=inputs['gdn_full']['best']['cycles']/count,
                  per_stage_choices=selected, memory_statistics=stats,
                  scheduled_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
                  scope='Complete connected GDN functional graph with explicit state and activation tensors',
                  calibration='Reported separately; matched-component TPOT still uses isolated operator measurements')
    (args.out_dir/'results.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({key: record[key] for key in ('cycles', 'us', 'base_full_search_cycles', 'full_layer_speedup')}, indent=2))


if __name__ == '__main__':
    main()
