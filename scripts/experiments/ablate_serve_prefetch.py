#!/usr/bin/env python3
"""Controlled 4096x4096 output-projection prefetch ablation on both memory profiles."""
import argparse
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
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    tvm = load_tvm()
    import semantics
    from serve_memory import annotate
    from serve_rules import apply
    from serve_lower import lower_vector
    from auto_ir import trace_json
    from gemm import cycles, memory_statistics
    records = []
    for architecture_name, filename in [('5M', 'a100-pcie40-simple.yaml'),
                                        ('3M2F', 'a100-pcie40-3m2f-grade1-simple.yaml')]:
        architecture = ROOT/'configs/v0.1.0'/filename
        hardware = json.loads(subprocess.check_output([str(ROOT/'build/gtsim_serve_ir'),
                                                       str(architecture), '--describe-config'], text=True))
        for depth in (1, 2, 3):
            folder = args.out_dir/f'{architecture_name}-stages{depth}'
            folder.mkdir()
            mod = annotate(tvm, tvm.IRModule({'main': tvm.te.create_prim_func(semantics.projection(4096, 4096))}),
                           hardware['topology'])
            sch = tvm.s_tir.Schedule(mod, debug_mask='all', seed=1)
            choice = dict(output_tile=128, reduction_tile=128, warps=4, stages=depth, inline=0)
            apply(tvm, sch, 'projection', choice)
            serialized = tvm.ir.save_json(sch.mod)
            (folder/'scheduled.json').write_text(serialized)
            (folder/'scheduled.tir').write_text(sch.mod.script())
            (folder/'trace.json').write_text(json.dumps(trace_json(sch.trace), indent=2)+'\n')
            graph, evidence = lower_vector(tvm, tvm.ir.load_json(serialized), hardware)
            graph.write(folder/'kernel.gtsir')
            del graph
            command = [str(ROOT/'build/gtsim_serve_ir'), str(architecture), str((folder/'kernel.gtsir').resolve())]
            with (folder/'simulation.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('Ablation failed: '+str(folder))
            text = (folder/'simulation.log').read_text()
            count = cycles(text)
            stats = memory_statistics(text)
            device = 'HBF' if architecture_name == '3M2F' else 'HBM'
            if architecture_name == '3M2F' and stats['HBF']['read_bytes'] != 2*4096*4096:
                raise ValueError('Output projection did not read exactly its BF16 HBF weights')
            if stats['HBF']['write_bytes']:
                raise ValueError('Unexpected HBF writes')
            records.append(dict(architecture=architecture_name, stages=depth,
                                fixed=dict(N=4096, K=4096, output_tile=128, reduction_tile=128, warps=4),
                                cycles=count, us=count/hardware['core_frequency_mhz'],
                                memory_statistics=stats,
                                nominal_B_tile_window_bytes=32*128*128*2*depth,
                                window_definition='32 CTA * BK128 * BN128 * BF16 * stages; capacity proxy, not measured in-flight bytes',
                                learned_weight_bytes=2*4096*4096))
            print(architecture_name, 'depth', depth, records[-1]['us'], 'us', flush=True)
            (args.out_dir/'results.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
