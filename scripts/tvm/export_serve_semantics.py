"""Export algorithm-only serving TensorIR and report the actual search state."""
import argparse
import json
import sys
from pathlib import Path

from pinned_tvm import ROOT, load_tvm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT/'validation/results/serve-3m2f/semantic-ir')
    parser.add_argument('--search-root', type=Path, default=ROOT/'validation/results/serve-3m2f/study-a100-fp32/search')
    args = parser.parse_args()
    tvm = load_tvm()
    from serve_memory import annotate, KEY
    sys.path.insert(0, str(ROOT/'workloads/serve'))
    import semantics
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, function in [('fa2', semantics.fa2), ('gdn', semantics.gdn)]:
        tensors = function()
        mod = annotate(tvm, tvm.IRModule({'main': tvm.te.create_prim_func(tensors)}), 'CoLocated')
        tiers = json.loads(str(mod['main'].attrs[KEY]))
        (args.out_dir/(name+'.tir')).write_text(mod.script()+'\n')
        records[name] = dict(tensors={t.name: dict(shape=[int(n) for n in t.shape], dtype=str(t.dtype), tier=tiers[t.name])
                                      for t in tensors},
                             status='functional_input_not_optimized_schedule')
    (args.out_dir/'manifest.json').write_text(json.dumps(records, indent=2)+'\n')
    names = ['fa2', 'gdn_qkvz', 'gqa_qkvgate', 'output_proj', 'mlp_gate_up', 'mlp_down', 'gdn_full']
    completed = [name for name in names if (args.search_root/name/'results.json').exists()]
    best = {name: str(args.search_root/name/'best'/'scheduled.tir') for name in completed}
    status = dict(status='complete' if len(completed)==len(names) else 'in_progress',
                  workers=1, completed=completed, expected=names, best_ir=best,
                  note='The files in this directory are unscheduled semantic inputs. Best scheduled/native IR is separate.')
    (args.out_dir/'search-status.json').write_text(json.dumps(status, indent=2)+'\n')
    print('Exported semantic input IR; completed searches:', len(completed), '/', len(names))


if __name__ == '__main__':
    main()
