#!/usr/bin/env python3
"""Native hierarchical GDN search seeded by independently measured projection winners."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from pinned_tvm import ROOT, load_tvm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=12)
    args = parser.parse_args()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'TVM_NUM_THREADS'):
        os.environ[name] = '1'
    tvm = load_tvm()
    sys.path.insert(0, str(ROOT/'workloads/serve'))
    import semantics
    from serve_memory import annotate
    from serve_search import search
    architecture = ROOT/'configs/v0.1.0/a100-pcie40-3m2f-grade1-simple.yaml'
    hardware = json.loads(subprocess.check_output([str(ROOT/'build/gtsim_serve_ir'), str(architecture),
                                                   '--describe-config'], text=True))
    names = dict(qkvz='gdn_qkvz', out_proj='output_proj', gate_up='mlp_gate_up', down='mlp_down')
    seed = dict(projections={name: json.loads((args.search_root/experiment/'results.json').read_text())['best']['choices']
                             for name, experiment in names.items()},
                auxiliary=json.loads((args.search_root/'gdn_full/results.json').read_text())['best']['choices'])
    mod = annotate(tvm, tvm.IRModule({'main': tvm.te.create_prim_func(semantics.gdn())}), hardware['topology'])
    mod['main'] = mod['main'].with_attr('gtsim.hierarchical_seed', json.dumps(seed, sort_keys=True))
    sources = [*sorted((ROOT/'scripts/tvm').glob('serve_*.py')), Path(__file__).resolve(), ROOT/'workloads/serve/semantics.py', architecture]
    provenance = dict(source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                      binary_sha256=hashlib.sha256((ROOT/'build/gtsim_serve_ir').read_bytes()).hexdigest(),
                      seed=29, trials=args.trials, workers=1)
    result = search(tvm, mod, 'gdn_split', hardware, architecture, ROOT/'build', args.out_dir.resolve(), args.trials, 29)
    (args.out_dir/'run-spec.json').write_text(json.dumps(provenance, indent=2)+'\n')
    record = dict(seed=seed, seed_result_sha256={name: hashlib.sha256((args.search_root/experiment/'results.json').read_bytes()).hexdigest()
                                               for name, experiment in {**names, 'auxiliary': 'gdn_full'}.items()},
                  hardware=hardware, scope='Full connected GDN; native rfactor Split-K and fusion search with measured projection-layout seeds')
    (args.out_dir/'hierarchy.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(best_us=result['best']['us'], choices=result['best']['choices'],
                          successful=result['successful_candidates']), indent=2))


if __name__ == '__main__':
    main()
