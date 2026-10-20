#!/usr/bin/env python3
"""Optimize computation-only TE FP16 GEMM using native TVM search and GTSim."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess

from pinned_tvm import ROOT, TVM_COMMIT, load_tvm


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload',type=Path,default=ROOT/'workloads/gemm/te_gemm.py')
    parser.add_argument('--m',type=int,default=512);parser.add_argument('--n',type=int,default=512);parser.add_argument('--k',type=int,default=512)
    parser.add_argument('--trials',type=int,default=32);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--jobs',type=int,default=4);parser.add_argument('--build-dir',type=Path,default=ROOT/'build')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'validation/results/tvm/auto-search')
    args=parser.parse_args()
    if min(args.m,args.n,args.k,args.jobs,args.trials)<=0 or args.trials<2 or args.seed<0:
        parser.error('positive dimensions/jobs, at least two trials, and a nonnegative seed are required')
    tvm=load_tvm()
    from gemm import ARCH, MEMORY, cycles, generate, hardware_config, run_simulation, source_hashes
    from h100_schedule import GemmSchedule
    from auto_search import search,write_json
    spec=importlib.util.spec_from_file_location('gtsim_te_workload',args.workload.resolve())
    workload=importlib.util.module_from_spec(spec);spec.loader.exec_module(workload)
    tensors=workload.workload(args.m,args.n,args.k)
    mod=tvm.IRModule({'main':tvm.te.create_prim_func(tensors)})
    hardware=hardware_config(args.build_dir)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'workload.py').write_bytes(args.workload.read_bytes())
    best,history,updates,database=search(mod,(args.m,args.n,args.k),hardware,args.build_dir,args.out_dir,args.trials,args.seed,args.jobs)
    baseline_dir=args.out_dir/'manual-baseline'
    path,_=generate(tvm,(args.m,args.n,args.k),GemmSchedule(),hardware,baseline_dir)
    baseline=[]
    for repeat in range(2):
        log=baseline_dir/f'simulation-{repeat+1}.log'
        run_simulation(args.build_dir,ARCH,MEMORY,path,log);baseline.append(cycles(log.read_text()))
    repeated=Path(best['directory'])/'simulation-confirm.log'
    run_simulation(args.build_dir,ARCH,MEMORY,Path(best['directory'])/'kernel.gtsir',repeated)
    if cycles(repeated.read_text())!=best['cycles'] or len(set(baseline))!=1:
        raise RuntimeError('best/baseline cycles were not reproducible')
    metadata=dict(tvm_commit=TVM_COMMIT,gtsim_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        python=platform.python_version(),tvm_version=tvm.__version__,seed=args.seed,trials=args.trials,
        dimensions=[args.m,args.n,args.k],hardware=hardware,source_sha256=source_hashes(),
        workload_sha256=hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        compiler_build=json.loads((ROOT/'build-tvm/gtsim-tvm-build.json').read_text()),
        architecture_sha256=hashlib.sha256(ARCH.read_bytes()).hexdigest(),memory_sha256=hashlib.sha256(MEMORY.read_bytes()).hexdigest(),
        baseline_cycles=baseline,best=best,speedup=baseline[0]/best['cycles'],feedback_updates=len(updates),
        successful_candidates=sum(row['status']=='complete' for row in history))
    write_json(args.out_dir/'results.json',metadata)
    lines=['| Trial | BM x BN x BK | Stages | Compute WGs | Persistent | Cycles | Speedup | Feedback samples | Status |',
           '|---|---|---|---|---|---|---|---|---|']
    for row in history:
        tile=' x '.join(str(row.get(key,'?')) for key in ('bm','bn','bk'))
        speedup=f"{baseline[0]/row['cycles']:.4f}x" if 'cycles' in row else '-'
        lines.append(f"| {row['candidate']} | {tile} | {row.get('stages','-')} | {row.get('groups','-')} | {row.get('persistent','-')} | {row.get('cycles','-')} | {speedup} | {row['training_samples_before']} | {row['status']} |")
    lines+=['',f"Baseline: {baseline[0]} cycles. Best: {best['cycles']} cycles; {metadata['speedup']:.4f}x speedup.",
            f"Native MetaSchedule feedback updates: {len(updates)}. Seed: {args.seed}.",
            '','The objective is simulated cycles, not physical GPU measurements.']
    (args.out_dir/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
