#!/usr/bin/env python3
"""Four independent GEMM optimizers, each tuning its memory profiles in order."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[2]
SHAPES=((512,512,512),(1024,1024,512),(2048,2048,512),(1024,1024,1024))
PROFILES=('full-hbm.yaml','colocated-1F4M.yaml','colocated-3F2M.yaml')


def worker(args,shape):
    label='x'.join(map(str,shape))
    for profile in PROFILES:
        out=args.out_dir/label/Path(profile).stem
        out.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(ROOT/'scripts/tvm/tune_gemm.py'),
                 '--m',str(shape[0]),'--n',str(shape[1]),'--k',str(shape[2]),
                 '--trials',str(args.trials),'--seed',str(args.seed),'--jobs','1',
                 '--build-dir',str(args.build_dir.resolve()),'--memory-config',str(ROOT/'configs/memconf'/profile),
                 '--out-dir',str(out.resolve())]
        if (out/'run-spec.json').exists():command.append('--resume')
        (out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        with (out/'optimizer.log').open('a') as log:
            completed=subprocess.run(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f'{label}/{profile}: optimizer failed; see {out}/optimizer.log')
        # Verify the best saved trace and interpreted instruction values in
        # this same worker before advancing to the next memory configuration.
        with (out/'replay.log').open('a') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/tvm/replay_gemm.py'),
                            '--search-dir',str(out.resolve()),'--build-dir',str(args.build_dir.resolve())],
                           cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'COMPLETE {label}/{profile}',flush=True)


def collect(args):
    rows=[]
    for shape in SHAPES:
        label='x'.join(map(str,shape));baseline=None
        for profile in PROFILES:
            path=args.out_dir/label/Path(profile).stem/'results.json'
            if not path.exists():
                rows.append(dict(M=shape[0],N=shape[1],K=shape[2],memory_config=profile,status='incomplete'))
                continue
            result=json.loads(path.read_text());best=result['best']
            if profile=='full-hbm.yaml':baseline=best['cycles']
            parameters={key:best[key] for key in ('bm','bn','bk','stages','groups','persistent')}
            rows.append(dict(M=shape[0],N=shape[1],K=shape[2],memory_config=profile,
                best_cycles=best['cycles'],slowdown_vs_full_hbm=best['cycles']/baseline if baseline else '',
                schedule_parameters=json.dumps(parameters,sort_keys=True),prefetch_strategy=best['prefetch_strategy'],
                status='complete',best_candidate=best['candidate'],memory_statistics=json.dumps(best['memory_statistics'],sort_keys=True),
                migration_payload_bytes=best['migration_payload_bytes'],capacity_known=best['capacity_known'],
                successful_trials=result['successful_candidates'],feedback_updates=result['feedback_updates'],
                directory=str(path.parent)))
    columns=['M','N','K','memory_config','best_cycles','slowdown_vs_full_hbm','schedule_parameters',
             'prefetch_strategy','status','best_candidate','memory_statistics','migration_payload_bytes',
             'capacity_known','successful_trials','feedback_updates','directory']
    with (args.out_dir/'best-results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    (args.out_dir/'best-results.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=int,default=32)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--build-dir',type=Path,default=ROOT/'build')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'validation/results/tvm/memory-auto-search')
    parser.add_argument('--collect-only',action='store_true')
    args=parser.parse_args();args.out_dir=args.out_dir.resolve();args.out_dir.mkdir(parents=True,exist_ok=True)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','TVM_NUM_THREADS'):
        os.environ[name]='1'
    if not args.collect_only:
        spec=dict(shapes=SHAPES,profiles=PROFILES,trials=args.trials,seed=args.seed,
                  optimizer_processes=4,threads_per_optimizer=1,
                  launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (args.out_dir/'batch-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(worker,args,shape) for shape in SHAPES]
            for future in futures:future.result()
    rows=collect(args)
    if any(row['status']!='complete' for row in rows):raise RuntimeError('some optimization experiments are incomplete')
    print('All 12 experiments completed; see '+str(args.out_dir/'best-results.csv'))


if __name__=='__main__':main()
