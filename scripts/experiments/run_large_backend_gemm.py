"""Initial C++ GEMM controls before the FlashAccel grouped SRAM changes."""
import argparse
import os
import concurrent.futures,subprocess,time,json
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'validation/results/1112-flashaccel/initial-gemm'
out.mkdir(parents=True,exist_ok=True)
cases={'hbm':'0F5M_channels','hbfine':'4F1M_hbfine','eventlab':'4F1M_eventlab','flashlab':'4F1M_flashlab'}
def run(job):
 name,cfg,dim=job;start=time.perf_counter();cmd=['./build/gtsim_gemm','configs/v0.1.0/h100-sxm.yaml','--memory-config',f'configs/memconf/{cfg}.yaml',*map(str,dim)]
 with (out/f'{name}-{dim[0]}.log').open('w') as log:
  p=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
 result=dict(backend=name,dimensions=dim,command=cmd,exit_code=p.returncode,wall_seconds=time.perf_counter()-start)
 (out/f'{name}-{dim[0]}.json').write_text(json.dumps(result,indent=2))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--jobs',type=int,default=os.cpu_count() or 1)
args=parser.parse_args()
if args.jobs<1:parser.error('--jobs must be positive')
JOBS=args.jobs
with concurrent.futures.ThreadPoolExecutor(JOBS) as pool:list(pool.map(run,[(n,c,(s,s,s)) for s in (512,1024,2048) for n,c in cases.items()]))
