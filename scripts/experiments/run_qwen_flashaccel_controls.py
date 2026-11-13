"""Run FullHBM and HBFine controls using the generated experiment inputs."""
import argparse
import os
import subprocess,concurrent.futures
from pathlib import Path
root=Path(__file__).resolve().parents[2];base=root/'validation/results/1112-flashaccel';inputs=base/'inputs'
def run(job):
 model,mode=job
 params='b1.json' if model=='30b' else 'qwen3-235b-a22b-b1.json'
 source='frontend/asm/qwen3/'+('one_block.progir' if mode=='hbm' else 'flash-accel/one_block.progir')
 memory=root/'configs/memconf/0F5M_channels.yaml' if mode=='hbm' else inputs/f'{mode}.yaml'
 out=base/f'qwen-{model}-{mode}'
 cmd=['python3','frontend/tools/run_qwen_simulations.py','--params',f'frontend/asm/qwen3/{params}','--source',source,'--runner','build/gtsim_program','--architecture','configs/v0.1.0/h100-sxm.yaml','--memory',str(memory),'--out',str(out),'--max-cycles','1000000000']
 if mode=='hbfine-pf':cmd+=['--prefetch-config',str(inputs/'double-buffer.json')]
 with (base/f'qwen-{model}-{mode}-launcher.log').open('w') as log:
  return subprocess.run(cmd,cwd=root,env={**os.environ,'PYTHONPATH':'frontend/src'},stdout=log,stderr=subprocess.STDOUT).returncode
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--jobs',type=int,default=os.cpu_count() or 1)
args=parser.parse_args()
if args.jobs<1:parser.error('--jobs must be positive')
JOBS=args.jobs
with concurrent.futures.ThreadPoolExecutor(JOBS) as pool:print(list(pool.map(run,[(m,v) for m in ('30b','235b') for v in ('hbm','hbfine-no-pf','hbfine-pf')])))
