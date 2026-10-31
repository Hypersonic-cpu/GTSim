"""Actual runner tests: persistent caches/allocations, events and rejection paths."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'frontend/src'))
from gtsim_frontend.ir.program import Program,Invocation,Command
from gtsim_frontend.programs.gemm_single import build as gemm
from gtsim_frontend.programs.fa3_single import build as fa3
from gtsim_frontend import driver


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runner',required=True);parser.add_argument('--architecture',required=True);parser.add_argument('--memory-config',required=True);args=parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        g=gemm(128,256,128);f=fa3()
        p=Program(symbol='mixed',tensors=g.tensors+f.tensors,launches=g.launches+f.launches,
            commands=(Command('LaunchKernel','gemm0'),Command('RecordEvent','gemm_done'),Command('WaitEvent','gemm_done'),
                      Command('LaunchKernel','fa30'),Command('StreamSynchronize'),Command('RecordEvent','all_done')))
        t=driver.run(p,Path(temporary)/'mixed',args.runner,args.architecture,args.memory_config)
        a,b=t['per_kernel']
        assert a['state']==b['state']=='COMPLETED'
        assert b['start_cycle']==a['complete_cycle'] and b['complete_cycle']>b['start_cycle']
        assert t['events']['gemm_done']==a['complete_cycle'] and t['events']['all_done']==t['total_cycles']
        assert b['l2_lines_before']==a['l2_lines_after']>0
        assert all(t[k] for k in ('persistent_gpu','persistent_memory','persistent_l2','stable_allocations'))
        warm=replace(g,symbol='warm',launches=g.launches+(replace(g.launches[0],id='gemm1'),),
                     commands=(Command('LaunchKernel','gemm0'),Command('LaunchKernel','gemm1'),Command('StreamSynchronize')))
        t=driver.run(warm,Path(temporary)/'warm',args.runner,args.architecture,args.memory_config)
        a,b=t['per_kernel'];assert b['hbm_read_bytes']<a['hbm_read_bytes'],(a,b)
        assert b['l2_lines_before']==a['l2_lines_after']
        # Check configured launch overhead on an otherwise identical cold launch.
        cold=driver.run(g,Path(temporary)/'cold',args.runner,args.architecture,args.memory_config)
        overhead=driver.run(g,Path(temporary)/'overhead',args.runner,args.architecture,args.memory_config,17)
        assert overhead['total_cycles']==cold['total_cycles']+17
        assert overhead['per_kernel'][0]['device_cycles']==cold['per_kernel'][0]['device_cycles']
        empty=driver.run(Program(symbol='empty'),Path(temporary)/'empty',args.runner,args.architecture,args.memory_config)
        assert empty['total_cycles']==0 and not empty['per_kernel']
        original=json.loads((Path(temporary)/'cold/program.gtsir').read_text())
        mutations=[('streams',['a','b']),('commands',[{'record':'Command','kind':'WaitEvent','target':'missing'}]),
                   ('commands',[{'record':'Command','kind':'LaunchKernel','target':'missing'}])]
        for key,value in mutations:
            bad=dict(original);bad[key]=value
            path=Path(temporary)/'cold/bad.gtsir';path.write_text(json.dumps(bad))
            result=subprocess.run([args.runner,args.architecture,'--memory-config',args.memory_config,str(path)],capture_output=True,text=True)
            assert result.returncode!=0,result.stdout
        # Malformed graphs must fail before executing any launch.
        graph=json.loads((Path(temporary)/'cold/kernel.gtsir').read_text())
        import copy
        bad_graphs=[]
        bad=copy.deepcopy(graph);bad['target']='sm100';bad_graphs.append(bad)
        bad=copy.deepcopy(graph);bad['ops'][0]['after_complete']=[len(bad['ops'])-1];bad_graphs.append(bad)
        bad=copy.deepcopy(graph);bad['ops'][0]['after_complete']=[999999];bad_graphs.append(bad)
        bad=copy.deepcopy(graph);bad['outputs']=[];bad_graphs.append(bad)
        bad=copy.deepcopy(graph);bad['ops'][-1]['op']='st.reg2sram';bad_graphs.append(bad)
        bad=copy.deepcopy(graph)
        index=next(i for i,o in enumerate(bad['ops']) if o['op']=='wgmma.m64n64k16')
        bad['ops'][index]['op']='local_max';bad['ops'][index]['pipeline']='SIMD';bad_graphs.append(bad)
        bad=copy.deepcopy(graph);bad['ops'][0]['op']='unknown_sfu';bad_graphs.append(bad)
        for bad in bad_graphs:
            (Path(temporary)/'cold/badkernel.gtsir').write_text(json.dumps(bad))
            manifest=copy.deepcopy(original);manifest['launches'][0]['kernel_file']='badkernel.gtsir'
            path=Path(temporary)/'cold/bad.gtsir';path.write_text(json.dumps(manifest))
            result=subprocess.run([args.runner,args.architecture,'--memory-config',args.memory_config,str(path)],capture_output=True,text=True)
            assert result.returncode!=0 and 'Total cycles:' not in result.stdout,(result.stdout,result.stderr)
        manifest=copy.deepcopy(original)
        manifest['tensors'][1]['read_only']=False
        manifest['launches'][0]['bindings'][2][1]=manifest['launches'][0]['bindings'][1][1]
        path=Path(temporary)/'cold/bad.gtsir';path.write_text(json.dumps(manifest))
        result=subprocess.run([args.runner,args.architecture,'--memory-config',args.memory_config,str(path)],capture_output=True,text=True)
        assert result.returncode!=0 and 'alias' in result.stderr,result.stderr
        print('Mixed GEMM→FA3, warm shared allocations/L2, events, empty stream, overhead and runtime rejection checks passed')


if __name__=='__main__':main()
