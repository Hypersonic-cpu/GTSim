"""Explicit legacy-oracle validation; never imported by production lowerers."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
from .programs.gemm_single import build as gemm
from .programs.fa3_single import build as fa3
from . import driver


def metrics(rows,key):
    measured=[r['measured_cycles'] for r in rows];simulated=[r[key] for r in rows]
    errors=[100*abs(a-b)/a for a,b in zip(measured,simulated)]
    xm=sum(measured)/len(rows);ym=sum(simulated)/len(rows)
    denominator=math.sqrt(sum((x-xm)**2 for x in measured)*sum((y-ym)**2 for y in simulated))
    return {'mape_percent':sum(errors)/len(errors),'max_ape_percent':max(errors),
            'pearson_r':sum((x-xm)*(y-ym) for x,y in zip(measured,simulated))/denominator if denominator else None}


def validate(args):
    if args.oracle!='legacy' or args.suite!='h100-v0.1.0':raise ValueError('unsupported oracle/suite')
    root=Path(__file__).resolve().parents[3];out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    build=Path(args.runner).resolve().parent
    helper=root/'tests/reference_parity/test_native_parity.py'
    spec=importlib.util.spec_from_file_location('independent_parity',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    profile=driver.hardware(args.runner,args.architecture,args.memory_config)
    (out/'graph-diffs').mkdir(exist_ok=True);(out/'failures').mkdir(exist_ok=True)
    summary={'status':'PASS','hardware':profile,'suite':args.suite,'workloads':{}}
    for name,fields,builder in (('gemm',('M','N','K'),gemm),('fa3',('batch','heads_q','heads_kv','seq_q','seq_kv'),fa3)):
        rows=[]
        with (root/f'validation/data/h100_{name}.csv').open() as f:cases=list(csv.DictReader(f))
        for index,case in enumerate(cases):
            dimensions=tuple(int(case[f]) for f in fields);directory=out/f'{name}-{index}'
            directory.mkdir(exist_ok=True)
            row={'case':index,'dimensions':str(dimensions),'measured_cycles':int(case['measured_cycles'])}
            try:
                kernel=module.compare(build/'reference_graph_dump',name,dimensions,profile['num_sms'])
                (out/'graph-diffs'/f'{name}-{index}.json').write_text(json.dumps({'status':'PASS','nodes':len(kernel.ops),'unapproved_differences':0})+'\n')
                del kernel
                timing=driver.run(builder(*dimensions),directory,args.runner,args.architecture,args.memory_config)
                command=[str(build/f'gtsim_{name}'),str(Path(args.architecture).resolve()),'--memory-config',str(Path(args.memory_config).resolve()),*map(str,dimensions)]
                with (directory/'reference.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,text=True)
                text=(directory/'reference.log').read_text();matches=re.findall(r'^Total cycles:\s*(\d+)\s*$',text,re.M)
                if result.returncode or not matches:raise RuntimeError('reference execution failed')
                row['reference_cycles']=int(matches[-1]);row['candidate_cycles']=timing['per_kernel'][0]['device_cycles']
                if row['reference_cycles']!=row['candidate_cycles']:raise RuntimeError(f'timing parity failure: {row}')
                row['ape_percent']=100*abs(row['candidate_cycles']-row['measured_cycles'])/row['measured_cycles']
                rows.append(row)
                print(name,dimensions,'PASS',row['candidate_cycles'],'cycles',flush=True)
            except Exception as error:
                (out/'failures'/f'{name}-{index}.json').write_text(json.dumps({'case':case,'error':str(error)},indent=2)+'\n')
                summary['status']='FAIL';(out/'validation-summary.json').write_text(json.dumps(summary,indent=2)+'\n');raise
        with (out/f'{name}_cases.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        reference=metrics(rows,'reference_cycles');candidate=metrics(rows,'candidate_cycles')
        alarms={metric:candidate[metric]>reference[metric]*factor for metric,factor in (('mape_percent',1.05),('max_ape_percent',1.10))}
        summary['workloads'][name]={'cases':len(rows),'reference':reference,'candidate':candidate,'relative_alarm':alarms,
            'mape_percentage_point_change':candidate['mape_percent']-reference['mape_percent'],
            'max_ape_percentage_point_change':candidate['max_ape_percent']-reference['max_ape_percent']}
        if any(alarms.values()):summary['status']='FAIL'
    (out/'validation-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Layered IR validation',f"\nStatus: {summary['status']}",'\nEvery case uses the same hardware YAML, memory YAML and current fork build for both candidate and reference.',
           '\nGraph parity includes operation names/types, tensor regions/strides, CTA resources, warp/SM placement and issue/completion edges.',
           '\n| Workload | Cases | MAPE | maxAPE | Pearson | MAPE change (pp) |','|---|---:|---:|---:|---:|---:|']
    for name,r in summary['workloads'].items():
        m=r['candidate'];lines.append(f"| {name} | {r['cases']} | {m['mape_percent']:.6f}% | {m['max_ape_percent']:.6f}% | {m['pearson_r']:.9f} | {r['mape_percentage_point_change']:.6f} |")
    lines+=['\nBoth relative accuracy alarms are evaluated against the corrected same-fork reference. Official v0.1.0 accuracy is a historical anchor, not a same-backend timing oracle.',
            '\nThis is a timing model; no numerical kernel execution or physical GPU validation is claimed.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    if summary['status']!='PASS':raise RuntimeError('accuracy alarms exceeded; see report')
    print(json.dumps(summary,indent=2))
