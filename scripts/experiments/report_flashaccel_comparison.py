"""Validate and report the complete frontend FlashAccel comparison matrix."""
import argparse
import csv
import json
from pathlib import Path

WORKLOADS=('gemm512','gemm1024','gemm2048','qwen30b','qwen235b')
MODES=(('hbm',0),*((b,p) for b in ('hbfine','eventlab','flashlab') for p in (0,1)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--shared-controls',type=Path, help='Reuse FullHBM and No Prefetch cases from this directory')
    parser.add_argument('--expect-streaming',choices=('true','false'),default='true')
    parser.add_argument('--allow-infeasible',action='store_true',help='Report explicit frontend SRAM-capacity rejection as infeasible')
    args=parser.parse_args()
    rows=[]; missing=[]; results={}
    for workload in WORKLOADS:
        for backend,prefetch in MODES:
            case=args.directory/f'{workload}-{backend}-pf{prefetch}'
            if args.shared_controls and (backend=='hbm' or not prefetch):
                case=args.shared_controls/case.name
            result=case/'run/result.json'
            job=json.loads((case/'job.json').read_text()) if (case/'job.json').exists() else {}
            if args.allow_infeasible and job.get('state')=='failed':
                log=(case/'launcher.log').read_text()
                reason='a full-reduction CTA exceeds SRAM capacity'
                if reason in log and not (case/'run/simulation.log').exists():
                    results[workload,backend,prefetch]=None
                    rows.append(dict(workload=workload,backend=backend,prefetch=prefetch,
                                     streaming=False,status='infeasible',reason=reason,
                                     wall_seconds=job['wall_seconds']))
                    continue
            if not result.exists() or job.get('state')!='completed':
                missing.append(case.name+': '+job.get('state','missing'));continue
            data=json.loads(result.read_text());timing=data['timing']
            residency=json.loads((case/'inputs/residency.json').read_text())
            if prefetch:
                assert residency['streaming']==(args.expect_streaming=='true'),case.name
            assert residency['buffer_count']==2,case.name
            assert residency['plane_bytes']*residency['dies_per_stack']*residency['planes_per_die']+residency['stack_bytes']==8*1024*1024,case.name
            assert all(value for key,value in timing.items() if key in
                       ('persistent_gpu','persistent_memory','persistent_l2','stable_allocations')),case.name
            assert data['returncode']==0 and not data['errors'],case.name
            assert all(k['state']=='COMPLETED' for k in timing['per_kernel']),case.name
            assert timing['traffic_bytes'].get('HBF_write',0)==0,case.name
            counters={};peak=0; read_completions=0; read_latency=0
            for kernel in timing['per_kernel']:
                memory=kernel['memory'].get('HBF',{})
                read_completions+=memory.get('read_completions',0)
                read_latency+=memory.get('read_latency_cycles',0)
                for key,value in memory.get('counters',{}).items():
                    counters[key]=counters.get(key,0)+value
                peak=max(peak,memory.get('gauges',{}).get('sram_peak_reserved_bytes',0))
            assert peak<=32*1024*1024,case.name
            if backend!='hbm' and prefetch:
                assert counters.get('prefetch_bytes',0)>0,case.name
                assert counters.get('resident_read_bytes',0)>0,case.name
            if not prefetch:
                assert counters.get('prefetch_bytes',0)==0,case.name
            results[workload,backend,prefetch]=timing['total_cycles']
            rows.append(dict(workload=workload,backend=backend,prefetch=prefetch,
                             status='completed',streaming=residency['streaming'] if prefetch else None,
                             prefetch_capacity_stalls=timing['prefetch_capacity_stalls'],
                             prefetch_canceled=timing['prefetch_canceled'],
                             cycles=timing['total_cycles'],time_us=timing['total_cycles']/1980,
                             wall_seconds=job['wall_seconds'],sram_peak_bytes=peak,
                             hbf_read_requests=read_completions,
                             hbf_avg_read_latency_cycles=read_latency/read_completions if read_completions else None,
                             sense_amplification=(counters.get('sensed_bytes',counters.get('nand_senses',0)*4096)/timing['traffic_bytes']['HBF_read']) if timing['traffic_bytes'].get('HBF_read',0) else None,
                             **timing['traffic_bytes'],**counters))
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with (args.directory/'comparison.csv').open('w') as output:
        writer=csv.DictWriter(output,keys);writer.writeheader();writer.writerows(rows)
    (args.directory/'incomplete.json').write_text(json.dumps(missing,indent=2)+'\n')
    if missing:
        print(f'{len(rows)}/35 completed; see incomplete.json')
        return 1
    columns=(('hbm',0),('hbfine',1),('eventlab',1),('flashlab',1),
             ('hbm',0),('hbfine',0),('eventlab',0),('flashlab',0))
    lines=[f'GPU cycles; streaming={args.expect_streaming}. FullHBM is the same shared control in both groups.',
           '', '| Workload | Pf: FullHBM | HBFine | EventLab | FlashLab | No Pf: FullHBM | HBFine | EventLab | FlashLab |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for workload in WORKLOADS:
        lines.append('| '+workload+' | '+' | '.join((f'{results[workload,b,p]:,}' if results[workload,b,p] is not None else 'Infeasible') for b,p in columns)+' |')
    text='\n'.join(lines)+'\n'
    (args.directory/'comparison.md').write_text(text);print(text)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
