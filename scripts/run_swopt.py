#!/usr/bin/env python3
"""Independent P2/P3 HBF experiments; default four concurrent simulations."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import run_sweep as sweep

WORKLOADS = {
    "G1": ("gemm", {"M":2048,"N":2048,"K":2048}),
    "G2": ("gemm", {"M":2048,"N":2048,"K":4096}),
    "G3": ("gemm", {"M":2048,"N":1536,"K":8192}),
    "F1": ("fa3", {"batch":1,"heads_q":28,"heads_kv":4,"seq_q":1024,"seq_kv":1024,"head_dim":128}),
    "F2": ("fa3", {"batch":1,"heads_q":28,"heads_kv":4,"seq_q":1536,"seq_kv":1536,"head_dim":128}),
    "F3": ("fa3", {"batch":2,"heads_q":28,"heads_kv":4,"seq_q":1024,"seq_kv":1024,"head_dim":128}),
}

def flags(**options):
    return [x for key,value in options.items() for x in ("--"+key.replace("_","-"),str(value))]

def variants(workload):
    yield "original", "baseline", []
    for depth in (3,4): yield "p2-depth", f"stages{depth}", flags(stages=depth)
    for depth in (2,3,4):
        tile = flags(bm=64,bn=128,bk=64) if workload=="gemm" else flags(q_tile=128,kv_tile=64)
        yield "p2-controlled-depth", f"small-tile-stages{depth}", tile+flags(stages=depth)
    if workload=="gemm":
        for bm in (64,128):
            for bn in (128,256):
                for bk in (32,64,128):
                    if (bm,bn,bk)!=(128,256,64):
                        yield "p3-tiles",f"BM{bm}-BN{bn}-BK{bk}",flags(bm=bm,bn=bn,bk=bk)
        orders=("m-major","n-major","grouped")
    else:
        for q in (64,128):
            for kv in (64,128,256):
                if (q,kv)!=(128,128):
                    yield "p3-tiles",f"Q{q}-KV{kv}-D128",flags(q_tile=q,kv_tile=kv)
        orders=("query-major","kv-head-grouped")
    for order in orders: yield "p3-cta-order",order,flags(cta_order=order)
    for factor in (2,4,8): yield "p3-split",f"split{factor}",flags(split=factor)


def combination_variants(wid, workload, records):
    candidates=[r for r in records if r['workload_id']==wid and r['topology']=='CoLocated' and r['status']=='completed']
    orders=[r for r in candidates if r['phase'] in ('original','p3-cta-order')]
    if not orders:raise RuntimeError('run independent baseline/order experiments first')
    best_order=min(orders,key=lambda r:r['total_cycles'])
    policy='original'
    argv=best_order['parameters'].get('kernel_args',[])
    if '--cta-order' in argv:policy=argv[argv.index('--cta-order')+1]
    yield 'p2-p3-combined', f'depth3-order-{policy}', flags(stages=3,cta_order=policy)
    tiles=[r for r in candidates if r['phase'] in ('original','p3-tiles')]
    best_tile=min(tiles,key=lambda r:r['total_cycles'])
    tile_flags=best_tile['parameters'].get('kernel_args',[])
    if not tile_flags:return
    options=dict(zip(tile_flags[::2],map(int,tile_flags[1::2])))
    if workload=='gemm':
        bm,bn,bk=options['--bm'],options['--bn'],options['--bk']
        shared=3*2*bk*(bm+bn)+2*bm*bn
    else:
        q,kv=options['--q-tile'],options['--kv-tile']
        shared=2*q*128+3*4*kv*128
    if shared<=227*1024:
        yield 'p2-p3-combined', f'depth3-{best_tile["variant"]}-order-{policy}', tile_flags+flags(stages=3,cta_order=policy)


def write_report(root, records, parallelism=4):
    base={(r['workload_id'],r['topology']):r['total_cycles'] for r in records if r['variant']=='baseline' and r['status']=='completed'}
    controls={(r['workload_id'],r['topology']):r['total_cycles'] for r in records if r['phase']=='p2-controlled-depth' and r['variant']=='small-tile-stages2' and r['status']=='completed'}
    for r in records:
        if r['status']=='completed':
            baseline=base.get((r['workload_id'],r['topology']))
            r['speedup_vs_original']=baseline/r['total_cycles'] if baseline else None
            if r['phase']=='p2-controlled-depth':
                control=controls.get((r['workload_id'],r['topology']))
                r['speedup_vs_same_tile_depth2']=control/r['total_cycles'] if control else None
    prior_path=sweep.ROOT/'validation/results/gemm_fa3-param-sweep/results/baseline/results.json'
    previous=json.loads(prior_path.read_text()).get('runs',[]) if prior_path.exists() else []
    mismatches=[]
    for r in records:
        if r['variant'] != 'baseline' or r['status'] != 'completed': continue
        params={k:v for k,v in r['parameters'].items() if k!='kernel_args'}
        for old in previous:
            if old['workload']==r['workload'] and old['topology']==r['topology'] and old['parameters']==params and old['status']=='completed':
                r['expected_original_cycles']=old['total_cycles']
                r['baseline_equivalent']=r['total_cycles']==old['total_cycles']
                if not r['baseline_equivalent']:mismatches.append(r['directory'])
                break
    sweep.write_json(root/'results.json', {'parallelism':parallelism,'runs':records})
    with (root/'results.csv').open('w',newline='') as f:
        keys=['workload_id','workload','phase','variant','topology','parameters_json','status','total_cycles',
              'speedup_vs_original','speedup_vs_same_tile_depth2','actual_nodes','resources_json','duration_seconds',
              'memory_statistics_json','git_commit','git_dirty','source_snapshot','expected_original_cycles','baseline_equivalent','directory','command_json','error']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for r in records:
            row={k:r.get(k,'') for k in keys}
            row['resources_json']=json.dumps(r.get('resources',{}),sort_keys=True)
            row['memory_statistics_json']=json.dumps(r.get('memory_statistics',{}),sort_keys=True)
            writer.writerow(row)
    done=sum(r['status']=='completed' for r in records)
    rejected=[r for r in records if r['status']=='infeasible']
    errors=[r for r in records if r['status'] not in ('completed','infeasible')]
    lines=['# HBF software optimization experiments','',
           f'{done} completed; {len(rejected)} resource rejections; {len(errors)} failures/timeouts.',
           f'Original-cycle mismatches against prior baseline: {len(mismatches)}.',
           'Speedup = original total cycles / optimized total cycles, on the same workload and topology.',
           'Controlled P2 rows also report speedup against depth 2 at the same smaller tile.','',
           '| Workload | Topology | Original cycles | Best variant | Best cycles | Speedup |',
           '|---|---|---:|---|---:|---:|']
    for (w,t),cycles in sorted(base.items()):
        candidates=[r for r in records if r['workload_id']==w and r['topology']==t and r['status']=='completed']
        best=min(candidates,key=lambda r:r['total_cycles'])
        lines.append(f"| {w} | {t} | {cycles:,} | {best['phase']}/{best['variant']} | {best['total_cycles']:,} | {cycles/best['total_cycles']:.3f}x |")
    lines+=['','## Independent contributions','',
            '| Workload | Topology | Family | Best variant | Speedup vs original |',
            '|---|---|---|---|---:|']
    for w,t in sorted(base):
        for phase in ('p2-depth','p2-controlled-depth','p3-tiles','p3-cta-order','p3-split','p2-p3-combined'):
            candidates=[r for r in records if r['workload_id']==w and r['topology']==t and r['phase']==phase and r['status']=='completed']
            if candidates:
                best=min(candidates,key=lambda r:r['total_cycles'])
                lines.append(f"| {w} | {t} | {phase} | {best['variant']} | {best.get('speedup_vs_original',0):.3f}x |")
    lines+=['','## Directory mapping','',
            '`<phase>/<variant>/<workload>-<topology>/` contains `result.json` and `run.log`.',
            '`configurations.json` maps every directory to exact dimensions and CLI flags.',
            '`source_snapshot/` and versioned `source_snapshots/` preserve source revisions; each result names its snapshot and binary hash.',
            '`original/baseline` is the unchanged kernel configuration. P2 changes depth alone; controlled P2 fixes a smaller tile across depths.',
            '`p3-tiles` changes tiles at depth 2, original ordering, split 1. `p3-cta-order` changes ordering alone. `p3-split` changes split factor alone.',
            '', '## Resource and modeling limits','',
            'The existing H100 YAML caps residency at one CTA per SM; that cap remains active. Resource admission can reduce residency further but never increases the cap.',
            'Shared-memory footprints are accounted explicitly. Register footprints include FP32 accumulators/scores plus an assumed 32 registers per thread for other state, not compiler allocation measurements.',
            'Split scratch is FP32 in HBM (overridden to HBF in FullHBF). Merge operations include reads, adds, exponentiation/normalization for attention, casts, and final stores.',
            'GTSim is a timing/DAG simulator and does not execute tensor values. The stable attention merge formula is verified numerically in the semantic unit test.',
            '', '## Rejected configurations','']
    for phase,variant,workload in sorted({(r['phase'],r['variant'],r['workload']) for r in rejected}):
        lines.append(f'- {workload}: `{phase}/{variant}` (see run.log for footprint/capacity).')
    if errors:
        lines+=['','## Failures','']+[f"- `{r['directory']}`: {r.get('error',r['status'])}" for r in errors]
    (root/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='validation/results/hbf-software-p2-p3')
    p.add_argument('--build-dir',default='build')
    p.add_argument('--parallelism',type=int,default=4)
    p.add_argument('--phase',action='append',choices=['original','p2-depth','p2-controlled-depth','p3-tiles','p3-cta-order','p3-split','p2-p3-combined'])
    p.add_argument('--workload',action='append',choices=tuple(WORKLOADS))
    p.add_argument('--variant',action='append')
    p.add_argument('--combinations',action='store_true',help='Evaluate feasible combinations selected from completed independent results.')
    p.add_argument('--topology',action='append',choices=sweep.TOPOLOGIES)
    p.add_argument('--no-representative',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--timeout-seconds',type=int,default=600)
    a=p.parse_args()
    if a.parallelism<1:p.error('parallelism must be positive')
    root=sweep.ROOT/a.output_dir
    build=sweep.ROOT/a.build_dir
    commit,dirty=sweep.git_metadata()
    arch_hash=sweep.sha256_file(sweep.ARCH_CONFIG)
    binaries={w:build/('gtsim_gemm' if w=='gemm' else 'gtsim_fa3') for w in ('gemm','fa3')}
    binary_hash={w:sweep.sha256_file(b) for w,b in binaries.items()}
    topologies=a.topology or ['FullHBM','CoLocated']
    source_paths=['workloads/common/workload_options.h','workloads/common/split_reduce.h','include/dag_node.h','include/simulator.h','include/sm.h',
                  'src/sm.cpp','workloads/gemm/gemm_h100.cpp','workloads/fa3/fa3_h100.cpp','scripts/run_sweep.py','scripts/run_swopt.py','tests/kernel_swopt_tests.cpp','CMakeLists.txt']
    source_hash=hashlib.sha256(''.join(sweep.sha256_file(sweep.ROOT/path) for path in source_paths).encode()).hexdigest()[:12]
    snapshot_name=f'source_snapshots/{source_hash}'
    prior_records=json.loads((root/'results.json').read_text()).get('runs',[]) if (root/'results.json').exists() else []
    specs=[]
    for wid,(workload,params) in WORKLOADS.items():
        if a.workload and wid not in a.workload:continue
        choices=combination_variants(wid,workload,prior_records) if a.combinations else variants(workload)
        for phase,variant,options in choices:
            if a.phase and phase not in a.phase:continue
            if a.variant and variant not in a.variant:continue
            tiers=list(topologies)
            representative=variant in ('baseline','stages3','small-tile-stages2','small-tile-stages4','split4') or a.combinations
            if not a.no_representative and not a.topology and wid in ('G3','F2') and representative:
                tiers+=['FullHBF','Cascaded']
            for topology in tiers:
                directory=f'{phase}/{variant}/{wid}-{topology}'
                params=dict(params,kernel_args=options)
                case={'name':wid,'workload':workload,'parameters':params}
                command=sweep.build_command(binaries[workload],case,sweep.MEMORY_CONFIGS[topology])
                specs.append(dict(workload_id=wid,workload=workload,parameters=params,
                    parameters_json=json.dumps(params,sort_keys=True),phase=phase,variant=variant,
                    topology=topology,directory=directory,command=command,command_json=json.dumps(command),
                    git_commit=commit,git_dirty=dirty,binary_sha256=binary_hash[workload],
                    architecture_sha256=arch_hash,memory_config_sha256=sweep.sha256_file(sweep.MEMORY_CONFIGS[topology]),
                    status='pending',total_cycles='',run_id=directory,source_snapshot=snapshot_name))
    if not specs:p.error('selectors matched no experiments')
    specs.sort(key=lambda r:(r['phase']!='original',r['phase'],r['variant'],r['workload_id'],r['topology']))
    if a.dry_run:
        for r in specs:print(r['directory'],' '.join(r['command']))
        print(f'{len(specs)} planned runs; parallelism={a.parallelism}')
        return 0
    root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'configurations.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'experiments':[]}
    entries={r['directory']:r for r in manifest['experiments']}
    for r in specs:
        prior=root/r['directory']/'result.json'
        entries[r['directory']]=json.loads(prior.read_text()) if prior.exists() and not a.overwrite else r
    sweep.write_json(manifest_path,{'git_commit':commit,'parallelism':a.parallelism,'experiments':list(entries.values())})
    snapshot=root/snapshot_name
    for path in source_paths:
        dest=snapshot/path;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(sweep.ROOT/path,dest)
    selected={r['directory'] for r in specs}
    previous=json.loads((root/'results.json').read_text()).get('runs',[]) if (root/'results.json').exists() else []
    records=[r for r in previous if r['directory'] not in selected]
    with ThreadPoolExecutor(max_workers=a.parallelism) as pool:
        futures={}
        for r in specs:
            directory=root/r['directory'];result=directory/'result.json'
            if result.exists() and not a.overwrite:
                records.append(json.loads(result.read_text()));continue
            directory.mkdir(parents=True,exist_ok=True)
            futures[pool.submit(sweep.execute_record,r,r['command'],directory/'run.log',a.timeout_seconds)]=result
        write_report(root,records,a.parallelism)
        for future in as_completed(futures):
            r=future.result();result=futures[future]
            records.append(r);sweep.write_json(result,r)
            write_report(root,records,a.parallelism)
            print(f"{sum(x['directory'] in selected for x in records)}/{len(specs)} {r['directory']}: {r['status']} {r['total_cycles']}",flush=True)
    write_report(root,records,a.parallelism)
    for r in records:sweep.write_json(root/r['directory']/'result.json',r)
    return int(any(r['status'] not in ('completed','infeasible') or r.get('baseline_equivalent') is False for r in records))

if __name__=='__main__':raise SystemExit(main())
