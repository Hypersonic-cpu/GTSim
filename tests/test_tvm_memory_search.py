#!/usr/bin/env python3
"""Native compiler placement, charged tier copies, capacity and launch ordering."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/tvm'),str(ROOT/'workloads/gemm')]
from pinned_tvm import load_tvm


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,default=ROOT/'build')
    args=parser.parse_args();tvm=load_tvm()
    from auto_ir import mutate
    from auto_rules import apply_rule,validate_placement
    from gemm import ARCH,hardware_config,inspect_graph,memory_statistics,run_simulation
    from hopper_compiler import lower_hopper
    from tier_memory import lower_module
    from te_gemm import workload
    reports=[]
    with tempfile.TemporaryDirectory() as tmp:
        for profile in ('full-hbm.yaml','colocated-1F4M.yaml','colocated-3F2M.yaml'):
            memory=ROOT/'configs/memconf'/profile
            hw=hardware_config(args.build_dir,memory=memory)
            for prefetch in ([0] if hw['topology']=='FullHBM' else [0,1]):
                dims=(128,256,128)
                mod=tvm.IRModule({'main':tvm.te.create_prim_func(workload(*dims))})
                sch=tvm.s_tir.Schedule(mod,debug_mask='all',seed=1)
                apply_rule(tvm,sch,dims,hw,dict(bm=64,n_per_group=128,bk=32,groups=2,pipeline_delay=1,persistent=1,prefetch=prefetch))
                native,_,meta=lower_hopper(tvm,sch.mod,hw)
                native=tvm.ir.load_json(tvm.ir.save_json(native))
                rng=np.random.default_rng(123)
                a=rng.integers(-2,3,(dims[0],dims[2])).astype('float16')
                b=rng.integers(-2,3,(dims[2],dims[1])).astype('float16')
                ir,evidence=lower_module(tvm,native,hw,inputs=(a,b))
                golden=(a.astype('int32')@b.astype('int32')).astype('float16')
                np.testing.assert_array_equal(evidence['numerical_output'],golden)
                path=Path(tmp)/f'{profile}-{prefetch}.gtsir';ir.write(path)
                graph=inspect_graph(args.build_dir.resolve(),memory,path)
                tiers=['HBM','HBM' if hw['topology']=='FullHBM' else 'HBF','HBM']+(['HBM'] if prefetch else [])
                assert [item['device'] for item in graph['tensors']]==tiers
                log=path.with_suffix('.log');run_simulation(args.build_dir,ARCH,memory,path,log)
                stats=memory_statistics(log.read_text())
                assert stats['HBF']['read_bytes']==(0 if hw['topology']=='FullHBM' else b.nbytes)
                assert stats['HBF']['write_bytes']==0
                assert stats['HBM']['write_bytes']==golden.nbytes+(b.nbytes if prefetch else 0)
                if prefetch:
                    stores={i for i,op in enumerate(ir.operations) if op.tensor==3 and op.op=='cp.sram2dram_tma'}
                    assert stores and all(stores.issubset(op.after_complete) for op in ir.operations if op.name.startswith('gemm_') and not op.after_issue and not any(i not in stores for i in op.after_complete))
                    assert sum(op.tile[2]*op.tile[3]*2 for op in ir.operations if op.tensor==3 and op.op=='cp.sram2dram_tma')==b.nbytes
                    for operation in ('tirx.cuda.mbarrier_wait','tirx.ptx.cp_async_bulk_wait_group'):
                        def remove(stmt,mapper):
                            if isinstance(stmt.value,tvm.ir.Call) and stmt.value.op.name==operation:return tvm.tirx.Evaluate(0)
                            return mapper.default_mutate(stmt)
                        bad=tvm.IRModule(dict(native.functions))
                        bad['prefetch']=mutate(native['prefetch'],tvm.tirx.Evaluate,remove)
                        try:lower_module(tvm,bad,hw)
                        except ValueError:pass
                        else:raise AssertionError('accepted missing copy synchronization')
                    bad=tvm.IRModule(dict(native.functions))
                    bad['main']=native['main'].with_body(tvm.tirx.SeqStmt([native['main'].body.seq[i] for i in (0,2,3)]))
                    try:lower_module(tvm,bad,hw)
                    except ValueError:pass
                    else:raise AssertionError('accepted missing launch synchronization')
                    bounded=dict(hw,hbm_capacity_bytes=a.nbytes+golden.nbytes)
                    validate_placement(dims,bounded,0)
                    try:validate_placement(dims,bounded,1)
                    except ValueError:pass
                    else:raise AssertionError('prefetch exceeded available HBM capacity')
                    limited=Path(tmp)/'limited.yaml'
                    limited.write_text(memory.read_text().replace('hbm:\n',f'hbm:\n  capacity_bytes: {bounded["hbm_capacity_bytes"]}\n',1))
                    try:run_simulation(args.build_dir,ARCH,limited,path,Path(tmp)/'limited.log')
                    except RuntimeError:pass
                    else:raise AssertionError('backend accepted over-capacity allocations')
                reports.append(dict(memory=profile,prefetch=meta['prefetch_strategy'],max_absolute_error=0,
                                    memory_statistics=stats))
        out=ROOT/'validation/results/tvm/memory-search-tests';out.mkdir(parents=True,exist_ok=True)
        (out/'correctness.json').write_text(json.dumps(reports,indent=2)+'\n')
    print('Native tier placement, numerical semantics, charged copies, capacity and launch ordering passed')


if __name__=='__main__':main()
