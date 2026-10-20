#!/usr/bin/env python3
"""Computation, native schedule replay, Hopper layout and synchronization tests."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/tvm'))
from pinned_tvm import load_tvm


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--build-dir',type=Path,default=ROOT/'build')
    parser.add_argument('--out-dir',type=Path,default=ROOT/'validation/results/tvm/auto-schedule-tests')
    args=parser.parse_args()
    tvm=load_tvm()
    import tvm_ffi
    from auto_ir import nodes,trace_json
    from auto_rules import apply_rule
    from gemm import ARCH,MEMORY,cycles,hardware_config,run_simulation
    from hopper_compiler import lower_hopper
    from native_dag import lower_native
    sys.path.insert(0,str(ROOT/'workloads/gemm'))
    from te_gemm import workload
    hw=hardware_config(args.build_dir)
    # The last three cases have >SM-count tiles and a partial final wave;
    # five/six K iterations reuse a two-stage ring with odd/even phase counts.
    cases=[((64,64,16),(64,64,16,1,1,0)),
           ((128,256,128),(128,128,64,2,2,0)),
           ((64,256,128),(64,128,16,2,8,1)),
           ((256,256,64),(256,64,32,1,2,0)),
           ((192,(hw['num_sms']//3+1)*64,96),(64,64,32,1,3,1)),
           ((192,(hw['num_sms']//3+1)*64,160),(64,64,32,1,2,1)),
           ((192,(hw['num_sms']//3+1)*64,192),(64,64,32,1,2,1))]
    last=None;reports=[]
    with tempfile.TemporaryDirectory() as directory:
        for index,(dimensions,parameters) in enumerate(cases):
            bm,per_group,bk,groups,depth,persistent=parameters
            computation=tvm.IRModule({'main':tvm.te.create_prim_func(workload(*dimensions))})
            sch=tvm.s_tir.Schedule(computation,debug_mask='all',seed=1)
            apply_rule(tvm,sch,dimensions,hw,dict(bm=bm,n_per_group=per_group,bk=bk,groups=groups,pipeline_delay=depth-1,persistent=persistent))
            saved=json.dumps(trace_json(sch.trace))
            replay=tvm.s_tir.Schedule(computation,debug_mask='all',seed=99)
            tvm.s_tir.schedule.Trace.apply_json_to_schedule(json.loads(saved),replay)
            tvm.ir.assert_structural_equal(sch.mod,replay.mod)
            native,pipeline,metadata=lower_hopper(tvm,sch.mod,hw)
            func=tvm.ir.load_json(tvm.ir.save_json(native))['main']
            rng=np.random.default_rng(123+index)
            m,n,k=dimensions
            a=rng.integers(-2,3,(m,k)).astype('float16')
            b=rng.integers(-2,3,(k,n)).astype('float16')
            ir,evidence=lower_native(tvm,func,hw,inputs=(a,b))
            # Integer inputs give an exact independent reduction oracle.
            golden=(a.astype('int32')@b.astype('int32')).astype('float16')
            assert np.isfinite(evidence['numerical_output']).all()
            np.testing.assert_array_equal(evidence['numerical_output'],golden)
            path=Path(directory)/f'case-{index}.gtsir';ir.write(path)
            log=Path(directory)/f'case-{index}.log'
            run_simulation(args.build_dir,ARCH,MEMORY,path,log)
            reports.append(dict(dimensions=dimensions,parameters=parameters,cycles=cycles(log.read_text()),
                                max_absolute_error=float(np.max(np.abs(evidence['numerical_output'].astype('float32')-golden))),
                                mismatched_elements=int(np.count_nonzero(evidence['numerical_output']!=golden))))
            # The lowerer must accept structurally different IR with the same
            # instruction semantics, rather than compare to a kernel template.
            unrolled=tvm.tirx.transform.UnrollLoop()(native)
            ir2,_=lower_native(tvm,unrolled['main'],hw)
            if ir.text()!=ir2.text():raise RuntimeError('native loop unrolling changed the lowered execution')
            last=(native,dimensions)
            print('PASS computation/replay/layout/synchronization/simulation',dimensions,parameters)
        native,dimensions=last
        def must_reject(bad,description):
            try:lower_native(tvm,bad['main'],hw)
            except ValueError:pass
            else:raise RuntimeError(f'accepted {description}')
        for operation in ('tirx.ptx.mbarrier_inval','tirx.cuda.mbarrier_wait','tirx.cuda.cta_sync','tirx.ptx.cp_async_bulk_wait_group',
                          'tirx.ptx.cp_async_bulk_commit_group','tirx.ptx.wgmma_wait_group',
                          'tirx.ptx.wgmma_commit_group','tirx.ptx.wgmma_fence','tirx.ptx.fence_proxy'):
            def remove(stmt,mapper):
                if isinstance(stmt.value,tvm.ir.Call) and stmt.value.op.name==operation:
                    return tvm.tirx.Evaluate(0)
                return mapper.default_mutate(stmt)
            bad=tvm_ffi.structural_mutate(native,(tvm.tirx.Evaluate,remove))
            must_reject(bad,f'missing synchronization: {operation}')
        for corruption in ('tensor_map_stride','register_order'):
            def corrupt(call,mapper):
                args=list(call.args)
                if corruption=='tensor_map_stride' and call.op.name=='tirx.tensormap_encode_tiled':
                    args[4]=args[4]+2
                elif corruption=='register_order' and call.op.name=='tirx.ptx.wgmma_f16_ss':
                    args[0],args[1]=args[1],args[0]
                else:return mapper.default_mutate(call)
                return tvm.ir.Call(call.op,args,call.attrs,ret_ty=call.ty)
            must_reject(tvm_ffi.structural_mutate(native,(tvm.ir.Call,corrupt)),corruption)
        lane=next(node.loop_var for node in nodes(native,tvm.tirx.For)
                  if node.thread_binding is not None and node.thread_binding.thread_tag=='threadIdx.x')
        def diverge(node,mapper):
            if node.attr_key=='gtsim.wgmma_region':
                return tvm.tirx.IfThenElse(lane%32==0,node,None)
            return mapper.default_mutate(node)
        must_reject(tvm_ffi.structural_mutate(native,(tvm.tirx.AttrStmt,diverge)),'lane-divergent WGMMA')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'correctness.json').write_text(json.dumps(reports,indent=2)+'\n')
    print('Computation-only scheduling and operation-driven lowering tests passed')


if __name__=='__main__':main()
