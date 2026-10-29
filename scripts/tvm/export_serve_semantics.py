"""Export unconstrained serving algorithms, with explicit placement metadata.

This exports semantic inputs only. The current GEMM-only Hopper compiler does
not lower these functions, so there is deliberately no fabricated best schedule.
"""
import argparse,json,sys
from pathlib import Path
from pinned_tvm import ROOT,load_tvm
p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,default=ROOT/'validation/results/serve-3m2f/semantic-ir');a=p.parse_args()
tvm=load_tvm();sys.path.insert(0,str(ROOT/'workloads/serve'));import semantics
a.out_dir.mkdir(parents=True,exist_ok=True)
records={}
for name,fun in [('fa2',semantics.fa2),('gdn',semantics.gdn)]:
    tensors=fun();tiers={t.name:('HBF' if t.name.startswith('W_') or t.name in ('A_log','dt_bias') else 'HBM') for t in tensors}
    f=tvm.te.create_prim_func(tensors).with_attr('gtsim.semantic_tensor_tiers',json.dumps(tiers,sort_keys=True))
    mod=tvm.IRModule({'main':f});(a.out_dir/(name+'.tir')).write_text(mod.script())
    records[name]={'tensors':{t.name:{'shape':[int(n) for n in t.shape],'dtype':str(t.dtype),'tier':tiers[t.name]} for t in tensors},'status':'semantic_export_only_not_optimized'}
(a.out_dir/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
(a.out_dir/'search-status.json').write_text(json.dumps({'status':'not_run_unsupported_lowering','workers':1,'reason':'Existing native_dag/compiler accepts Hopper GEMM reduction plus cast, not Ampere FA2 or recurrent/fused GDN. Extend and validate lowering before MetaSchedule feedback.','best_ir':None},indent=2)+'\n')
print('Exported FA2/GDN semantic TensorIR; search not run (unsupported lowering).')
