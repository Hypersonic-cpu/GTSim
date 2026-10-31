"""Hopper expansion of the explicit FA3 streaming-softmax Flow schedule."""
from collections import Counter
from dataclasses import replace
from ..ir.arch import ArchKernel, ArchInstruction, KernelTensor, CTA, HOPPER_OPS
from ..serialization.json_codec import digest


def lower(flow):
    flow.verify()
    if flow.family!='fa3' or flow.schedule_id!='ref_h100_fa3_v0_1_0':raise ValueError('unsupported FA3 schedule')
    allowed={'','initial_max_negative_infinity','initial_sum_zero','initial_output_zero','scaled_QK','PV_accumulate',
             'local_max','shfl_max','reduce_max','merge_max','scores_scale','exp2','local_sum','shfl_sum',
             'reduce_sum','merge_sum','logsum','acc_o_scale','acc_s_cast','acc_o_div','cast_fp16_store'}
    if any(o.semantic not in allowed for o in flow.ops):raise ValueError('unsupported FA3 Flow operation')
    qs=[t for t in flow.tensors if t.id.startswith('Q.')]
    ks=[t for t in flow.tensors if t.id.startswith('K.')]
    batch=max(int(t.id.split('.')[1]) for t in qs)+1
    hq=len(qs)//batch;hkv=len(ks)//batch
    tensors=[]
    # Keep reference tensor identities, including GQA holes, without allocating
    # any physical address in the compiler.
    for b in range(batch):
        for h in range(hq):
            for offset,letter in ((0,'Q'),(3,'O')):
                t=next(t for t in flow.tensors if t.id==f'{letter}.{b}.{h}')
                tensors.append(KernelTensor((b*hq+h)*4+offset,f'{letter}_b{b}_h{h}',t.shape,(t.shape[1],1),preloaded=letter=='Q',read_only=letter=='Q'))
        for h in range(hkv):
            for offset,letter in ((1,'K'),(2,'V')):
                t=next(t for t in flow.tensors if t.id==f'{letter}.{b}.{h}')
                tensors.append(KernelTensor((b*hkv+h)*4+offset,f'{letter}_b{b}_h{h}',t.shape,(t.shape[1],1)))
    ops=[];ctas=[]
    for cta,sm in flow.mapping.ctas:
        logical=[o for o in flow.ops if o.id.startswith(f'cta{cta}.')]
        stages=Counter(o.semantic for o in logical)
        rounds=stages['scaled_QK']
        required={'PV_accumulate':rounds,'local_max':2*rounds,'shfl_max':2*rounds,'reduce_max':2*rounds,
                  'merge_max':rounds,'scores_scale':rounds,'exp2':2*rounds,'local_sum':2*rounds,
                  'shfl_sum':2*rounds,'reduce_sum':2*rounds,'merge_sum':rounds,'logsum':rounds,
                  'acc_o_scale':rounds,'acc_s_cast':rounds,'acc_o_div':1}
        if rounds<1 or any(stages[s]!=n for s,n in required.items()):raise ValueError('incomplete FA3 stage graph')
        qload=next(o for o in logical if o.id.endswith('.Q.load')).reads[0]
        b,h=map(int,qload.tensor.split('.')[1:]);base_r=qload.offsets[0]
        kvh=h//(hq//hkv);p=f'TB{cta}'
        ctas.append(CTA(cta,sm,12,2*128*128+2*4*128*128,4*128*(128+128)+12*32*32*4))
        def emit(name,op,warp,tile,deps=(),issue=(),tensor=0,dtype='fp16',ld=0):
            nid=len(ops)
            ops.append(ArchInstruction(nid,name,op,HOPPER_OPS[op],cta,cta*12+warp,tensor,tuple(tile),dtype,ld,tuple(issue),tuple(deps)))
            return nid
        qids=[emit(f'{p}_Q_Load{i}_b{b}_h{h}','cp.dram2sram_tma',8,(base_r+i*64,0,64,128),tensor=(b*hq+h)*4,ld=128) for i in range(2)]
        ss_rounds=[];rs_rounds=[];last_ss=[None]*8;last_rs=[None]*8
        logsum=[None]*8;casts=[None]*8;scales=[None]*8
        for k in range(rounds):
            kr=next(o for o in logical if o.id==f'cta{cta}.k{k}.K.load').reads[0]
            vr=next(o for o in logical if o.id==f'cta{cta}.k{k}.V.load').reads[0]
            if kr.shape!=(128,128) or vr.shape!=(128,128):raise ValueError('unsupported FA3 KV tile geometry')
            kvlen=next(t.shape[1] for t in flow.tensors if t.id==kr.tensor)
            kid=[emit(f'{p}_K_Load{i}_k{k}','cp.dram2sram_tma',8,(i*64,kr.offsets[1],64,128),
                      ss_rounds[k-2] if k>=2 else (),tensor=(b*hkv+kvh)*4+1,ld=kvlen) for i in range(2)]
            vid=[emit(f'{p}_V_Load{i}_k{k}','cp.dram2sram_tma',8,(i*64,vr.offsets[1],64,128),
                      rs_rounds[k-2] if k>=2 else (),tensor=(b*hkv+kvh)*4+2,ld=kvlen) for i in range(2)]
            ss=[];rs=[]
            for warp in range(8):
                g=warp//4;gw=cta*12+warp;row=g*64
                for step in range(16):
                    deps=qids+kid+([last_ss[warp]] if last_ss[warp] is not None else [])
                    issue=[logsum[warp],casts[warp]] if k>0 and step==0 else []
                    last_ss[warp]=emit(f'{p}_WGMMA_SS_g{g}_w{gw}_k{k}_s{step}','wgmma.m64n64k16',warp,(row,0,16,64),deps,issue,tensor=3,dtype='fp32')
                    ss.append(last_ss[warp])
                if k>0:
                    index=scales[warp]
                    ops[index]=replace(ops[index],after_issue=ops[index].after_issue+(last_ss[warp],))
                def soft(label,opcode,shape,deps,issue=()):
                    return emit(f'{p}_Softmax_{label}_g{g}_w{gw}_k{k}',opcode,warp,(row,0,*shape),deps,issue)
                maxima=[]
                for branch in range(2):
                    a=soft(f'LocalMax_r{branch}','local_max',(8,128),[last_ss[warp]],[last_rs[warp]] if k>0 else [])
                    a=soft(f'ShflMax_r{branch}','shfl_max',(8,4),[a])
                    maxima.append(soft(f'ReduceMax_r{branch}','reduce_max',(8,4),[a]))
                maximum=soft('MergeMax','merge_max',(8,4),maxima)
                scale=soft('Scale','scores_scale',(8,16),[maximum])
                exps=[soft(f'Exp2_r{i}','exp2',(16,64),[scale]) for i in range(2)]
                sums=[]
                for branch in range(2):
                    a=soft(f'LocalSum_r{branch}','local_sum',(8,128),[exps[branch]])
                    a=soft(f'ShflSum_r{branch}','shfl_sum',(8,4),[a])
                    sums.append(soft(f'ReduceSum_r{branch}','reduce_sum',(8,4),[a]))
                sum_id=soft('MergeSum','merge_sum',(8,4),sums)
                logsum[warp]=soft('Logsum','logsum',(8,8),[sum_id,scale])
                scales[warp]=soft('AccOScale','acc_o_scale',(32,64),[scale])
                casts[warp]=soft('Cast','acc_s_cast',(8,64),exps)
                for step in range(16):
                    deps=vid+[casts[warp],scales[warp],logsum[warp]]+([last_rs[warp]] if last_rs[warp] is not None else [])
                    last_rs[warp]=emit(f'{p}_WGMMA_RS_g{g}_w{gw}_k{k}_s{step}','wgmma.m64n64k16',warp,(row,0,16,64),deps,tensor=4,dtype='fp32')
                    rs.append(last_rs[warp])
            ss_rounds.append(ss);rs_rounds.append(rs)
        divisions=[]
        for warp in range(8):
            g=warp//4;gw=cta*12+warp
            divisions.append(emit(f'{p}_AccO_Div_g{g}_w{gw}','acc_o_div',warp,(g*64,0,32,64),[last_rs[warp],logsum[warp]]+rs_rounds[-1]))
        stores=[]
        for warp in range(8):
            g=warp//4;gw=cta*12+warp
            for col in range(8):
                stores.append(emit(f'{p}_StSram_g{g}_w{gw}_c{col}','st.reg2sram',warp,(base_r+g*64+(warp%4)*16,col*16,16,16),divisions,tensor=5,ld=128))
        for g in range(2):
            emit(f'{p}_Out_Store_g{g}','cp.sram2dram_tma',8,(base_r+g*64,0,64,128),stores,tensor=(b*hq+h)*4+3,ld=128)
    kernel=ArchKernel(symbol=flow.symbol,source_hash=digest(flow),sync_semantics_version='ref-fa3-v1',tensors=tuple(tensors),ctas=tuple(ctas),ops=tuple(ops))
    kernel.verify()
    return kernel
