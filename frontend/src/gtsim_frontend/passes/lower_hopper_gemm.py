"""Expand scheduled logical GEMM tiles into Hopper warp instructions.

No legacy builder, graph fixture, timing parameter, or TVM import is used.
"""
from collections import defaultdict
from ..ir.arch import ArchKernel, ArchInstruction, CTA, KernelTensor, HOPPER_OPS
from ..serialization.json_codec import digest


def lower(flow):
    flow.verify()
    if flow.family!='gemm' or flow.schedule_id!='ref_h100_gemm_v0_1_0':
        raise ValueError('unsupported GEMM schedule')
    allowed={'','zero_accumulator','fp16_matmul_fp32_accumulate','cast_fp16_store'}
    if any(o.semantic not in allowed for o in flow.ops):raise ValueError('unsupported GEMM Flow operation')
    ops=[];ctas=[]
    tensors={t.id:t for t in flow.tensors}
    tiles=defaultdict(dict)
    for op in flow.ops:
        prefix=op.id.split('.')
        if len(prefix)>2 and prefix[0].startswith('cta') and prefix[1].startswith('tile'):
            tiles[int(prefix[0][3:])].setdefault(int(prefix[1][4:]),[]).append(op)
    for cta,sm in flow.mapping.ctas:
        ctas.append(CTA(cta,sm,12,2*2*64*(128+256)+128*256*2,4*128*256+12*32*32*4))
        previous_stores=[];previous_loads=[];previous_compute=[]
        def emit(name,op,warp,tensor,tile,complete=(),issue=(),dtype='fp16',ld=0):
            nid=len(ops)
            ops.append(ArchInstruction(nid,name,op,HOPPER_OPS[op],cta,cta*12+warp,tensor,tuple(tile),dtype,ld,tuple(issue),tuple(complete)))
            return nid
        for tile_id,logical in sorted(tiles[cta].items()):
            prefix=f'TB{cta}_Tile{tile_id}'
            loads_a=[o for o in logical if o.id.endswith('.A')]
            loads_b=[o for o in logical if o.id.endswith('.B')]
            matrices=[o for o in logical if o.capability=='matmul']
            stores=[o for o in logical if o.capability=='store_async']
            if len(loads_a)!=len(loads_b) or len(matrices)!=len(loads_a) or len(stores)!=1:
                raise ValueError(f'{prefix}: malformed GEMM flow stages')
            output=stores[0].writes[0]
            r,c=output.offsets
            if output.shape!=(128,256):raise ValueError('unsupported GEMM output geometry')
            initial=[o for o in logical if o.semantic=='zero_accumulator']
            if len(initial)!=1 or initial[0].results[0].shape!=(128,256):raise ValueError('missing GEMM accumulator initialization')
            accumulator=initial[0].results[0].id
            stage_loads=[];compute=[];last=[None]*8
            for k,(a,b,matmul) in enumerate(zip(loads_a,loads_b,matrices)):
                if matmul.operands!=(a.results[0].id,b.results[0].id,accumulator) or matmul.results[0].shape!=(128,256):raise ValueError('invalid GEMM accumulator/operand contract')
                accumulator=matmul.results[0].id
                if a.reads[0].shape!=(128,64) or b.reads[0].shape!=(64,256):raise ValueError('invalid GEMM operand coverage')
                deps=[]
                if k>=2:deps=compute[k-2]
                elif previous_compute and (k==0 or len(previous_compute)%2==0):
                    prior=max((j for j in range(len(previous_compute)) if j%2==k%2),default=-1)
                    if prior>=0:deps=previous_compute[prior]
                issue=stage_loads[-1] if stage_loads else previous_loads
                ar,ak=a.reads[0].offsets;bk,bc=b.reads[0].offsets
                if ar!=r or bc!=c or ak!=bk:raise ValueError('inconsistent GEMM tile regions')
                current=[emit(f'{prefix}_TMA_LoadA_k{k}','cp.dram2sram_tma',8,0,(ar,ak,128,64),deps,issue,ld=tensors['A'].shape[1])]
                current.extend(emit(f'{prefix}_TMA_LoadB_k{k}_c{chunk}','cp.dram2sram_tma',8,1,(bk,bc+chunk*64,64,64),deps,issue,ld=tensors['B'].shape[1]) for chunk in range(4))
                stage_loads.append(current);current_compute=[]
                for warp in range(8):
                    for half in range(2):
                        for nc in range(2):
                            chunk=(warp//4)*2+nc
                            for step in range(4):
                                ready=list(current)
                                if last[warp] is not None:ready.append(last[warp])
                                if k==0:ready.extend(previous_stores)
                                nid=emit(f'{prefix}_WGMMA_w{cta*12+warp}_k{k}_mh{half}_n{chunk}_ks{step}',
                                    'wgmma.m64n64k16',warp,2,(r+half*64+(warp%4)*16,c+chunk*64,16,64),ready,dtype='fp32')
                                current_compute.append(nid);last[warp]=nid
                compute.append(current_compute)
            if stores[0].operands!=(accumulator,):raise ValueError("store does not consume final accumulator")
            staging=[]
            for warp in range(8):
                for half in range(2):
                    for nc in range(8):
                        staging.append(emit(f'{prefix}_StSram_w{cta*12+warp}_mh{half}_n{nc}',
                            'st.reg2sram',warp,2,(r+half*64+(warp%4)*16,c+(warp//4)*128+nc*16,16,16),[last[warp]]+compute[-1]))
            previous_stores=[]
            for chunk in range(4):
                nid=emit(f'{prefix}_TMA_StoreC_c{chunk}','cp.sram2dram_tma',8,2,(r,c+chunk*64,128,64),staging,
                         previous_stores[-1:] if previous_stores else (),ld=tensors['C'].shape[1])
                previous_stores.append(nid)
            previous_loads=stage_loads[-1];previous_compute=compute
    kernel=ArchKernel(symbol=flow.symbol,source_hash=digest(flow),sync_semantics_version='pingpong-cross-wave-v2',tensors=tuple(KernelTensor(i,t.id,t.shape,(t.shape[1],1),preloaded=(i==1),read_only=(i==1)) for i,t in enumerate(flow.tensors)),ctas=tuple(ctas),ops=tuple(ops),outputs=(2,))
    kernel.verify()
    return kernel
