"""Compiler-owned device placement and synchronized cross-tier copy launches."""

from dataclasses import replace

from auto_ir import nodes, seq, substitute
from auto_rules import validate_placement
from tile_ir import TileIR


def place_module(tvm, module, hardware, prefetch, copy_rows):
    t=tvm.tirx
    original=module['main']
    m,k=map(int,original.params[0].shape);n=int(original.params[1].shape[1])
    allocated=validate_placement((m,n,k),hardware,prefetch)
    hybrid=hardware['topology']!='FullHBM'
    tiers=['HBM','HBF' if hybrid else 'HBM','HBM']
    params=[t.decl_buffer(p.shape,str(p.dtype),name=str(p.name),scope='global.'+tier.lower())
            for p,tier in zip(original.params,tiers)]
    mapping=dict(zip(original.params,params))
    if prefetch:
        cached=t.decl_buffer((k,n),'float16',name='B_hbm',scope='global.hbm')
        mapping[original.params[1]]=cached
    gemm=substitute(tvm,original,mapping)
    evidence=dict(tensor_tiers=tiers,prefetch_strategy='whole_B_tma' if prefetch else 'direct',
                  allocated_bytes=allocated,capacity_known=all(hardware.get(x.lower()+'_capacity_bytes',0)>0
                  for x in allocated if allocated[x]),migration_payload_bytes=2*k*n if prefetch else 0)
    if not prefetch:
        return tvm.IRModule({'main':gemm}),evidence
    copy=copy_kernel(tvm,params[1],cached,copy_rows)
    gemm=gemm.with_attr('global_symbol','gemm')
    # Explicit runtime launch/wait operations represent two stream-ordered
    # device kernels. The DAG lowerer interprets these operations, never
    # inferring a free placement change from a tensor annotation.
    host=t.PrimFunc([*params,cached],seq(tvm,[
        t.Evaluate(t.call_extern('int32','gtsim.launch_kernel','prefetch',params[1].data,cached.data)),
        t.Evaluate(t.call_extern('int32','gtsim.stream_synchronize')),
        t.Evaluate(t.call_extern('int32','gtsim.launch_kernel','gemm',params[0].data,cached.data,params[2].data)),
        t.Evaluate(t.call_extern('int32','gtsim.stream_synchronize')),
    ])).with_attr('global_symbol','main').with_attr('gtsim.compiler','tier_launch_v1').with_attr('target',gemm.attrs['target'])
    return tvm.IRModule({'main':host,'prefetch':copy,'gemm':gemm}),evidence


def copy_kernel(tvm, source, destination, rows):
    """Lower a compiler identity-copy stage to TMA through shared memory.

    Each CTA copies one rectangular box. Its size follows the sampled BK and
    the supported 128-byte TMA box width; no GEMM computation is constructed.
    """
    t=tvm.tirx
    from tvm.backend.cuda import op as cuda
    from tvm.backend.cuda.ptx import PTXNamespace
    ptx=PTXNamespace()
    k,n=map(int,source.shape)
    if tuple(map(int,destination.shape))!=(k,n) or k%rows or n%64:
        raise ValueError('unsupported rectangular tier-copy region')
    stage=t.decl_buffer((rows,64),'float16',scope='shared',name='tier_stage',data_alignment=128)
    barrier=t.decl_buffer((1,),'uint64',scope='shared',name='tier_full')
    maps=[];host=[]
    for buffer in (source,destination):
        allocation=t.tvm_stack_alloca('tensormap',1)
        descriptor=t.Var('map_'+str(buffer.name),allocation.ty);maps.append(descriptor)
        host.extend([t.Bind(descriptor,allocation),t.Evaluate(t.tensormap_encode_tiled(
            descriptor,buffer.data,n,k,n*2,64,rows,1,1,descriptor_dtype='float16',rank=2,swizzle=0))])
    bx,tx=t.Var('copy_cta','int32'),t.Var('copy_lane','int32')
    kk=(bx//(n//64))*rows;nn=(bx%(n//64))*64
    pointer=lambda buffer:t.address_of(t.BufferLoad(buffer,[0]*len(buffer.shape)))
    ev=t.Evaluate
    leader=lambda body:t.IfThenElse(tx==256,body,None)
    body=seq(tvm,[t.AllocBuffer(stage),t.AllocBuffer(barrier),
        ev(ptx.setmaxnreg.dec.sync.aligned.u32(32)),
        leader(ev(ptx.mbarrier.init.shared.b64(pointer(barrier),t.const(1,'uint32')))),
        ev(ptx.fence.proxy.async_.shared__cta()),ev(cuda.cuda_cta_sync()),
        leader(seq(tvm,[
            ev(ptx.mbarrier.arrive.expect_tx.shared.b64(pointer(barrier),t.const(rows*64*2,'uint32'))),
            ev(ptx['cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes'](
                pointer(stage),maps[0],nn,kk,pointer(barrier))),
            ev(cuda.cuda_mbarrier_wait(pointer(barrier),0)),
            ev(ptx.fence.proxy.async_.shared__cta()),
            ev(ptx['cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group'](maps[1],nn,kk,pointer(stage))),
            ev(ptx.cp.async_.bulk.commit_group()),ev(ptx.cp.async_.bulk.wait_group(0)),
            ev(ptx.mbarrier.inval.shared.b64(pointer(barrier))),
        ])),ev(cuda.cuda_cta_sync())])
    body=t.For(tx,0,384,t.ForKind.THREAD_BINDING,body,thread_binding=t.IterVar((0,384),tx,t.IterVar.ThreadIndex,'threadIdx.x'))
    grid=(k//rows)*(n//64)
    body=t.For(bx,0,grid,t.ForKind.THREAD_BINDING,body,thread_binding=t.IterVar((0,grid),bx,t.IterVar.ThreadIndex,'blockIdx.x'))
    return t.PrimFunc([source,destination],seq(tvm,[*host,body])).with_attr('global_symbol','prefetch').with_attr(
        'target',tvm.target.Target({'kind':'cuda','arch':'sm_90a'})).with_attr(
        'gtsim.compiler','hopper_intrinsics_v1').with_attr('gtsim.kernel_role','tier_copy').with_attr(
        'gtsim.resources',[rows*64*2+8,12*32*32*4,12])


def lower_module(tvm,module,hardware,inputs=None):
    from native_dag import lower_native
    if module['main'].attrs.get('gtsim.compiler')!='tier_launch_v1':
        return lower_native(tvm,module['main'],hardware,inputs)
    host=module['main']
    statements=list(host.body.seq)
    expected=('gtsim.launch_kernel','gtsim.stream_synchronize','gtsim.launch_kernel','gtsim.stream_synchronize')
    if len(statements)!=4 or any(not isinstance(stmt,tvm.tirx.Evaluate) or
        stmt.value.op.name!='tirx.call_extern' or str(stmt.value.args[0].value)!=name
        for stmt,name in zip(statements,expected)):
        raise ValueError('unsupported or unsynchronized device launch sequence')
    if [str(statements[i].value.args[1].value) for i in (0,2)]!=['prefetch','gemm']:
        raise ValueError('unsupported device launch order')
    for index,name in ((0,'prefetch'),(2,'gemm')):
        args=statements[index].value.args[2:]
        if len(args)!=len(module[name].params) or any(not isinstance(arg,tvm.ir.Call) or not arg.args[0].same_as(param) for arg,param in zip(args,module[name].params)):
            raise ValueError('device launch buffer binding mismatch')
    source,cached=module['prefetch'].params
    ga,gb,gc=module['gemm'].params
    if not cached.same_as(gb) or source.scope()!='global.hbf' or cached.scope()!='global.hbm':
        raise ValueError('cross-tier prefetch has inconsistent device buffers')
    copied,copy_evidence=lower_native(tvm,module['prefetch'],hardware)
    computed,evidence=lower_native(tvm,module['gemm'],hardware,inputs)
    ir=TileIR()
    # Keep the public A/B/C identities stable, with a distinct HBM allocation
    # for the initialized copy of B. Copy-kernel and GEMM IDs are remapped.
    for ident,buffer in enumerate((ga,source,gc,cached)):
        shape=tuple(map(int,buffer.shape))
        ir.tensor(ident,str(buffer.name),shape,(shape[1],1),tier='HBF' if ident==1 else 'HBM')
    for cta in copied.ctas:ir.cta(*cta)
    for op in copied.operations:
        ir.add(replace(op,tensor={0:1,1:3}[op.tensor]))
    terminal=[i for i,op in enumerate(ir.operations) if op.op=='cp.sram2dram_tma']
    op_offset=len(ir.operations);cta_offset=len(ir.ctas)
    for ident,sm,shared,registers in computed.ctas:ir.cta(ident+cta_offset,sm,shared,registers)
    for op in computed.operations:
        complete=[i+op_offset for i in op.after_complete]
        if not op.after_issue and not complete:complete=terminal.copy()
        ir.add(replace(op,name='gemm_'+op.name,cta=op.cta+cta_offset,tensor={0:0,1:3,2:2}[op.tensor],
                       after_issue=[i+op_offset for i in op.after_issue],after_complete=complete))
    evidence.update(prefetch_nodes=len(copied.operations),prefetch_ctas=len(copied.ctas),
                    migration_payload_bytes=int(source.shape[0])*int(source.shape[1])*2,
                    launch_synchronization=True,nodes=len(ir.operations))
    return ir,evidence
