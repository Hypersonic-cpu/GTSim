"""Compiler passes from tensorized S-TIR to native Hopper operations.

The scalar cache-copy regions, tensorized accumulator regions, loop ordering,
thread groups and TVM-generated software pipeline are consumed as IR, not
reconstructed from a GEMM parameter tuple. No reference kernel is imported.
"""

import math
import tvm_ffi

from auto_ir import address, mutate, nodes, scalar, seq, substitute


def lower_hopper(tvm, mod, hardware):
    t = tvm.tirx
    from tvm.backend.cuda import op as cuda
    from tvm.backend.cuda.ptx import PTXNamespace
    ptx = PTXNamespace()

    def tensor_region(block, mapper):
        if any(call.op.name=='tirx.call_extern' and getattr(call.args[0],'value',None)=='gtsim.hopper.m64n64k16'
               for call in nodes(block.body,tvm.ir.Call)) and block.name_hint.endswith('_o'):
            # Preserve the tensorizer's mathematical output/reduction region
            # before buffer compaction removes the original global offsets.
            c = block.writes[0].region
            a = block.reads[1].region
            region=t.call_extern('int32','gtsim.region',c[0].min,c[1].min,a[1].min)
            body = t.AttrStmt(tvm.ir.StringImm('accumulator'), 'gtsim.wgmma_region',region,block.body)
            return tvm.s_tir.SBlock(block.iter_vars,block.reads,block.writes,block.name_hint,body,
                block.init,block.alloc_buffers,block.match_buffers,block.annotations,block.span)
        return mapper.default_mutate(block)

    persistent = int(next(node for node in nodes(mod,t.For) if node.thread_binding is not None and node.thread_binding.thread_tag=='blockIdx.x').annotations.get('gtsim.persistent',0))
    prefetch = int(next(node for node in nodes(mod,t.For) if node.thread_binding is not None and node.thread_binding.thread_tag=='blockIdx.x').annotations.get('gtsim.prefetch',0))
    mod = mutate(mod,tvm.s_tir.SBlock,tensor_region)
    stages = [tvm.s_tir.transform.PlanAndUpdateBufferAllocationLocation(),
              tvm.s_tir.transform.ConvertBlocksToOpaque(),tvm.s_tir.transform.CompactBufferAllocation(),
              tvm.s_tir.transform.LowerMatchBuffer(),tvm.s_tir.transform.InjectSoftwarePipeline(),
              tvm.s_tir.transform.LowerOpaqueBlock(),t.transform.StmtSimplify()]
    for compiler_pass in stages:
        mod = compiler_pass(mod)
    pipeline_mod = mod
    def normalize_thread(node, mapper):
        node = mapper.default_mutate(node)
        if node.attr_key=='thread_extent':
            return t.For(node.node.var,0,node.value,t.ForKind.THREAD_BINDING,node.body,
                         thread_binding=node.node,
                         annotations={'gtsim.persistent':persistent} if node.node.thread_tag=='blockIdx.x' else {})
        return node
    mod = mutate(mod,t.AttrStmt,normalize_thread)
    func = mod['main']
    block_loop = next(node for node in nodes(func.body,t.For)
                      if node.thread_binding is not None and node.thread_binding.thread_tag=='blockIdx.x')
    group_loops = [node for node in nodes(block_loop.body,t.For)
                   if node.thread_binding is not None and node.thread_binding.thread_tag=='threadIdx.y']
    groups = int(group_loops[0].extent)
    allocs = {node.buffer for node in nodes(block_loop.body,t.AllocBuffer)}
    copies = []
    for store in nodes(block_loop.body,t.BufferStore):
        if store.buffer.scope()=='shared' and isinstance(store.value,tvm.ir.TensorLoad) and store.value.source.scope()=='global':
            copies.append(store)
    if len({store.buffer for store in copies})!=2:
        raise ValueError('Hopper compiler expects two rectangular shared input caches')
    a = next(store.buffer for store in copies if store.value.source.same_as(func.params[0]))
    b = next(store.buffer for store in copies if store.value.source.same_as(func.params[1]))
    old_c = next(buffer for buffer in allocs if buffer.scope()=='local')
    bm,bn = map(int,old_c.shape)
    bk = int(a.shape[-1]); depth = int(a.shape[0]) if len(a.shape)==3 else 1
    if tuple(map(int,b.shape[-2:]))!=(bk,bn):
        raise ValueError('inconsistent shared cache regions')
    per_group = bn//groups
    if bn>256 or bm%64 or per_group%64:
        raise ValueError('unsupported native WGMMA/TMA tile geometry')
    compute_regs = bm*per_group//128+32
    shared = 2*depth*bk*(bm+bn)+2*bm*bn+32*depth
    registers = 4*(groups*128*compute_regs+(384-groups*128)*32)
    if compute_regs>255 or registers>hardware['register_file_per_sm_bytes'] or shared>min(hardware['shared_memory_per_cta_bytes'],hardware['shared_memory_per_sm_bytes']):
        raise ValueError('actual lowered allocations exceed H100 resource limits')

    # TMA swizzling must match the WGMMA canonical layouts, not merely
    # describe a dense logical matrix. A uses K-major swizzle BK*2 bytes;
    # B is packed into N=64 chunks with 128-byte N-major swizzling.
    if bk not in (16,32,64):
        raise ValueError('wide-K WGMMA layout needs an additional packing rule')
    physical_a=t.decl_buffer((depth,bm,bk),'float16',scope='shared',name='tma_a',data_alignment=1024)
    physical_b=t.decl_buffer((depth,bn//64,bk,64),'float16',scope='shared',name='tma_b',data_alignment=1024)
    swizzle_a={16:1,32:2,64:3}[bk]
    tx = t.Var('lane','int32')
    acc = t.decl_buffer((bm//64,per_group//64,32),'float32',scope='local',name='fragments')
    sc = t.decl_buffer((bm,bn),'float16',scope='shared',name='output_stage',data_alignment=128)
    full = {buffer:t.decl_buffer((depth,),'uint64',scope='shared',name=f'full_{index}') for index,buffer in enumerate((a,b))}
    empty = {buffer:t.decl_buffer((depth,),'uint64',scope='shared',name=f'empty_{index}') for index,buffer in enumerate((a,b))}
    da = t.decl_buffer((1,),'uint64',scope='local',name='descriptor_a')
    db = t.decl_buffer((1,),'uint64',scope='local',name='descriptor_b')
    maps,host = {},[]
    for tensor,box,swizzle in zip(func.params,((bm,bk),(bk,64),(bm,bn)),(swizzle_a,3,0)):
        allocation = t.tvm_stack_alloca('tensormap',1)
        descriptor = t.Var(f'map_{tensor.name}',allocation.ty)
        maps[tensor] = descriptor
        host.extend([t.Bind(descriptor,allocation),t.Evaluate(t.tensormap_encode_tiled(descriptor,tensor.data,
            tensor.shape[1],tensor.shape[0],tensor.shape[1]*2,box[1],box[0],1,1,descriptor_dtype='float16',rank=2,swizzle=swizzle))])

    def ev(call):
        return t.Evaluate(call)
    def when(condition,body):
        return t.IfThenElse(condition,body,None)
    def loop(var,extent,body):
        return t.For(var,0,extent,t.ForKind.SERIAL,body)
    def addr(buffer,indices):
        return t.address_of(t.BufferLoad(buffer,indices))
    def rewrite_group_expr(obj,var):
        return substitute(tvm,obj,{var:tx//128})

    def scalar_copy(node):
        loops=[]; leaf=node
        while isinstance(leaf,t.For) and leaf.thread_binding is None:
            loops.append(leaf); leaf=leaf.body
        if not isinstance(leaf,t.BufferStore) or not loops:
            return None
        if len(loops)!=2:
            return None
        zeros={item.loop_var:item.min for item in loops}
        if leaf.buffer in (a,b) and isinstance(leaf.value,tvm.ir.TensorLoad):
            source=leaf.value.source
            global_indices=substitute(tvm,list(leaf.value.indices),zeros)
            local_indices=substitute(tvm,list(leaf.indices),zeros)
            slot=local_indices[0] if len(local_indices)==3 else t.const(0,'int32')
            step=global_indices[1 if leaf.buffer.same_as(a) else 0]//bk
            generation=step//depth
            target=leaf.buffer
            if target.same_as(a):
                transfer=ev(ptx['cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes'](
                    addr(physical_a,[slot,*local_indices[-2:]]),maps[source],global_indices[1],global_indices[0],addr(full[target],[slot])))
            else:
                chunk=t.Var('tma_n_chunk','int32')
                transfer=loop(chunk,bn//64,ev(ptx['cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes'](
                    addr(physical_b,[slot,chunk,0,0]),maps[source],global_indices[1]+chunk*64,global_indices[0],addr(full[target],[slot]))))
            return when(tx==256,seq(tvm,[
                when(step>=depth,ev(cuda.cuda_mbarrier_wait(addr(empty[target],[slot]),(generation-1)%2))),
                ev(ptx.mbarrier.arrive.expect_tx.shared.b64(addr(full[target],[slot]),t.const(2*math.prod(map(int,target.shape[-2:])),'uint32'))),
                transfer]))
        if leaf.buffer.same_as(func.params[2]) and isinstance(leaf.value,t.Cast) and isinstance(leaf.value.value,tvm.ir.TensorLoad) and leaf.value.value.source.same_as(old_c):
            base=substitute(tvm,list(leaf.indices),zeros)
            mh,nc,part,ri=[t.Var(name,'int32') for name in ('out_m','out_n','part','register')]
            reg=part*8+ri
            row=mh*64+(tx%128//32)*16+tx%32//4+(reg%4//2)*8
            col=tx//128*per_group+nc*64+reg//4*8+tx%4*2+reg%2
            body=loop(ri,8,t.BufferStore(sc,t.Cast('float16',acc[mh,nc,reg]),[row,col]))
            body=t.AttrStmt(tvm.ir.StringImm('output'), 'gtsim.store_region',
                t.call_extern('int32','gtsim.region',base[0]+mh*64+(tx%128//32)*16,
                             base[1]+tx//128*per_group+nc*64+part*16),body)
            body=when(tx<groups*128,loop(mh,bm//64,loop(nc,per_group//64,loop(part,4,body))))
            return seq(tvm,[ev(cuda.cuda_cta_sync()),body,ev(cuda.cuda_cta_sync()),
                when(tx==256,seq(tvm,[ev(ptx.fence.proxy.async_.shared__cta()),
                    ev(ptx['cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group'](
                        maps[func.params[2]],base[1],base[0],addr(sc,[0,0]))),
                    ev(ptx.cp.async_.bulk.commit_group()),ev(ptx.cp.async_.bulk.wait_group(0))])),
                ev(cuda.cuda_cta_sync())])
        return None

    def lower_mma(attr, mapper):
        if attr.attr_key!='gtsim.wgmma_region':
            return mapper.default_mutate(attr)
        call=attr.body.value
        aa,ai=address(tvm,call.args[1]);bb,bi=address(tvm,call.args[2]);cc,ci=address(tvm,call.args[3])
        if not aa.same_as(a) or not bb.same_as(b) or not cc.same_as(old_c):
            raise ValueError('tensorized operands do not match staged buffers')
        mi=ci[0]//64; ni=(ci[1]%per_group)//64
        mma=ptx['wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16'](
            *[acc[mi,ni,index] for index in range(32)],da[0],db[0],True,1,1,0,1)
        return seq(tvm,[ev(cuda.cuda_wgmma_encode_matrix_descriptor(addr(da,[0]),addr(physical_a,[ai[0] if len(ai)==3 else 0,*ai[-2:]]),1,bk,swizzle_a)),
            ev(cuda.cuda_wgmma_encode_matrix_descriptor(addr(db,[0]),addr(physical_b,[bi[0] if len(bi)==3 else 0,bi[-1]//64,bi[-2],bi[-1]%64]),1,64,3)),
            t.AttrStmt(attr.node,'gtsim.wgmma_region',attr.value,ev(mma))])

    def transform_loop(node, mapper):
        copy=scalar_copy(node)
        if copy is not None:
            return copy
        if node.thread_binding is not None and node.thread_binding.thread_tag=='threadIdx.y':
            tags=[tag for tag in nodes(node.body,t.AttrStmt) if tag.attr_key=='gtsim.wgmma_region']
            if not tags:
                writes=nodes(node.body,t.BufferStore)
                if not writes or any(not store.buffer.same_as(old_c) or type(store.value).__name__!='FloatImm' or float(store.value.value)!=0 for store in writes):
                    raise ValueError('unsupported accumulator initialization')
                # Tensorized accumulator initialization, distributed by the
                # same register layout used by the instruction intrinsic.
                mh,nc,ri=[t.Var(name,'int32') for name in ('zero_m','zero_n','zero_r')]
                return when(tx<groups*128,loop(mh,bm//64,loop(nc,per_group//64,loop(ri,32,
                    t.BufferStore(acc,t.const(0,'float32'),[mh,nc,ri])))))
            group=rewrite_group_expr(node.body,node.loop_var)
            tags=[tag for tag in nodes(group,t.AttrStmt) if tag.attr_key=='gtsim.wgmma_region']
            first=tags[0]
            zeros={item.loop_var:item.min for item in nodes(group,t.For)}
            kbase=substitute(tvm,first.value.args[3],zeros)
            stage=(kbase//bk)%depth
            phase=(kbase//bk//depth)%2
            group=mutate(group,t.AttrStmt,lower_mma)
            return when(tx<groups*128,seq(tvm,[
                ev(cuda.cuda_mbarrier_wait(addr(full[a],[stage]),phase)),
                ev(cuda.cuda_mbarrier_wait(addr(full[b],[stage]),phase)),
                ev(ptx.wgmma.fence.sync.aligned()),group,
                ev(ptx.wgmma.commit_group.sync.aligned()),ev(ptx.wgmma.wait_group.sync.aligned(0)),
                when(tx%128==0,seq(tvm,[ev(ptx.mbarrier.arrive.shared.b64(addr(empty[a],[stage]))),
                                      ev(ptx.mbarrier.arrive.shared.b64(addr(empty[b],[stage])))]))]))
        return mapper.default_mutate(node)

    body=mutate(block_loop.body,t.For,transform_loop)
    body=mutate(body,t.AllocBuffer,lambda node,mapper:t.Evaluate(0) if any(node.buffer.same_as(buffer) for buffer in (old_c,a,b)) else node)
    slot=t.Var('init_slot','int32')
    init=[]
    invalidate=[]
    for buffer in (a,b):
        init.extend([ev(ptx.mbarrier.init.shared.b64(addr(full[buffer],[slot]),t.const(1,'uint32'))),
                     ev(ptx.mbarrier.init.shared.b64(addr(empty[buffer],[slot]),t.const(groups,'uint32')))])
        invalidate.extend([ev(ptx.mbarrier.inval.shared.b64(addr(full[buffer],[slot]))),
                           ev(ptx.mbarrier.inval.shared.b64(addr(empty[buffer],[slot])))])
    new_buffers=[physical_a,physical_b,acc,sc,da,db,*full.values(),*empty.values()]
    body=seq(tvm,[*[t.AllocBuffer(buffer) for buffer in new_buffers],
        when(tx>=groups*128,ev(ptx.setmaxnreg.dec.sync.aligned.u32(32))),
        when(tx<groups*128,ev(ptx.setmaxnreg.inc.sync.aligned.u32(compute_regs))),
        when(tx==256,loop(slot,depth,seq(tvm,init))),ev(ptx.fence.proxy.async_.shared__cta()),
        ev(cuda.cuda_cta_sync()),body,
        # The epilogue's CTA join proves every consumer and output DMA has
        # completed. PTX requires invalidation before initializing these
        # shared addresses as new barriers on the next persistent tile.
        when(tx==256,loop(slot,depth,seq(tvm,invalidate)))])
    count=int(block_loop.extent)
    persistent=int(block_loop.annotations.get('gtsim.persistent',0))
    if persistent:
        bx,wave=t.Var('cta','int32'),t.Var('wave','int32')
        grid=min(count,hardware['num_sms'])
        tile=wave*grid+bx
        body=substitute(tvm,body,{block_loop.loop_var:tile})
        body=loop(wave,(count+grid-1)//grid,when(tile<count,body))
    else:
        bx,grid=block_loop.loop_var,count
    body=t.For(tx,0,384,t.ForKind.THREAD_BINDING,body,thread_binding=t.IterVar((0,384),tx,t.IterVar.ThreadIndex,'threadIdx.x'))
    body=t.For(bx,0,grid,t.ForKind.THREAD_BINDING,body,thread_binding=t.IterVar((0,grid),bx,t.IterVar.ThreadIndex,'blockIdx.x'))
    func=t.PrimFunc(func.params,seq(tvm,host+[body])).with_attr('global_symbol','main')
    func=func.with_attr('target',tvm.target.Target({'kind':'cuda','arch':'sm_90a','max_num_threads':1024,'thread_warp_size':32}))
    func=func.with_attr('gtsim.compiler','hopper_intrinsics_v1').with_attr('gtsim.resources',[shared,registers,12])
    func=func.with_attr('gtsim.num_sms',hardware['num_sms'])
    result=tvm.IRModule({'main':func})
    for compiler_pass in (t.transform.ConvertSSA(),t.transform.StmtSimplify()):
        result=compiler_pass(result)
    if not t.analysis.verify_ssa(result['main']) or not t.analysis.verify_well_formed(result['main']):
        raise ValueError('native Hopper IR verification failed')
    evidence=dict(bm=bm,bn=bn,bk=bk,stages=depth,groups=groups,persistent=persistent,
                  shared_bytes=shared,register_bytes=registers,grid=grid,output_tiles=count)
    from tier_memory import place_module
    result,placement=place_module(tvm,result,hardware,prefetch,bk)
    evidence.update(placement)
    return result,pipeline_mod,evidence
