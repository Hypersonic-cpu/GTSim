"""Operation-driven Hopper TIR interpreter producing GTSim's existing Tile IR.

No schedule template comparison: loops/guards, per-warp program order, mbarrier
phases/transactions, asynchronous groups and CTA joins determine DAG edges.
Compiler region bindings retain tensorization's output/reduction coordinates
through distributed register lowering. Unknown operations/layouts fail closed.
"""

from collections import defaultdict
import math

import numpy as np

from auto_ir import address, nodes
from tile_ir import TileIR, Operation


def lower_native(tvm, func, hardware, inputs=None):
    t=tvm.tirx
    if not func.attrs or func.attrs.get('gtsim.compiler')!='hopper_intrinsics_v1':
        raise ValueError('missing supported compiler buffer-layout ABI')
    if len(func.params)!=3 or any(str(p.dtype)!='float16' or len(p.shape)!=2 for p in func.params):
        raise ValueError('only rank-2 FP16 GEMM buffers are supported')
    if any(p.layout is not None for p in func.params):
        raise ValueError('only contiguous tensor-map source layouts are supported')
    a,b,c=func.params
    m,k=map(int,a.shape); kb,n=map(int,b.shape)
    if k!=kb or tuple(map(int,c.shape))!=(m,n):
        raise ValueError('inconsistent GEMM tensor dimensions')
    block=next(node for node in nodes(func.body,t.For) if node.thread_binding is not None and node.thread_binding.thread_tag=='blockIdx.x')
    grid=int(block.extent)
    shared_buffers={node.buffer for node in nodes(block.body,t.AllocBuffer) if node.buffer.scope()=='shared'}
    shared=sum(math.prod(map(int,buffer.shape))*(8 if str(buffer.dtype)=='uint64' else 2) for buffer in shared_buffers)
    budgets={}
    maps={}
    for call in nodes(func.body,tvm.ir.Call):
        if call.op.name=='tirx.tensormap_encode_tiled':
            descriptor=call.args[0]
            tensor=next((i for i,p in enumerate(func.params) if isinstance(call.args[1],tvm.ir.Call) and call.args[1].args[0].same_as(p)),None)
            if tensor is None or int(call.args[7])!=1 or int(call.args[8])!=1:
                raise ValueError('unsupported tensor-map source or element stride')
            source=func.params[tensor]
            if tuple(map(int,call.args[2:5]))!=(int(source.shape[1]),int(source.shape[0]),2*int(source.shape[1])):
                raise ValueError('tensor-map dimensions/byte stride disagree with its source')
            attrs=call.attrs
            if (str(attrs.descriptor_dtype),int(attrs.rank),int(attrs.interleave),int(attrs.l2_promotion),int(attrs.oob_fill),int(attrs.force_cu_dtype))!=('float16',2,0,0,0,-1):
                raise ValueError('unsupported tensor-map encoding attributes')
            maps[descriptor]=(tensor,int(call.args[6]),int(call.args[5]),int(call.attrs['swizzle']))
    if len(maps)!=3:
        raise ValueError('expected input/output tensor maps')

    # Compile bounded index expressions once, so graph expansion does not
    # repeatedly cross the Python/C++ FFI for every node and CTA.
    variables={}; expressions={}
    def variable(var):
        if var not in variables:
            variables[var]=len(variables)
        return variables[var]
    def expression(expr):
        if isinstance(expr,t.IntImm):
            val=int(expr);return lambda env:val
        if isinstance(expr,tvm.ir.Var):
            index=variable(expr);return lambda env:env[index]
        if expr in expressions:
            return expressions[expr]
        functions={'Add':lambda a,b:a+b,'Sub':lambda a,b:a-b,'Mul':lambda a,b:a*b,
                   'FloorDiv':lambda a,b:a//b,'FloorMod':lambda a,b:a%b,
                   'LT':lambda a,b:a<b,'LE':lambda a,b:a<=b,'GE':lambda a,b:a>=b,'GT':lambda a,b:a>b,
                   'EQ':lambda a,b:a==b,'NE':lambda a,b:a!=b,'And':lambda a,b:a and b,'Or':lambda a,b:a or b,
                   'BitwiseAnd':lambda a,b:a&b,'BitwiseOr':lambda a,b:a|b,'Min':min,'Max':max}
        name=type(expr).__name__
        if name in functions:
            aa,bb=expression(expr.a),expression(expr.b);op=functions[name]
            result=lambda env:op(aa(env),bb(env))
        else:
            raise ValueError(f'unsupported native index expression: {expr}')
        expressions[expr]=result
        return result
    def coords(items):
        functions=[expression(item) for item in items]
        return lambda env:tuple(fn(env) for fn in functions)
    def pointer(expr):
        buffer,indices=address(tvm,expr);evaluate=coords(indices)
        return lambda env:(buffer,evaluate(env))
    def event(kind,payload):
        def run(env):
            yield kind,payload(env)
        return run
    def noop(env):
        return
        yield
    def compile_stmt(stmt,region=None):
        if isinstance(stmt,t.SeqStmt):
            children=[compile_stmt(child,region) for child in stmt.seq]
            def run(env):
                for child in children:
                    yield from child(env)
            return run
        if isinstance(stmt,t.For):
            index=variable(stmt.loop_var)
            if stmt.thread_binding is not None:
                tag=stmt.thread_binding.thread_tag
                if tag=='threadIdx.x':
                    if int(stmt.extent)!=384:raise ValueError('the current Tile IR ABI requires 12 warps')
                    thread_vars.append(index)
                    return compile_stmt(stmt.body,region)
                raise ValueError(f'unsupported nested binding: {tag}')
            body=compile_stmt(stmt.body,region)
            minimum,extent=expression(stmt.min),expression(stmt.extent)
            def run(env):
                for i in range(minimum(env),minimum(env)+extent(env)):
                    env[index]=i
                    yield from body(env)
            return run
        if isinstance(stmt,t.IfThenElse):
            condition=expression(stmt.condition);yes=compile_stmt(stmt.then_case,region)
            no=compile_stmt(stmt.else_case,region) if stmt.else_case is not None else noop
            uses_thread=any(var in variables and variables[var] in thread_vars for var in nodes(stmt.condition,tvm.ir.Var))
            # Warp actors represent all 32 lanes. Only supported scalar
            # producer/barrier operations may execute on a single leader.
            leader_ops={'tirx.ptx.mbarrier_init','tirx.ptx.mbarrier_inval',
                        'tirx.ptx.mbarrier_arrive_expect_tx','tirx.ptx.mbarrier_arrive_nocount',
                        'tirx.cuda.mbarrier_wait','tirx.ptx.cp_async_bulk_tensor_g2s_cluster',
                        'tirx.ptx.cp_async_bulk_tensor_s2g','tirx.ptx.cp_async_bulk_commit_group',
                        'tirx.ptx.cp_async_bulk_wait_group','tirx.ptx.fence_proxy'}
            leader_only=stmt.else_case is None and all(
                isinstance(item.value,tvm.ir.Call) and item.value.op.name in leader_ops
                for item in nodes(stmt.then_case,t.Evaluate)) and not nodes(stmt.then_case,t.BufferStore)
            def run(env):
                take=condition(env)
                if uses_thread:
                    index=thread_vars[0];local=env.copy();mask=[]
                    for lane in range(env[index],env[index]+32):
                        local[index]=lane;mask.append(bool(condition(local)))
                    if len(set(mask))>1 and (not leader_only or mask!=[True]+[False]*31):
                        raise ValueError('unsupported lane-divergent instruction control flow')
                yield from (yes if take else no)(env)
            return run
        if isinstance(stmt,t.AllocBuffer):
            return noop
        if isinstance(stmt,t.AttrStmt):
            if stmt.attr_key not in ('gtsim.wgmma_region','gtsim.store_region'):
                raise ValueError(f'unsupported native annotation {stmt.attr_key}')
            if not isinstance(stmt.value,tvm.ir.Call) or getattr(stmt.value.args[0],'value',None)!='gtsim.region':
                raise ValueError('invalid compiler region binding')
            region=coords(stmt.value.args[1:])
            if stmt.attr_key=='gtsim.store_region':
                stores=nodes(stmt.body,t.BufferStore)
                if len(stores)!=1 or stores[0].buffer.scope()!='shared' or not isinstance(stores[0].value,t.Cast):
                    raise ValueError('unsupported scalar epilogue under tile region')
                store=stores[0];buffer=store.buffer
                scalar_loops=nodes(stmt.body,t.For)
                if len(scalar_loops)!=1 or int(scalar_loops[0].extent)!=8:
                    raise ValueError('unsupported distributed epilogue fragment')
                register_index=variable(scalar_loops[0].loop_var)
                spatial=coords(store.indices)
                source=store.value.value
                if not isinstance(source,tvm.ir.TensorLoad):raise ValueError('epilogue must cast accumulator registers')
                registers=coords(source.indices);source_buffer=source.source
                def payload(env):
                    warp=env[thread_vars[0]]//32
                    local=env.copy();local[thread_vars[0]]=np.repeat(np.arange(warp*32,warp*32+32),8)
                    local[register_index]=np.tile(np.arange(8),32)
                    return region(env),buffer,spatial(local),(source_buffer,registers(local))
                return event('stage',payload)
            return compile_stmt(stmt.body,region)
        if isinstance(stmt,t.BufferStore):
            if stmt.buffer.scope()=='local' and type(stmt.value).__name__=='FloatImm' and float(stmt.value)==0:
                indices=coords(stmt.indices);buffer=stmt.buffer
                return event('zero',lambda env:(buffer,indices(env)))
            raise ValueError('unlowered scalar memory operation')
        if isinstance(stmt,t.Evaluate):
            if isinstance(stmt.value,t.IntImm) and int(stmt.value)==0:return noop
            call=stmt.value
            if not isinstance(call,tvm.ir.Call):raise ValueError('unsupported Evaluate')
            name=call.op.name;args=call.args
            if name=='tirx.cuda.cta_sync':return event('sync',lambda env:None)
            if name=='tirx.ptx.mbarrier_init':
                ptr=pointer(args[0]);count=expression(args[1])
                return event('init',lambda env:(ptr(env),count(env)))
            if name=='tirx.ptx.mbarrier_inval':
                ptr=pointer(args[0]);return event('invalidate',lambda env:ptr(env))
            if name=='tirx.ptx.mbarrier_arrive_expect_tx':
                ptr=pointer(args[0]);count=expression(args[1])
                return event('expect',lambda env:(ptr(env),count(env)))
            if name=='tirx.ptx.mbarrier_arrive_nocount':
                ptr=pointer(args[0]);return event('arrive',lambda env:ptr(env))
            if name=='tirx.cuda.mbarrier_wait':
                ptr=pointer(args[0]);phase=expression(args[1])
                return event('wait',lambda env:(ptr(env),phase(env)))
            if name=='tirx.ptx.cp_async_bulk_tensor_g2s_cluster':
                dst=pointer(args[0]);descriptor=args[1];xy=coords(args[2:4]);bar=pointer(args[4])
                return event('load',lambda env:(dst(env),descriptor,xy(env),bar(env)))
            if name=='tirx.ptx.cp_async_bulk_tensor_s2g':
                descriptor=args[0];xy=coords(args[1:3]);src=pointer(args[3])
                return event('store',lambda env:(src(env),descriptor,xy(env)))
            if name=='tirx.cuda.wgmma_encode_matrix_descriptor':
                dst=pointer(args[0]);src=pointer(args[1]);layout=coords(args[2:5])
                return event('descriptor',lambda env:(dst(env),src(env),layout(env)))
            if name=='tirx.ptx.wgmma_f16_ss':
                if region is None:raise ValueError('WGMMA output region was lost during register lowering')
                regs=[pointer(t.address_of(arg)) for arg in args[:32]]
                descs=[pointer(t.address_of(arg)) for arg in args[32:34]]
                flags=coords(args[34:39])
                return event('mma',lambda env:(region(env),[fn(env) for fn in regs],[fn(env) for fn in descs],flags(env)))
            if name=='tirx.ptx.wgmma_commit_group':return event('mma_commit',lambda env:None)
            if name=='tirx.ptx.cp_async_bulk_commit_group':return event('bulk_commit',lambda env:None)
            if name=='tirx.ptx.wgmma_fence':return event('mma_fence',lambda env:None)
            if name=='tirx.ptx.fence_proxy':return event('proxy_fence',lambda env:None)
            if name=='tirx.ptx.wgmma_wait_group':
                if int(args[0])!=0:raise ValueError('partial WGMMA wait groups are unsupported')
                return event('mma_wait',lambda env:None)
            if name=='tirx.ptx.cp_async_bulk_wait_group':
                if int(args[0])!=0 or any(getattr(arg,'value',None)=='read' for arg in args):
                    raise ValueError('only full-completion bulk.wait_group(0) is supported')
                return event('bulk_wait',lambda env:None)
            if name=='tirx.ptx.setmaxnreg':
                amount=expression(args[0]);return event('registers',lambda env:amount(env))
            raise ValueError(f'unsupported native operation {name}')
        raise ValueError(f'unsupported native statement {type(stmt).__name__}')

    thread_vars=[]
    block_var=variable(block.loop_var)
    program=compile_stmt(block.body)
    ir=TileIR()
    for i,p in enumerate(func.params):
        shape=tuple(map(int,p.shape));ir.tensor(i,str(p.name),shape,(shape[1],1))
    arithmetic=defaultdict(int);staged_coverage=set();stored_coverage=set()
    numerical_output=np.empty((m,n),dtype='float16') if inputs is not None else None
    input_a,input_b=(np.asarray(x,dtype='float32') for x in inputs) if inputs is not None else (None,None)
    statistics=defaultdict(int)
    max_registers=0

    for cta in range(grid):
        base=len(ir.operations);before=[];collectives=defaultdict(list);shared_state={};barriers={};epoch=defaultdict(int)
        descriptors={};stage_nodes=[];output_reads=defaultdict(list)
        actors=[]
        for warp in range(12):
            env=[0]*len(variables);env[block_var]=cta
            for index in thread_vars:env[index]=warp*32
            actor=dict(warp=warp,stream=iter(program(env)),frontier=set(),known=0,last_issue=None,
                       pending_mma=[],committed_mma=[],pending_bulk=[],committed_bulk=[],mma_fenced=False,proxy_known=0,pending_scalar=[],last_acc={},zeroed=set(),waits=defaultdict(int),
                       event=None,done=False,registers=None,numeric={})
            actors.append(actor)
        def closure(ident):return before[ident-base] | (1<<(ident-base))
        def complete(actor,ids):
            combined=actor['frontier']|set(ids)
            for ident in ids:actor['known']|=closure(ident)
            ancestor=0
            for ident in combined:ancestor|=before[ident-base]
            actor['frontier']={ident for ident in combined if not ancestor & (1<<(ident-base))}
        def emit(actor,op,tensor,region,extra=(),dtype='fp16'):
            deps=actor['frontier']|set(extra)
            issue=[] if actor['last_issue'] is None else [actor['last_issue']]
            ident=len(ir.operations)
            mask=0
            for parent in deps:mask|=closure(parent)
            for parent in issue:mask|=before[parent-base]
            before.append(mask)
            ir.add(Operation(f'CTA{cta}_op{ident}',op,cta,actor['warp'],tensor,tuple(region),dtype,sorted(deps),issue))
            actor['last_issue']=ident
            return ident
        def known(actor,ids):
            return all(actor['known'] & (1<<(ident-base)) for ident in ids)
        def validate_region(tensor,region):
            row,col,rows,cols=region;shape=ir.tensors[tensor][2]
            if min(row,col)<0 or min(rows,cols)<=0 or row+rows>shape[0] or col+cols>shape[1]:
                raise ValueError(f'out-of-bounds native access {region}')
        def barrier(key):
            if key not in barriers:raise ValueError('use of uninitialized mbarrier')
            return barriers[key]
        def finish_phase(state):
            if state['arrived']==state['count'] and state['bytes']==0:
                state['complete'].append(set(state['deps']))
                state.update(arrived=0,deps=set())
        def process(actor,kind,payload):
            nonlocal max_registers
            if kind=='registers':actor['registers']=payload;return True
            if kind=='zero':
                if payload in actor['last_acc'] and not known(actor,[actor['last_acc'][payload]]):
                    raise ValueError('accumulator overwritten before WGMMA completion')
                actor['zeroed'].add(payload);actor['last_acc'].pop(payload,None);actor['mma_fenced']=False
                if inputs is not None:
                    buffer,(mi,ni,ri)=payload;key=(buffer,mi,ni)
                    if key not in actor['numeric']:actor['numeric'][key]=np.empty((16,64),dtype='float32')
                    lanes=np.arange(32)
                    actor['numeric'][key][lanes//4+(ri%4//2)*8,ri//4*8+(lanes%4)*2+ri%2]=0
                return True
            if kind=='mma_fence':actor['mma_fenced']=True;return True
            if kind=='proxy_fence':
                actor['proxy_known']=actor['known']
                for state in barriers.values():state['published']=True
                return True
            if kind=='mma_commit':
                actor['committed_mma']+=actor['pending_mma'];actor['pending_mma']=[];return True
            if kind=='bulk_commit':
                actor['committed_bulk']+=actor['pending_bulk'];actor['pending_bulk']=[];return True
            if kind=='init':
                key,count=payload
                if key in barriers:
                    raise ValueError('reinitializing a valid mbarrier without invalidation')
                epoch[key]+=1;barriers[key]=dict(count=count,arrived=0,bytes=0,deps=set(),complete=[],published=False)
                return True
            if kind=='invalidate':
                state=barrier(payload)
                if state['bytes'] or state['arrived'] or any(not known(actor,ids) for ids in state['complete']):
                    raise ValueError('invalidating an mbarrier before its operations complete')
                del barriers[payload]
                return True
            if kind=='expect':
                key,count=payload;state=barrier(key)
                if not state['published']:raise ValueError('mbarrier initialization lacks an async proxy fence')
                if state['bytes'] or state['arrived']:raise ValueError('overlapping mbarrier transaction phases')
                state['bytes']=count;state['arrived']=1;return True
            if kind=='arrive':
                state=barrier(payload);state['arrived']+=1;state['deps'].update(actor['frontier']);finish_phase(state);return True
            if kind=='wait':
                key,parity=payload;state=barrier(key);sequence=(key,epoch[key]);index=actor['waits'][sequence]
                if parity!=index%2:raise ValueError('incorrect mbarrier phase parity')
                if index>=len(state['complete']):return False
                complete(actor,state['complete'][index]);actor['waits'][sequence]+=1;statistics['barrier_waits']+=1;return True
            if kind=='descriptor':
                dst,src,layout=payload;descriptors[(actor['warp'],dst)]=(src,layout);return True
            if kind=='load':
                dst,descriptor,xy,key=payload;tensor,rows,cols,swizzle=maps[descriptor]
                region=(xy[1],xy[0],rows,cols);validate_region(tensor,region)
                if tensor not in (0,1):raise ValueError('TMA input is not A/B')
                if dst in shared_state and not known(actor,shared_state[dst]['consumers']):
                    raise ValueError('shared input buffer overwritten before its consumers complete')
                ident=emit(actor,'cp.dram2sram_tma',tensor,region)
                shared_state[dst]=dict(id=ident,region=region,tensor=tensor,swizzle=swizzle,consumers=set())
                state=barrier(key);state['deps'].add(ident);state['bytes']-=2*rows*cols
                if state['bytes']<0:raise ValueError('TMA bytes exceed mbarrier transaction expectation')
                finish_phase(state);return True
            if kind=='mma':
                output,regs,descs,flags=payload;row,col,rk=output
                if not actor['mma_fenced']:raise ValueError('WGMMA register initialization lacks wgmma.fence')
                if flags!=(1,1,1,0,1):raise ValueError('unsupported WGMMA scaling/transpose operands')
                reg_buffer,reg_index=regs[0]
                if len(reg_index)!=3 or any(not buffer.same_as(reg_buffer) or index!=(*reg_index[:2],i)
                                            for i,(buffer,index) in enumerate(regs)):
                    raise ValueError('unsupported WGMMA accumulator register ordering')
                if any(reg not in actor['zeroed'] and reg not in actor['last_acc'] for reg in regs):
                    raise ValueError('uninitialized accumulator fragment')
                if any((actor['warp'],descriptor) not in descriptors for descriptor in descs):raise ValueError('missing WGMMA matrix descriptor')
                aa,la=descriptors[(actor['warp'],descs[0])];bb,lb=descriptors[(actor['warp'],descs[1])]
                abuf,ai=aa;bbuf,bi=bb
                if len(ai)!=3 or len(bi)!=4:raise ValueError('unsupported WGMMA shared descriptor layout')
                akey=(abuf,(ai[0],0,0));bkey=(bbuf,(bi[0],bi[1],0,0))
                if akey not in shared_state or bkey not in shared_state:raise ValueError('WGMMA reads an uninitialized shared stage')
                sa,sb=shared_state[akey],shared_state[bkey]
                if la!=(1,int(abuf.shape[-1]),{16:1,32:2,64:3}.get(int(abuf.shape[-1]))) or lb!=(1,64,3):
                    raise ValueError('unsupported WGMMA descriptor strides/swizzle')
                if la[2]!=sa['swizzle'] or lb[2]!=sb['swizzle']:
                    raise ValueError('TMA/WGMMA physical layouts disagree')
                ar,ac,am,ak=sa['region'];br,bc,bk,bn=sb['region']
                if (ar+ai[1],ac+ai[2],br+bi[2],bc+bi[3])!=(row,rk,rk,col) or ai[1]+64>am or ai[2]+16>ak or bi[2]+16>bk or bi[3]+64>bn:
                    raise ValueError('WGMMA operands do not implement the tensorized GEMM region')
                if not known(actor,[sa['id'],sb['id']]):raise ValueError('WGMMA before full TMA completion wait')
                region=(row+(actor['warp']%4)*16,col,16,64);validate_region(2,region)
                if rk<0 or rk%16 or rk+16>k:raise ValueError('invalid reduction region')
                work_key=region[:2];bit=1<<(rk//16)
                if arithmetic[work_key]&bit:raise ValueError('duplicate GEMM reduction work')
                arithmetic[work_key]|=bit
                if inputs is not None:
                    buffer,(mi,ni,_)=regs[0]
                    actor['numeric'][(buffer,mi,ni)]+=np.einsum(
                        'ik,kj->ij',input_a[region[0]:region[0]+16,rk:rk+16],
                        input_b[rk:rk+16,col:col+64],dtype=np.float32,optimize=False)
                extra={actor['last_acc'][reg] for reg in regs if reg in actor['last_acc']}
                ident=emit(actor,'wgmma.m64n64k16',2,region,extra,'fp32')
                for reg in regs:actor['last_acc'][reg]=ident
                actor['pending_mma'].append(ident)
                collectives[(actor['warp']//4,row,col,rk)].append(ident)
                sa['consumers'].add(ident);sb['consumers'].add(ident)
                return True
            if kind=='mma_wait':
                if actor['pending_mma']:raise ValueError('WGMMA wait before async group commit')
                pending=set(actor['committed_mma'])
                mates=[ids for ids in collectives.values() if pending.intersection(ids)]
                if any(len(ids)!=4 for ids in mates):return False
                complete(actor,[ident for ids in mates for ident in ids]);actor['committed_mma']=[];return True
            if kind=='stage':
                output,buffer,spatial,source=payload;region=(*output,16,16);validate_region(2,region)
                if output in staged_coverage:raise ValueError('duplicate output staging region')
                if actor['pending_mma'] or actor['committed_mma']:raise ValueError('output staging before WGMMA completion wait')
                source_buffer,indices=source
                mi,ni,ri=indices;ri=np.asarray(ri)
                rows,cols=np.broadcast_arrays(*spatial)
                lanes=np.repeat(np.arange(32),8)
                register_ids=[actor['last_acc'].get((source_buffer,(int(mi),int(ni),int(reg)))) for reg in ri]
                if any(ident is None for ident in register_ids) or not known(actor,register_ids):
                    raise ValueError('epilogue reads uncomputed/unwaited accumulator registers')
                global_rows=np.array([ir.operations[ident].tile[0] for ident in register_ids])+lanes//4+(ri%4//2)*8
                global_cols=np.array([ir.operations[ident].tile[1] for ident in register_ids])+ri//4*8+(lanes%4)*2+ri%2
                local=(int(rows.min()),int(cols.min()),16,16)
                if (int(global_rows.min()),int(global_cols.min()))!=output or np.ptp(rows)!=15 or np.ptp(cols)!=15 or np.ptp(global_rows-rows) or np.ptp(global_cols-cols) or len(np.unique((rows-rows.min())*16+cols-cols.min()))!=256:
                    raise ValueError('distributed register epilogue disagrees with output region/storage layout')
                if inputs is not None:
                    values=actor['numeric'][(source_buffer,int(mi),int(ni))][lanes//4+(ri%4//2)*8,ri//4*8+(lanes%4)*2+ri%2]
                    numerical_output[global_rows,global_cols]=values.astype('float16')
                if not known(actor,output_reads[buffer]):raise ValueError('output staging overwritten before prior TMA store completion')
                staged_coverage.add(output);ident=emit(actor,'st.reg2sram',2,region)
                actor['pending_scalar'].append(ident);stage_nodes.append((ident,region,buffer,local));return True
            if kind=='store':
                src,descriptor,xy=payload;tensor,rows,cols,swizzle=maps[descriptor]
                if tensor!=2 or swizzle!=0:raise ValueError('unsupported output TMA layout')
                region=(xy[1],xy[0],rows,cols);validate_region(2,region)
                required={(row,col) for row in range(region[0],region[0]+rows,16) for col in range(region[1],region[1]+cols,16)}
                candidates=[(ident,tile,local) for ident,tile,buffer,local in stage_nodes if buffer.same_as(src[0]) and tile[:2] in required]
                if {tile[:2] for _,tile,_ in candidates}!=required or not known(actor,[ident for ident,_,_ in candidates]):
                    raise ValueError('output TMA before complete shared staging/CTA join')
                if any((region[0]+local[0]-src[1][0],region[1]+local[1]-src[1][1])!=tile[:2] for _,tile,local in candidates):
                    raise ValueError('TMA output coordinates disagree with staged accumulator values')
                if not all(actor['proxy_known'] & (1<<(ident-base)) for ident,_,_ in candidates):
                    raise ValueError('TMA output reads lack a shared async proxy fence')
                if required.intersection(stored_coverage):raise ValueError('overlapping TMA output regions')
                stored_coverage.update(required);ident=emit(actor,'cp.sram2dram_tma',2,region)
                actor['pending_bulk'].append(ident);output_reads[src[0]].append(ident);return True
            if kind=='bulk_wait':
                if actor['pending_bulk']:raise ValueError('bulk wait before async group commit')
                complete(actor,actor['committed_bulk']);actor['committed_bulk']=[];return True
            if kind=='sync':return False
            raise ValueError(f'unsupported native event {kind}')

        while not all(actor['done'] for actor in actors):
            progress=False
            for actor in actors:
                if actor['done']:continue
                if actor['event'] is None:
                    try:actor['event']=next(actor['stream'])
                    except StopIteration:
                        if actor['pending_mma'] or actor['pending_bulk'] or actor['committed_mma'] or actor['committed_bulk']:raise ValueError('kernel exits with unwaited async operations')
                        actor['done']=True;progress=True;continue
                kind,payload=actor['event']
                if process(actor,kind,payload):actor['event']=None;progress=True
            waiting=[actor for actor in actors if actor['event'] is not None and actor['event'][0]=='sync']
            if waiting and len(waiting)==12:
                union=set()
                for actor in actors:
                    complete(actor,actor['pending_scalar']);actor['pending_scalar']=[];union.update(actor['frontier'])
                for actor in actors:complete(actor,union);actor['event']=None
                progress=True;statistics['cta_joins']+=1
            if not progress:raise ValueError('native TIR synchronization deadlock or divergent CTA barrier')
        register_bytes=0
        for actor in actors:
            budget=actor['registers']
            if budget is None or budget<24 or budget>255 or budget%8:raise ValueError('invalid/missing register budget')
            register_bytes+=32*budget*4
        max_registers=max(max_registers,register_bytes)
        ir.cta(cta,cta%hardware['num_sms'],shared,register_bytes)
    declared=tuple(map(int,func.attrs['gtsim.resources']))
    if declared!=(shared,max_registers,12):raise ValueError('resource metadata differs from actual allocations/thread budgets')
    if shared>min(hardware['shared_memory_per_cta_bytes'],hardware['shared_memory_per_sm_bytes']) or max_registers>hardware['register_file_per_sm_bytes']:
        raise ValueError('actual kernel resource use exceeds hardware limits')
    expected={(row,col) for row in range(0,m,16) for col in range(0,n,16)}
    if staged_coverage!=expected or stored_coverage!=expected:
        raise ValueError('native output regions do not cover C exactly once')
    full_mask=(1<<(k//16))-1
    if len(arithmetic)!=(m//16)*(n//64) or any(mask!=full_mask for mask in arithmetic.values()):
        raise ValueError('native WGMMA work does not preserve the complete GEMM reduction')
    evidence=dict(nodes=len(ir.operations),dimensions=[m,n,k],shared_bytes=shared,register_bytes=max_registers,
                   grid=grid,native_semantics_pass=True,multiply_accumulates=m*n*k,**statistics)
    if inputs is not None:evidence['numerical_output']=numerical_output
    return ir,evidence
