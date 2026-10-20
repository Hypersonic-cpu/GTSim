"""Native MetaSchedule design space: TE loop transformations and tensorization."""

import math

from hopper_intrinsics import register_wgmma


def divisors(value, unit, maximum):
    return [unit * factor for factor in range(1, maximum // unit + 1)
            if value % (unit * factor) == 0]


def design_space(tvm, dimensions, hardware, group_count=None):
    m,n,k = dimensions
    # ISA granularity and the unchanged backend's four-warp collectives bound
    # the axes. This is a factor space, not a list of GEMM configurations.
    shared = min(hardware['shared_memory_per_sm_bytes'], hardware['shared_memory_per_cta_bytes'])
    registers = min(hardware['register_file_per_sm_bytes'], 12*32*255*4)
    max_m = min(m, (255-32)*128//64, registers // (4*64), shared // (2*16))
    max_n_per_group = min(n, 256, registers // (4*64))
    max_groups=min(2,n//64)
    if group_count is not None:
        max_n_per_group=min(max_n_per_group,256//group_count,n//group_count)
    min_bn=(group_count or 1)*64
    max_depth=min(k//16,(shared-2*64*min_bn)//(2*16*(64+min_bn)+32))
    return {'bm': divisors(m,64,max_m),
            'n_per_group': divisors(n,64,max_n_per_group),
            'bk': divisors(k,16,min(k,128//2,shared//(2*(64+64)))),
            'groups': [group_count] if group_count is not None else list(range(1,max_groups+1)),
            'pipeline_delay': list(range(max_depth)),
            'persistent': [0,1]}


def apply_rule(tvm, sch, dimensions, hardware, decisions=None, group_count=None):
    register_wgmma(tvm)
    space = design_space(tvm,dimensions,hardware,group_count)
    choices = {}
    for key, values in space.items():
        if not values:
            raise ValueError(f'no legal ISA factors for {key}')
        decision = None if decisions is None else values.index(decisions[key])
        weights=[1/(value+1)**2 for value in values] if key=='pipeline_delay' else [1.0]*len(values)
        choices[key] = sch.sample_categorical(values, [value/sum(weights) for value in weights], decision=decision)
    blocks=sch.get_child_blocks(sch.get_sblock('root'))
    reductions=[block for block in blocks if sch.get(block).init is not None]
    outputs=[block for block in blocks if sch.get(block).init is None]
    if len(reductions)!=1 or len(outputs)!=1:
        raise ValueError('initial scope requires one GEMM reduction and one FP16 cast epilogue')
    b=reductions[0]
    i,j,k = sch.get_loops(b)
    io,ii = sch.split(i,factors=[None,choices['bm']])
    jo,jg,ji = sch.split(j,factors=[None,choices['groups'],choices['n_per_group']])
    ko,ki = sch.split(k,factors=[None,choices['bk']])
    sch.reorder(io,jo,ko,jg,ii,ji,ki)
    a = sch.cache_read(b,0,'shared'); bb = sch.cache_read(b,1,'shared')
    sch.compute_at(a,ko); sch.compute_at(bb,ko)
    sch.set_scope(b,0,'local')
    sch.reverse_compute_at(outputs[0],jo)
    sch.bind(jg,'threadIdx.y')
    block = sch.fuse(io,jo); sch.bind(block,'blockIdx.x')
    sch.annotate(block,'gtsim.persistent',choices['persistent'])
    sch.decompose_reduction(b,ko)
    mi,ii = sch.split(ii,factors=[None,64])
    nj,ji = sch.split(ji,factors=[None,64])
    ks,ki = sch.split(ki,factors=[None,16])
    sch.reorder(jg,mi,nj,ks,ii,ji,ki)
    sch.tensorize(ii,'gtsim_hopper_m64n64k16')
    # TVM's native pipeline injector derives ring buffers and prologue/tail.
    # The pin's trace DCE does not descend into Array annotation inputs. A
    # scalar stage annotation preserves this sampled decision during replay.
    sch.annotate(ko,'gtsim.pipeline_delay',choices['pipeline_delay'])
    sch.annotate(ko,'software_pipeline_stage',[0,0,choices['pipeline_delay']])
    sch.annotate(ko,'software_pipeline_order',[0,1,2])
    return sch


def trace_choices(trace):
    keys = ('bm','n_per_group','bk','groups','pipeline_delay','persistent')
    insts = [inst for inst in trace.insts if inst.kind.name=='SampleCategorical']
    return {key:int(inst.attrs[0][int(trace.decisions[inst])]) for key,inst in zip(keys,insts)}


def validate_choices(choices, dimensions, hardware):
    m,n,k = dimensions
    bm,bk,groups = (choices[key] for key in ('bm','bk','groups'))
    per_group = choices['n_per_group']; bn = groups*per_group
    stages = choices['pipeline_delay']+1
    if m%bm or n%bn or k%bk or stages>k//bk:
        raise ValueError('nondivisible tile or pipeline deeper than reduction')
    if bn>256:
        raise ValueError('TMA dense 2-D box width exceeds 256 elements')
    compute_regs = bm*per_group//128+32
    if compute_regs>255:
        raise ValueError('distributed accumulator exceeds per-thread register limit')
    shared = 2*stages*bk*(bm+bn)+2*bm*bn+32*stages
    registers = 4*(groups*128*compute_regs+(384-groups*128)*32)
    if shared>min(hardware['shared_memory_per_cta_bytes'],hardware['shared_memory_per_sm_bytes']):
        raise ValueError('shared memory exceeds hardware capacity')
    if registers>hardware['register_file_per_sm_bytes'] or hardware['max_warps_per_sm']<12:
        raise ValueError('register/warp footprint exceeds hardware capacity')
    return dict(choices,bn=bn,stages=stages,shared_bytes=shared,register_bytes=registers)
