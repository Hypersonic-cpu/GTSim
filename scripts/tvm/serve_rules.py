"""Native TVM schedule rules for batch-one serving, separate from semantics."""
import math

from auto_ir import nodes


INLINE = {
    1: ('q', 'k', 'a', 'softplus', 'beta', 'conv_silu'),
    2: ('q', 'k', 'a', 'softplus', 'beta', 'conv_silu', 'decay',
        'q_norm', 'k_norm', 'delta'),
}


def space(kind):
    if kind == 'gdn_split':
        return dict(ba_splits=[1, 2, 4, 8, 16, 32, 64], inline=[0, 1, 2])
    if kind == 'fa2':
        return dict(kv_tile=[32, 64, 128], splits=[16, 32, 47, 64, 96, 128],
                    warps=[1, 2, 4], stages=[1, 2])
    return dict(output_tile=[64, 128, 256], reduction_tile=[32, 64, 128],
                warps=[1, 2, 4], stages=[1, 2, 3],
                inline=[0, 1, 2] if kind == 'gdn' else [0])


def choices(trace, kind):
    instructions = [inst for inst in trace.insts if inst.kind.name == 'SampleCategorical']
    return {key: int(inst.attrs[0][int(trace.decisions[inst])])
            for key, inst in zip(space(kind), instructions)}


def apply(tvm, sch, kind, decisions=None):
    if kind == 'gdn_split':
        from serve_hierarchy import apply_hierarchy
        return apply_hierarchy(tvm, sch, decisions)
    selected = {}
    for key, values in space(kind).items():
        decision = None if decisions is None or key not in decisions else values.index(decisions[key])
        selected[key] = sch.sample_categorical(values, [1 / len(values)] * len(values), decision=decision)
    # Resolving IntImm decisions here drives actual TVM transformations. The
    # backend subsequently receives only serialized scheduled TIR, not this map.
    values = choices(sch.trace, kind)
    if kind == 'fa2':
        block = sch.get_sblock('raw_score')
        b, h, s, d = sch.get_loops(block)
        so, si = sch.split(s, factors=[None, selected['kv_tile']])
        split, block_kv = sch.split(so, factors=[selected['splits'], None])
        sch.reorder(b, h, split, block_kv, si, d)
        sch.annotate(split, 'gtsim.split_kv', 1)
        sch.annotate(si, 'gtsim.kv_tile', 1)
        sch.annotate(block, 'gtsim.ampere_warps', selected['warps'])
        sch.annotate(block, 'gtsim.register_stages', selected['stages'])
        return sch
    sch.annotate(sch.get_sblock('root'), 'gtsim.inline_level', selected['inline'])
    if kind == 'gdn':
        for name in INLINE.get(values['inline'], ()):
            sch.compute_inline(sch.get_sblock(name))
    blocks = sch.get_child_blocks(sch.get_sblock('root'))
    for handle in blocks:
        block = sch.get(handle)
        loops = list(sch.get_loops(handle))
        spatial = sum(int(var.iter_type) == 0 for var in block.iter_vars)
        reduction = [var for var in block.iter_vars if int(var.iter_type) == 2]
        if len(reduction) > 1 or not spatial:
            raise ValueError('serving vector compiler requires at most one reduction axis')
        output = sch.fuse(*loops[:spatial])
        outer, inner = sch.split(output, factors=[None, selected['output_tile']])
        warp, vector, lane = sch.split(inner, factors=[selected['warps'], None, 32])
        if reduction:
            red_outer, red_inner = sch.split(loops[-1], factors=[None, selected['reduction_tile']])
            sch.reorder(outer, red_outer, warp, vector, lane, red_inner)
            sch.annotate(red_inner, 'gtsim.reduction_tile', 1)
        sch.bind(outer, 'blockIdx.x')
        sch.bind(warp, 'threadIdx.y')
        sch.bind(lane, 'threadIdx.x')
        sch.annotate(vector, 'gtsim.output_vector', 1)
        sch.annotate(handle, 'gtsim.register_stages', selected['stages'])
    return sch


def stages(tvm, mod):
    """Recover tile factors from actual loops of deserialized scheduled TIR."""
    sch = tvm.s_tir.Schedule(mod, debug_mask='all')
    result = []
    for handle in sch.get_child_blocks(sch.get_sblock('root')):
        block = sch.get(handle)
        loops = [sch.get(loop) for loop in sch.get_loops(handle)]
        bound = {loop.thread_binding.thread_tag: int(loop.extent)
                 for loop in loops if loop.thread_binding is not None}
        vector = [int(loop.extent) for loop in loops if 'gtsim.output_vector' in loop.annotations]
        reduction = [int(loop.extent) for loop in loops if 'gtsim.reduction_tile' in loop.annotations]
        if not vector or bound.get('threadIdx.x') != 32:
            raise ValueError('missing scheduled vector thread layout')
        tile = bound['threadIdx.y'] * 32 * vector[0]
        shape = [int(n) for n in block.writes[0].source.shape]
        if bound['blockIdx.x'] != math.ceil(math.prod(shape) / tile):
            raise ValueError('block/thread tiling does not cover the semantic output')
        result.append(dict(block=block, output_tile=tile, warps=bound['threadIdx.y'],
                           reduction_tile=reduction[0] if reduction else 1,
                           stages=int(block.annotations['gtsim.register_stages'])))
    return result
