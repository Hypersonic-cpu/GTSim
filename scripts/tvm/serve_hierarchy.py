"""Hierarchical GDN rule: reuse measured projection layouts, search b/a Split-K and fusion."""
import json

from auto_ir import nodes
from serve_rules import INLINE


def apply_hierarchy(tvm, sch, decisions=None):
    split_values = [1, 2, 4, 8, 16, 32, 64]
    inline_values = [0, 1, 2]
    selected = {}
    for key, values in [('ba_splits', split_values), ('inline', inline_values)]:
        decision = None if decisions is None or key not in decisions else values.index(decisions[key])
        selected[key] = sch.sample_categorical(values, [1/len(values)]*len(values), decision=decision)
    samples = [inst for inst in sch.trace.insts if inst.kind.name == 'SampleCategorical']
    level = int(samples[1].attrs[0][int(sch.trace.decisions[samples[1]])])
    sch.annotate(sch.get_sblock('root'), 'gtsim.inline_level', selected['inline'])
    seed = json.loads(str(sch.mod['main'].attrs['gtsim.hierarchical_seed']))
    # Move Split-K outside all output loops before factoring. Both partial and
    # final reductions then become separate root stages with contiguous output.
    ba = sch.get_sblock('ba_acc')
    b, n, k = sch.get_loops(ba)
    flat = sch.fuse(b, n)
    ko, ki = sch.split(k, factors=[selected['ba_splits'], None])
    sch.reorder(ko, flat, ki)
    sch.rfactor(ko, 0)
    ko, flat = sch.get_loops(ba)
    sch.reorder(flat, ko)
    for name in INLINE.get(level, ()):
        sch.compute_inline(sch.get_sblock(name))
    for handle in sch.get_child_blocks(sch.get_sblock('root')):
        block = sch.get(handle)
        prefix = next((name for name in seed['projections'] if block.name_hint in (name, name+'_acc')), None)
        choice = seed['projections'][prefix] if prefix else seed['auxiliary']
        loops = list(sch.get_loops(handle))
        reductions = [var for var in block.iter_vars if int(var.iter_type) == 2]
        # rfactor has fused B/N spatial loops; count surrounding loops rather
        # than assuming one loop per semantic block iteration variable.
        spatial_count = len(loops)-len(reductions)
        output = sch.fuse(*loops[:spatial_count])
        outer, inner = sch.split(output, factors=[None, choice['output_tile']])
        warp, vector, lane = sch.split(inner, factors=[choice['warps'], None, 32])
        if reductions:
            red_outer, red_inner = sch.split(loops[-1], factors=[None, choice['reduction_tile']])
            sch.reorder(outer, red_outer, warp, vector, lane, red_inner)
            sch.annotate(red_inner, 'gtsim.reduction_tile', 1)
        sch.bind(outer, 'blockIdx.x')
        sch.bind(warp, 'threadIdx.y')
        sch.bind(lane, 'threadIdx.x')
        sch.annotate(vector, 'gtsim.output_vector', 1)
        sch.annotate(handle, 'gtsim.register_stages', choice['stages'])
    return sch
