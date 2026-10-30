"""Bounded stable-attention -> online FA2 compiler for Ampere tile simulation.

The algorithm rewrite is compiler supplied; native MetaSchedule chooses tile,
split, warp and register-stage factors. No C++ reference builder is invoked.
The emitted functional online TensorIR is saved and validated before lowering.
"""
import math

from auto_ir import nodes
from serve_lower import Graph


def online_module(tvm, batch, heads_q, heads_kv, seq, dim, bn, splits, warps, depth):
    jobs, group, bm = batch * heads_kv, heads_q // heads_kv, 16 * warps
    blocks = math.ceil(seq / bn)
    per_split = math.ceil(blocks / splits)
    if group > bm or min(batch, heads_q, heads_kv, seq, dim, bn, splits, warps, depth) <= 0:
        raise ValueError('invalid online FA2 geometry')
    source = f'''
@I.ir_module
class OnlineFA2:
    @T.prim_func
    def main(Q: T.Buffer(({jobs}, {group}, {dim}), "bfloat16"), K: T.Buffer(({jobs}, {seq}, {dim}), "bfloat16"), V: T.Buffer(({jobs}, {seq}, {dim}), "bfloat16"), PO: T.Buffer(({jobs}, {splits}, {group}, {dim}), "float32"), PL: T.Buffer(({jobs}, {splits}, {group}), "float32")):
        T.func_attr({{"gtsim.compiler": "ampere_online_fa2_v1", "gtsim.heads_kv": {heads_kv}}})
        for job in T.thread_binding({jobs}, thread="blockIdx.y"):
            for part in T.thread_binding({splits}, thread="blockIdx.x"):
                Qs = T.alloc_shared(({bm}, {dim}), "bfloat16")
                Ks = T.alloc_shared(({depth}, {bn}, {dim}), "bfloat16")
                Vs = T.alloc_shared(({depth}, {bn}, {dim}), "bfloat16")
                score = T.alloc_local(({bm}, {bn}), "float32")
                P = T.alloc_local(({bm}, {bn}), "bfloat16")
                Oacc = T.alloc_local(({bm}, {dim}), "float32")
                M = T.alloc_local(({bm},), "float32")
                L = T.alloc_local(({bm},), "float32")
                old_m = T.alloc_local(({bm},), "float32")
                alpha = T.alloc_local(({bm},), "float32")
                tile_max = T.alloc_local(({bm},), "float32")
                tile_sum = T.alloc_local(({bm},), "float32")
                for r, d in T.grid({bm}, {dim}):
                    Qs[r,d] = T.if_then_else(r < {group}, Q[job,r,d], T.bfloat16(0))
                    Oacc[r,d] = T.float32(0)
                for r in T.serial({bm}):
                    M[r] = T.float32(-1e30)
                    L[r] = T.float32(0)
                for iteration in T.serial(T.max(0,T.min({blocks},(part+1)*{per_split})-part*{per_split})):
                    offset = (T.min({blocks},(part+1)*{per_split})-1-iteration)*{bn}
                    slot = iteration % {depth}
                    for n,d in T.grid({bn},{dim}):
                        Ks[slot,n,d] = T.if_then_else(offset+n < {seq}, K[job,offset+n,d], T.bfloat16(0))
                        Vs[slot,n,d] = T.if_then_else(offset+n < {seq}, V[job,offset+n,d], T.bfloat16(0))
                    for r,n in T.grid({bm},{bn}):
                        score[r,n] = T.float32(0)
                        for d in T.serial({dim}):
                            score[r,n] = score[r,n] + T.Cast("float32",Qs[r,d])*T.Cast("float32",Ks[slot,n,d])
                        score[r,n] = T.if_then_else(offset+n < {seq},score[r,n]*T.float32({1/math.sqrt(dim)}),T.float32(-1e30))
                    for r in T.serial({bm}):
                        tile_max[r] = T.float32(-1e30)
                        for n in T.serial({bn}):
                            tile_max[r] = T.max(tile_max[r],score[r,n])
                        old_m[r] = M[r]
                        M[r] = T.max(M[r],tile_max[r])
                        alpha[r] = T.exp(old_m[r]-M[r])
                        tile_sum[r] = T.float32(0)
                        for n in T.serial({bn}):
                            score[r,n] = T.exp(score[r,n]-M[r])
                            tile_sum[r] = tile_sum[r]+score[r,n]
                            P[r,n] = T.Cast("bfloat16",score[r,n])
                        L[r] = L[r]*alpha[r]+tile_sum[r]
                    for r,d in T.grid({bm},{dim}):
                        Oacc[r,d] = Oacc[r,d]*alpha[r]
                        for n in T.serial({bn}):
                            Oacc[r,d] = Oacc[r,d]+T.Cast("float32",P[r,n])*T.Cast("float32",Vs[slot,n,d])
                for r,d in T.grid({group},{dim}):
                    PO[job,part,r,d] = T.if_then_else(L[r]>T.float32(0),Oacc[r,d]/L[r],T.float32(0))
                for r in T.serial({group}):
                    PL[job,part,r] = T.if_then_else(L[r]>T.float32(0),M[r]+T.log(L[r]),T.float32(-1e30))

    @T.prim_func
    def combine(PO: T.Buffer(({jobs},{splits},{group},{dim}),"float32"), PL: T.Buffer(({jobs},{splits},{group}),"float32"), O: T.Buffer(({jobs},{group},{dim}),"bfloat16")):
        T.func_attr({{"gtsim.compiler":"ampere_online_fa2_combine_v1"}})
        for job in T.thread_binding({jobs},thread="blockIdx.x"):
            maximum = T.alloc_local(({group},),"float32")
            denom = T.alloc_local(({group},),"float32")
            weights = T.alloc_shared(({group},{splits}),"float32")
            acc = T.alloc_local(({group},{dim}),"float32")
            for r in T.serial({group}):
                maximum[r] = T.float32(-1e30)
                for s in T.serial({splits}):
                    maximum[r] = T.max(maximum[r],PL[job,s,r])
                denom[r] = T.float32(0)
                for s in T.serial({splits}):
                    weights[r,s] = T.exp(PL[job,s,r]-maximum[r])
                    denom[r] = denom[r]+weights[r,s]
                for s in T.serial({splits}):
                    weights[r,s] = weights[r,s]/denom[r]
                for d in T.serial({dim}):
                    acc[r,d] = T.float32(0)
                for s in T.serial({splits}):
                    for d in T.serial({dim}):
                        acc[r,d] = acc[r,d]+weights[r,s]*PO[job,s,r,d]
                for d in T.serial({dim}):
                    O[job,r,d] = T.Cast("bfloat16",acc[r,d])
'''
    from tvm.script import ir as I, tirx as T
    from serve_memory import annotate
    return annotate(tvm, tvm.script.from_source(source, extra_vars={'I': I, 'T': T}), 'CoLocated')


def compile_online(tvm, scheduled):
    """A recognized algebraic rewrite, with factors read from scheduled loops."""
    from serve_memory import validate
    validate(scheduled['main'], dict(topology='CoLocated'))
    sch = tvm.s_tir.Schedule(scheduled, debug_mask='all')
    block = sch.get_sblock('raw_score')
    loops = [sch.get(loop) for loop in sch.get_loops(block)]
    bn = int(next(loop.extent for loop in loops if 'gtsim.kv_tile' in loop.annotations))
    splits = int(next(loop.extent for loop in loops if 'gtsim.split_kv' in loop.annotations))
    annotation = sch.get(block).annotations
    warps, depth = int(annotation['gtsim.ampere_warps']), int(annotation['gtsim.register_stages'])
    func = scheduled['main']
    parameters = {str(param.name): param for param in func.params}
    b, hq, dim = map(int, parameters['Q'].shape)
    _, hkv, seq, _ = map(int, parameters['K'].shape)
    # Fail closed if the mathematical blocks do not match the supported stable
    # softmax graph. Loop transformations are intentionally not template-matched.
    import sys
    from pinned_tvm import ROOT
    sys.path.insert(0, str(ROOT / 'workloads/serve'))
    from semantics import fa2
    canonical = tvm.te.create_prim_func(fa2(b, hq, hkv, seq, dim))
    expected = {node.name_hint: node for node in nodes(canonical, tvm.s_tir.SBlock) if node.name_hint != 'root'}
    actual = {node.name_hint: node for node in nodes(func, tvm.s_tir.SBlock) if node.name_hint != 'root'}
    import tvm_ffi
    if actual.keys() != expected.keys():
        raise ValueError('unsupported attention computation graph')
    for name in actual:
        if (not tvm_ffi.structural_equal(actual[name].body, expected[name].body, map_free_vars=True)
                or not tvm_ffi.structural_equal(actual[name].init, expected[name].init, map_free_vars=True)):
            raise ValueError('unsupported attention algebra: ' + name)
    return online_module(tvm, b, hq, hkv, seq, dim, bn, splits, warps, depth)


def lower_online(tvm, native, hardware):
    from serve_memory import validate
    validate(native['main'], hardware)
    validate(native['combine'], hardware)
    main = native['main']
    q, k, v, po, pl = main.params
    jobs, group, dim = map(int, q.shape)
    seq, splits = int(k.shape[1]), int(po.shape[1])
    allocs = {str(node.buffer.name): node.buffer for node in nodes(main, tvm.tirx.AllocBuffer)}
    bm = int(allocs['Qs'].shape[0])
    depth, bn, _ = map(int, allocs['Ks'].shape)
    warps = bm // 16
    hkv = int(main.attrs['gtsim.heads_kv'])
    import tvm_ffi
    reconstructed = online_module(tvm, jobs // hkv, group * hkv, hkv, seq, dim, bn, splits, warps, depth)
    if not tvm_ffi.structural_equal(native, reconstructed, map_free_vars=True):
        raise ValueError('unsupported online FA2 statements or synchronization geometry')
    graph = Graph(hardware)
    shared = 2 * dim * (bm + 2 * depth * bn)
    registers = 4 * bm * (dim + bn) + 2 * bm * dim + warps * 32 * 24 * 4
    # Validate once before graph expansion.
    if shared > min(hardware['shared_memory_per_cta_bytes'], hardware['shared_memory_per_sm_bytes']):
        raise ValueError('FA2 shared buffer ring exceeds capacity')
    if registers > min(hardware['register_file_per_sm_bytes'], warps * 32 * 255 * 4):
        raise ValueError('FA2 padded accumulator/score/Q register footprint exceeds capacity')
    blocks = math.ceil(seq / bn)
    per_split = math.ceil(blocks / splits)
    main_stores = []
    tensors = []
    for job in range(jobs):
        ids = [graph.tensor(f'{name}_{job}', shape, dtype) for name, shape, dtype in
               [('Q', (group, dim), 'bfloat16'), ('K', (seq, dim), 'bfloat16'),
                ('V', (seq, dim), 'bfloat16'), ('O', (group, dim), 'bfloat16')]]
        partial = [(graph.tensor(f'PO_{job}_{s}', (group, dim)),
                    graph.tensor(f'PL_{job}_{s}', (group, 1))) for s in range(splits)]
        tensors.append((ids, partial))
        for part in range(splits):
            cta = graph.cta(warps, shared, registers)
            def add(w, op, tile, deps=(), tensor=0, dtype='float32'):
                return graph.add(cta, w, op, tile, deps, tensor, dtype, 'fa2_main',
                                 layout='swizzled' if op == 'ld.sram2reg' else 'row')
            q_copies = []
            for w in range(warps):
                live = max(0, min(16, group - 16 * w))
                if live:
                    q_copies.append(add(w, 'cp.dram2sram', (16 * w, 0, live, dim), tensor=ids[0], dtype='bfloat16'))
            qreg, previous_o, previous_l, previous_m = [], [], [], []
            for w in range(warps):
                loads = [add(w, 'ld.sram2reg', (16 * w, d, 16, 16), q_copies, dtype='bfloat16') for d in range(0, dim, 16)]
                qreg.append(add(w, 'barrier.registers', (0, 0, 1, 1), loads))
                previous_o.append(add(w, 'simd.fill', (0, 0, 16, dim)))
                previous_l.append(add(w, 'simd.fill', (0, 0, 16, 1)))
                previous_m.append(add(w, 'simd.fill', (0, 0, 16, 1)))
            consumed = []
            lo, hi = part * per_split, min(blocks, (part + 1) * per_split)
            for iteration, block in enumerate(range(hi - 1, lo - 1, -1)):
                offset = block * bn
                live = min(bn, seq - offset)
                reuse = consumed[iteration - depth] if iteration >= depth else []
                kc, vc = [], []
                for w in range(warps):
                    row = w * math.ceil(bn / warps)
                    count = max(0, min(math.ceil(bn / warps), live - row))
                    if count:
                        kc.append(add(w, 'cp.dram2sram', (offset + row, 0, count, dim), reuse, ids[1], 'bfloat16'))
                        vc.append(add(w, 'cp.dram2sram', (offset + row, 0, count, dim), reuse, ids[2], 'bfloat16'))
                read_done = []
                for w in range(warps):
                    scores = [-1] * (bn // 8)
                    fragments = []
                    for kk in range(0, dim, 16):
                        for n in range(bn // 8):
                            reuse_fragment = fragments[-4] if len(fragments) >= 4 else previous_o[w]
                            loaded = add(w, 'ld.sram2reg', (n * 8, kk, 8, 16), [*kc, reuse_fragment], dtype='bfloat16')
                            deps = [qreg[w], loaded]
                            if scores[n] >= 0:
                                deps.append(scores[n])
                            scores[n] = add(w, 'mma.sync.m16n8k16', (16 * w, n * 8, 16, 8), deps)
                            fragments.append(scores[n])
                            read_done.append(loaded)
                    scaled = add(w, 'simd.mul', (0, 0, 16, bn), scores)
                    maximum = add(w, 'simd.max', (0, 0, 16, bn), [scaled])
                    maximum = add(w, 'shfl', (0, 0, 16, 4), [maximum])
                    new_m = add(w, 'simd.max', (0, 0, 16, 1), [maximum, previous_m[w]])
                    difference = add(w, 'simd.sub', (0, 0, 16, 1), [new_m, previous_m[w]])
                    alpha = add(w, 'sfu.exp', (0, 0, 16, 1), [difference])
                    shift = add(w, 'simd.sub', (0, 0, 16, bn), [scaled, new_m])
                    probability = add(w, 'sfu.exp', (0, 0, 16, bn), [shift])
                    sum_node = add(w, 'simd.add', (0, 0, 16, bn), [probability])
                    sum_node = add(w, 'shfl', (0, 0, 16, 4), [sum_node])
                    new_l = add(w, 'simd.fma', (0, 0, 16, 1), [sum_node, previous_l[w], alpha])
                    scale_o = add(w, 'simd.mul', (0, 0, 16, dim), [previous_o[w], alpha])
                    p = add(w, 'simd.cast', (0, 0, 16, bn), [probability])
                    acc = [scale_o] * (dim // 8)
                    fragments = []
                    for kk in range(0, bn, 16):
                        for d in range(dim // 8):
                            reuse_fragment = fragments[-4] if len(fragments) >= 4 else scale_o
                            loaded = add(w, 'ld.sram2reg', (kk, d * 8, 16, 8), [*vc, reuse_fragment], dtype='bfloat16')
                            acc[d] = add(w, 'mma.sync.m16n8k16', (16 * w, d * 8, 16, 8), [p, loaded, acc[d]])
                            fragments.append(acc[d])
                            read_done.append(loaded)
                    previous_o[w] = add(w, 'barrier.registers', (0, 0, 1, 1), acc)
                    previous_l[w], previous_m[w] = new_l, new_m
                consumed.append([add(0, 'barrier.registers', (0, 0, 1, 1), read_done)])
            for w in range(warps):
                live = max(0, min(16, group - 16 * w))
                if not live:
                    continue
                norm = add(w, 'simd.fill' if lo >= hi else 'sfu.divide', (0, 0, live, dim), [previous_o[w], previous_l[w]])
                lse = add(w, 'simd.fill' if lo >= hi else 'sfu.log', (0, 0, live, 1), [previous_l[w]])
                lse = add(w, 'simd.add', (0, 0, live, 1), [lse, previous_m[w]])
                main_stores.append(add(w, 'st.reg2dram', (16 * w, 0, live, dim), [norm], partial[part][0]))
                main_stores.append(add(w, 'st.reg2dram', (16 * w, 0, live, 1), [lse], partial[part][1]))
    main_operations = len(graph.operations)
    main_join = None
    padded = 2 ** math.ceil(math.log2(splits))
    for job, (ids, partial) in enumerate(tensors):
        cta = graph.cta(max(1, min(4, group)), max(4 * group * splits, 4 * padded), min(32768, max(1, min(4, group)) * 32 * 255 * 4))
        if main_join is None:
            main_join = graph.add(cta, 0, 'barrier.registers', (0, 0, 1, 1), main_stores, stage='fa2_combine')
        for row in range(group):
            w = row % max(1, min(4, group))
            def combine(op, tile, deps=(), tensor=0, dtype='float32'):
                return graph.add(cta, w, op, tile, deps, tensor, dtype, 'fa2_combine')
            loads = [combine('ld.dram2reg', (row, 0, 1, 1), [main_join], lse) for _, lse in partial]
            mx = combine('simd.max', (0, 0, 1, padded), loads)
            for _ in range(5):
                mx = combine('shfl', (0, 0, 1, 32), [mx])
            sub = combine('simd.sub', (0, 0, 1, padded), [mx])
            exp = combine('sfu.exp', (0, 0, 1, padded), [sub])
            summed = combine('simd.add', (0, 0, 1, padded), [exp])
            for _ in range(5):
                summed = combine('shfl', (0, 0, 1, 32), [summed])
            weights = combine('sfu.divide', (0, 0, 1, padded), [summed, exp])
            weights = combine('st.reg2sram', (0, 0, 1, padded), [weights])
            acc = -1
            for s, (out, _) in enumerate(partial):
                dep = [weights] if acc < 0 else [weights, acc]
                loaded = combine('ld.dram2reg', (row, 0, 1, dim), dep, out)
                weight = combine('ld.sram2reg', (0, 0, 1, 1), dep)
                acc = combine('simd.fma', (0, 0, 1, dim), [loaded, weight, *([] if acc < 0 else [acc])])
            cast = combine('simd.cast', (0, 0, 1, dim), [acc])
            combine('st.reg2dram', (row, 0, 1, dim), [cast], ids[3], 'bfloat16')
    return graph, dict(batch=jobs // hkv, heads_q=group * hkv, heads_kv=hkv, seq=seq, dim=dim,
                       kv_tile=bn, splits=splits, warps=warps, stages=depth,
                       shared_bytes=shared, register_bytes=registers,
                       nodes=len(graph.operations), main_nodes=main_operations, ctas=len(graph.ctas),
                       scope='Online FA2 with native warp MMA; FP32 stable softmax and BF16 P; separate combine',
                       tensor_tiers={name: graph.tensors[ident][-1] for name, ident in graph.tensor_ids.items()})
