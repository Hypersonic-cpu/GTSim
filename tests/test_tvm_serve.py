#!/usr/bin/env python3
"""Independent numerical, trace-replay and fail-closed serving compiler checks."""
from pathlib import Path
import sys
import time

import ml_dtypes
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/tvm'))
sys.path.insert(0, str(ROOT / 'workloads/serve'))
from pinned_tvm import load_tvm


BF = ml_dtypes.bfloat16


def gdn_reference(x, args, hidden, nk, nv, dk, dv, ff):
    f = lambda value: value.astype('float32')
    b = lambda value: value.astype(BF)
    def rms(value, weight):
        value = f(value)
        return b(value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + 1e-6) * (1 + f(weight)))
    def linear(value, weight):
        return b(f(value) @ f(weight))
    def silu(value):
        return value / (1 + np.exp(-value))
    kd, vd = nk * dk, nv * dv
    channels = 2 * kd + vd
    normalized = rms(x, args['W_input_norm'])
    qkvz = linear(normalized, args['W_qkvz'])
    ba = linear(normalized, args['W_ba'])
    conv_state = np.concatenate([args['ConvState'][..., 1:], qkvz[:, :channels, None]], axis=-1)
    conv = b(silu(np.sum(f(conv_state) * f(args['W_conv'])[None, :, :], axis=-1)))
    q = f(conv[:, :kd]).reshape(-1, nk, dk)
    k = f(conv[:, kd:2 * kd]).reshape(-1, nk, dk)
    q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6) / np.sqrt(dk)
    k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    q = np.repeat(q, nv // nk, axis=1)
    k = np.repeat(k, nv // nk, axis=1)
    value = f(conv[:, 2 * kd:]).reshape(-1, nv, dv)
    z = f(qkvz[:, channels:]).reshape(-1, nv, dv)
    a = f(ba[:, nv:]) + f(args['dt_bias'])
    softplus = np.maximum(a, 0) + np.log1p(np.exp(-np.abs(a)))
    decay = np.exp(-np.exp(f(args['A_log'])) * softplus)
    beta = 1 / (1 + np.exp(-f(ba[:, :nv])))
    decayed = args['State'] * decay[..., None, None]
    prediction = np.einsum('bhij,bhi->bhj', decayed, k)
    delta = beta[..., None] * (value - prediction)
    next_state = decayed + k[..., :, None] * delta[..., None, :]
    attention = f(b(np.einsum('bhi,bhij->bhj', q, next_state)))
    gated = b(attention / np.sqrt(np.mean(attention * attention, axis=-1, keepdims=True) + 1e-6)
              * f(args['W_gated_norm']) * silu(z)).reshape(-1, vd)
    projected = linear(gated, args['W_out'])
    residual = b(f(x) + f(projected))
    up = linear(rms(residual, args['W_post_norm']), args['W_gate_up'])
    act = b(silu(f(up[:, :ff])) * f(up[:, ff:]))
    output = b(f(residual) + f(linear(act, args['W_down'])))
    return dict(Output=output, NextState=next_state, NextConvState=conv_state)


def main():
    tvm = load_tvm()
    import semantics
    from serve_memory import annotate
    from auto_ir import trace_json
    from serve_rules import apply
    from serve_lower import evaluate, lower_vector
    from serve_fa2 import online_module, lower_online
    from serve_eval import execute
    import json
    import subprocess
    hardware = json.loads(subprocess.check_output([str(ROOT / 'build/gtsim_serve_ir'),
                           str(ROOT / 'configs/v0.1.0/a100-pcie40-3m2f-grade1-simple.yaml'), '--describe-config'], text=True))
    rng = np.random.default_rng(240)
    dims = dict(hidden=64, nk=2, nv=4, dk=16, dv=16, ff=128)
    mod = tvm.IRModule({'main': tvm.te.create_prim_func(semantics.gdn(**dims))})
    mod = annotate(tvm, mod, hardware['topology'])
    inputs = {str(p.name): (rng.normal(size=tuple(int(n) for n in p.shape)) * .1).astype(BF if str(p.dtype) == 'bfloat16' else 'float32')
              for p in mod['main'].params if str(p.name) not in ('Output', 'NextState', 'NextConvState')}
    for inline in [0, 1, 2]:
        sch = tvm.s_tir.Schedule(mod, debug_mask='all', seed=1)
        apply(tvm, sch, 'gdn', dict(output_tile=128, reduction_tile=64, warps=4, stages=1, inline=inline))
        replay = tvm.s_tir.Schedule(mod, debug_mask='all')
        tvm.s_tir.schedule.Trace.apply_json_to_schedule(trace_json(sch.trace), replay)
        tvm.ir.assert_structural_equal(sch.mod, replay.mod)
        current = {name: value.copy() for name, value in inputs.items()}
        for step in range(2):
            reference = gdn_reference(current['X'], current, **dims)
            actual = evaluate(tvm, tvm.ir.load_json(tvm.ir.save_json(sch.mod)), current)
            for name in ('Output', 'NextState', 'NextConvState'):
                assert np.isfinite(actual[name].astype('float32')).all()
                assert np.isfinite(reference[name].astype('float32')).all()
                np.testing.assert_allclose(actual[name].astype('float32'), reference[name].astype('float32'), atol=.002, rtol=.03)
            current['State'] = actual['NextState']
            current['ConvState'] = actual['NextConvState']
            current['X'] = actual['Output']
        graph, evidence = lower_vector(tvm, replay.mod, hardware)
        assert all(tier == ('HBF' if name.startswith('W_') or name in ('A_log', 'dt_bias') else 'HBM')
                   for name, tier in evidence['tensor_tiers'].items())
        print('PASS GDN independent reference, two recurrent steps, trace replay, tiers; inline', inline, flush=True)
    # Native rfactor changes both reduction order and intermediate HBM layout.
    # Exercise independent state propagation, trace replay, and lowering.
    import json
    seed_choice = dict(output_tile=128, reduction_tile=32, warps=4, stages=1)
    seed = dict(projections={name: seed_choice for name in ('qkvz', 'out_proj', 'gate_up', 'down')},
                auxiliary=seed_choice)
    hierarchical = tvm.IRModule({'main': mod['main'].with_attr('gtsim.hierarchical_seed', json.dumps(seed))})
    for inline in (0, 1, 2):
        for splits in (1, 4, 16, 64):
            sch = tvm.s_tir.Schedule(hierarchical, debug_mask='all', seed=1)
            apply(tvm, sch, 'gdn_split', dict(ba_splits=splits, inline=inline))
            replay = tvm.s_tir.Schedule(hierarchical, debug_mask='all')
            tvm.s_tir.schedule.Trace.apply_json_to_schedule(trace_json(sch.trace), replay)
            tvm.ir.assert_structural_equal(sch.mod, replay.mod)
            current = {name: value.copy() for name, value in inputs.items()}
            for step in range(2):
                reference = gdn_reference(current['X'], current, **dims)
                actual = evaluate(tvm, tvm.ir.load_json(tvm.ir.save_json(sch.mod)), current)
                for name in ('Output', 'NextState', 'NextConvState'):
                    assert np.isfinite(actual[name].astype('float32')).all()
                    assert np.isfinite(reference[name].astype('float32')).all()
                    np.testing.assert_allclose(actual[name].astype('float32'), reference[name].astype('float32'), atol=.002, rtol=.03)
                current.update(State=actual['NextState'], ConvState=actual['NextConvState'], X=actual['Output'])
            graph, evidence = lower_vector(tvm, replay.mod, hardware)
            partials = {name: tier for name, tier in evidence['tensor_tiers'].items() if 'rf' in name}
            assert partials and set(partials.values()) == {'HBM'}, partials
            del graph
            print('PASS GDN rfactor two steps, replay, HBM partials; splits/inline', splits, inline, flush=True)
    # Reject placement mistakes at the TIR boundary, before emitting traffic.
    from serve_memory import KEY
    bad_tiers = json.loads(str(mod['main'].attrs[KEY]))
    bad_tiers['State'] = 'HBF'
    bad_mod = tvm.IRModule({'main': mod['main'].with_attr(KEY, json.dumps(bad_tiers))})
    try:
        lower_vector(tvm, bad_mod, hardware)
    except ValueError:
        pass
    else:
        raise RuntimeError('accepted recurrent state in HBF')

    # The FA2 rewrite must validate reduction initialization, not just updates.
    import tvm_ffi
    from serve_fa2 import compile_online
    dense = annotate(tvm, tvm.IRModule({'main': tvm.te.create_prim_func(
        semantics.fa2(heads_q=4, heads_kv=1, seq=17, dim=16))}), hardware['topology'])
    check = tvm.s_tir.Schedule(dense, debug_mask='all', seed=1)
    apply(tvm, check, 'fa2', dict(kv_tile=32, splits=16, warps=1, stages=1))
    compile_online(tvm, check.mod)
    def corrupt_max_initialization(store, mapper):
        store = mapper.default_mutate(store)
        if str(store.buffer.name) == 'row_max' and type(store.value).__name__ == 'FloatImm':
            return tvm.tirx.BufferStore(store.buffer, tvm.tirx.const(0.0, 'float32'), store.indices)
        return store
    corrupted = tvm_ffi.structural_mutate(check.mod, (tvm.tirx.BufferStore, corrupt_max_initialization))
    try:
        compile_online(tvm, corrupted)
    except ValueError:
        pass
    else:
        raise RuntimeError('accepted corrupted softmax max initialization')
    print('PASS fail-closed placement and attention initialization mutations', flush=True)
    for batch, hq, hkv, seq, splits in [(1, 4, 1, 17, 1), (1, 4, 1, 17, 3), (2, 8, 2, 35, 3)]:
        dim = 16
        jobs, group = batch * hkv, hq // hkv
        q = rng.normal(size=(jobs, group, dim)).astype(BF)
        k = rng.normal(size=(jobs, seq, dim)).astype(BF)
        v = rng.normal(size=k.shape).astype(BF)
        score = np.einsum('jhd,jnd->jhn', q.astype('float32'), k.astype('float32')) / np.sqrt(dim)
        probability = np.exp(score - score.max(-1, keepdims=True))
        probability /= probability.sum(-1, keepdims=True)
        reference = np.einsum('jhn,jnd->jhd', probability, v.astype('float32'))
        native = online_module(tvm, batch, hq, hkv, seq, dim, 32, splits, 1, 2)
        partial = np.empty((jobs, splits, group, dim), 'float32')
        lse = np.empty((jobs, splits, group), 'float32')
        output = np.empty(q.shape, dtype=BF)
        execute(tvm, native['main'], dict(Q=q, K=k, V=v, PO=partial, PL=lse))
        execute(tvm, native['combine'], dict(PO=partial, PL=lse, O=output))
        assert np.isfinite(output.astype('float32')).all() and np.isfinite(reference).all()
        np.testing.assert_allclose(output.astype('float32'), reference, atol=.01, rtol=.04)
        graph, evidence = lower_online(tvm, tvm.ir.load_json(tvm.ir.save_json(native)), hardware)
        assert set(evidence['tensor_tiers'].values()) == {'HBM'}
        assert not any(op[0].startswith('wgmma') or 'tma' in op[0] for op in graph.operations)
        print('PASS saved online FA2 TensorIR, dense reference, tails/empty splits/batch', batch, seq, splits, flush=True)


if __name__ == '__main__':
    main()
