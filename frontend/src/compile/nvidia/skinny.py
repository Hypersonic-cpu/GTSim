"""Small-M/grouped schedules use the same masked copy/MMA/store primitives."""

from dataclasses import replace
from parser import ast as A
from .hopper import index, dimension, loop, region


def select_skinny(op, flow, positions):
    p = flow.config
    mm, mn, mk = p.matrix_shape
    bm, bn, bk = p.tile_shape
    if op.kind not in (A.OpKind.COPY, A.OpKind.MATRIX, A.OpKind.STORE):
        return A.NativePattern(op.id, op.scope, (), op.dependencies)
    source = op.region
    r, c = (v.expression for v in source.offsets)
    if op.kind == A.OpKind.COPY:
        dtype, loops, shape = op.results[0].dtype, op.iterations, source.shape
        steps = (
            A.NativeStep(
                A.NativeOpcode.TMA_LOAD,
                A.ExecutionResource.LOAD,
                index(p.producer_warp),
                loops,
                source,
                dtype,
                dimension(
                    next(t for t in flow.tensors if t.id == source.tensor).shape[1]
                ),
                op.dependencies,
                (),
                (),
                op.guard,
                op.static_args,
            ),
        )
    elif op.kind == A.OpKind.MATRIX:
        loops = op.iterations + (
            loop("executor", p.consumer_warps),
            loop("n_fragment", bn // (p.consumer_warps * mn)),
            loop("k_fragment", bk // mk),
        )
        steps = (
            A.NativeStep(
                A.NativeOpcode.MMA,
                A.ExecutionResource.MATRIX,
                index("%executor"),
                loops,
                region(
                    source,
                    r,
                    f"({c}) + (%executor*{bn // (p.consumer_warps * mn)}+%n_fragment)*{mn}",
                    (mm, mn),
                ),
                op.results[0].dtype,
                None,
                op.dependencies,
                (),
                (),
                op.guard,
                op.static_args,
            ),
        )
    else:
        # Physical accumulator fragments, followed by only valid logical output rows.
        loops = (
            loop("executor", p.consumer_warps),
            loop("n_fragment", bn // (p.consumer_warps * mn)),
        )
        staging = A.NativeStep(
            A.NativeOpcode.SHARED_STORE,
            A.ExecutionResource.STORE,
            index("%executor"),
            loops,
            region(
                source,
                r,
                f"({c}) + (%executor*{bn // (p.consumer_warps * mn)}+%n_fragment)*{mn}",
                (mm, mn),
            ),
            op.cast,
            None,
            op.dependencies,
            (),
            (),
            op.guard,
            op.static_args,
        )
        final = A.NativeStep(
            A.NativeOpcode.TMA_STORE,
            A.ExecutionResource.STORE,
            index(p.producer_warp),
            op.iterations,
            source,
            op.cast,
            dimension(next(t for t in flow.tensors if t.id == source.tensor).shape[1]),
            (),
            (0,),
            (),
            op.guard,
            op.static_args,
        )
        steps = (staging, final)
    return A.NativePattern(op.id, op.scope, steps, op.dependencies)
