"""Small-M/grouped schedules use the same masked copy/MMA/store primitives."""

from dataclasses import replace
from parser import ast as A
from .hopper import index, dimension, loop, region


def tma_box(source):
    """Keep each native tensor-map box within the 256-element width limit."""
    width = source.shape[1]
    if isinstance(width, int) and width > 256:
        r, c = (v.expression for v in source.offsets)
        return (
            region(
                source,
                r,
                f"({c}) + %tma_chunk*256",
                (source.shape[0], f"min(256,{width}-%tma_chunk*256)"),
            ),
            (loop("tma_chunk", f"ceildiv({width},256)"),),
        )
    return source, ()


def select_skinny(op, flow, positions):
    p = flow.config
    mm, mn, mk = p.matrix_shape
    bm, bn, bk = p.tile_shape
    if op.kind not in (A.OpKind.COPY, A.OpKind.MATRIX, A.OpKind.STORE):
        return A.NativePattern(op.id, op.scope, (), op.dependencies)
    source = op.region
    r, c = (v.expression for v in source.offsets)
    m_fragments = (loop("m_fragment", bm // mm),) if bm > mm else ()
    fragment_row = f"({r}) + %m_fragment*{mm}" if m_fragments else r
    if op.kind == A.OpKind.COPY:
        box, chunks = tma_box(source)
        dtype, loops = op.results[0].dtype, op.iterations + chunks
        steps = (
            A.NativeStep(
                A.NativeOpcode.TMA_LOAD,
                A.ExecutionResource.LOAD,
                index(p.producer_warp),
                loops,
                box,
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
            *m_fragments,
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
                    fragment_row,
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
            *m_fragments,
        )
        staging = A.NativeStep(
            A.NativeOpcode.SHARED_STORE,
            A.ExecutionResource.STORE,
            index("%executor"),
            loops,
            region(
                source,
                fragment_row,
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
        box, chunks = tma_box(source)
        final = A.NativeStep(
            A.NativeOpcode.TMA_STORE,
            A.ExecutionResource.STORE,
            index(p.producer_warp),
            op.iterations + chunks,
            box,
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
