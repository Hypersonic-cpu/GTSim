"""The two explicit reference schedules, represented once as loop bodies."""

from ....ir.common import Tensor
from ....ir.flow import FlowOp, FlowKernel
from ....ir.mapping import Mapping
from ....ir.profiles import GEMM, FA3, SMALL_M
from ....ir.schedule import (
    TailPredicate,
    LoopVar as V,
    Phase as P,
    OpKind as K,
    Scope as S,
    ExecutorRole as E,
    BufferRole as B,
    OpRole as R,
    Relation as L,
    IndexExpr,
    TileId,
    IterationId,
    PipelineStage,
    SlotUse,
    Dependency,
    SymbolicRegion,
    TileLoop,
    PipelineRegion,
    ScheduleDomain,
    ValueId,
    SSAValue,
    InitialValue,
)


def index(var, scale=1):
    return IndexExpr(terms=((var, scale),))


def domain(family, dims, sms, profile=None, groups=()):
    profile = profile or (GEMM if family == "gemm" else FA3)
    if family == "gemm":
        m, n, k = dims
        tiles = ((m + profile.bm - 1) // profile.bm) * (
            (n + profile.bn - 1) // profile.bn
        )
        if groups:
            tiles = sum(
                (len(g.input_rows) + profile.bm - 1) // profile.bm for g in groups
            ) * ((n + profile.bn - 1) // profile.bn)
        ctas = tiles if groups else min(sms, tiles)
        rounds = (k + profile.bk - 1) // profile.bk
    else:
        batch, hq, hkv, q, kv = dims
        tiles = batch * hq * (q // FA3.q_tile)
        ctas = tiles
        rounds = kv // FA3.kv_tile
    waves = (tiles + ctas - 1) // ctas
    stage = PipelineStage(
        IterationId(index(V.K)),
        profile.stages,
        tuple(
            max(0, (rounds + profile.stages - 1 - slot) // profile.stages)
            for slot in range(profile.stages)
        ),
    )
    pipeline = PipelineRegion(
        TileLoop(V.K, rounds),
        profile.stages,
        TileId(IndexExpr(terms=((V.CTA, 1), (V.WAVE, ctas)))),
        stage,
    )
    return ScheduleDomain(
        dims,
        sms,
        ctas,
        tiles,
        waves,
        pipeline,
        (TileLoop(V.CTA, ctas), TileLoop(V.WAVE, waves, TailPredicate.TILE_EXISTS)),
        matrix_groups=groups,
    )


def mapping(family, profile=None):
    execute = (
        (E.PRODUCER, "TMA"),
        (E.MATRIX, "MMA" if profile == SMALL_M else "WGMMA"),
    )
    buffers = (B.AB, B.OUTPUT)
    if family == "fa3":
        execute += ((E.VECTOR, "SIMD"),)
        buffers = (B.Q, B.K, B.V, B.OUTPUT)
    return Mapping(execute=execute, store=tuple((b, "SMEM") for b in buffers))


def operations(family, d, profile=None, dtype="fp16"):
    p = profile or (GEMM if family == "gemm" else FA3)
    stage = d.pipeline.stage

    def dep(source, relation, phase=P.COMPLETE):
        return Dependency(source, phase, relation)

    def op(
        role,
        kind,
        executor,
        dependencies=(),
        operands=(),
        region=None,
        buffer=None,
        scope=S.PIPELINE,
    ):
        values = tuple(ValueId(producer) for producer in operands)
        if role == R.MATMUL:
            values += (
                ValueId(R.MATMUL, iteration_delta=-1, initial=InitialValue.ZERO),
            )
        if role == R.MERGE_MAX:
            values += (
                ValueId(
                    R.MERGE_MAX,
                    iteration_delta=-1,
                    initial=InitialValue.NEGATIVE_INFINITY,
                ),
            )
        if role == R.LOGSUM:
            values += (
                ValueId(R.LOGSUM, iteration_delta=-1, initial=InitialValue.ZERO),
            )
        if role == R.SCALE_O:
            values += (ValueId(R.PV, iteration_delta=-1, initial=InitialValue.ZERO),)
        result_shape = region.shape if region is not None else (FA3.q_tile,)
        if role in (R.QK, R.CAST):
            result_shape = (FA3.q_tile, FA3.kv_tile)
        if role in (R.PV, R.SCALE_O, R.NORMALIZE):
            result_shape = (FA3.q_tile, FA3.head_dim)
        if role in (R.EXP0, R.EXP1):
            result_shape = (FA3.q_tile, FA3.kv_tile // 2)
        results = (
            ()
            if kind in (K.ACQUIRE, K.RELEASE, K.STORE)
            else (
                SSAValue(
                    ValueId(role),
                    result_shape,
                    dtype if kind == K.COPY or role == R.CAST else "fp32",
                ),
            )
        )
        return FlowOp(
            role.value,
            role,
            kind,
            scope,
            executor,
            values,
            tuple(dependencies),
            region,
            SlotUse(buffer, stage) if buffer else None,
            results,
        )

    if family == "gemm":
        a = SymbolicRegion("A", (index(V.ROW, p.bm), index(V.K, p.bk)), (p.bm, p.bk))
        b = SymbolicRegion("B", (index(V.K, p.bk), index(V.COLUMN, p.bn)), (p.bk, p.bn))
        c = SymbolicRegion(
            "C", (index(V.ROW, p.bm), index(V.COLUMN, p.bn)), (p.bm, p.bn)
        )
        reuse = dep(R.MATMUL, L.BUFFER_REUSE)
        loads = (reuse, dep(R.LOAD_A, L.PREVIOUS_LOAD, P.ISSUE))
        return (
            op(R.ACQUIRE, K.ACQUIRE, E.PRODUCER, (reuse,), buffer=B.AB),
            op(R.LOAD_A, K.COPY, E.PRODUCER, loads, region=a, buffer=B.AB),
            op(R.LOAD_B, K.COPY, E.PRODUCER, loads, region=b, buffer=B.AB),
            op(
                R.MATMUL,
                K.MATMUL,
                E.MATRIX,
                (
                    dep(R.LOAD_A, L.LOAD_A, P.READY),
                    dep(R.LOAD_B, L.LOAD_B, P.READY),
                    dep(R.MATMUL, L.PREVIOUS_FRAGMENT),
                    dep(R.STORE, L.PREVIOUS_TILE_STORE),
                ),
                (R.LOAD_A, R.LOAD_B),
                c,
            ),
            op(
                R.RELEASE,
                K.RELEASE,
                E.PRODUCER,
                (dep(R.MATMUL, L.WARP_TAIL),),
                buffer=B.AB,
            ),
            op(
                R.STORE,
                K.STORE,
                E.PRODUCER,
                (
                    dep(R.MATMUL, L.WARP_TAIL),
                    dep(R.MATMUL, L.FINAL_ROUND),
                    dep(R.STORE, L.PREVIOUS_CHUNK, P.ISSUE),
                ),
                (R.MATMUL,),
                c,
                scope=S.EPILOGUE,
            ),
        )
    q = SymbolicRegion("Q", (index(V.ROW, 128), IndexExpr()), (128, 128))
    kv = lambda role: SymbolicRegion(role, (IndexExpr(), index(V.K, 128)), (128, 128))
    o = SymbolicRegion("O", (index(V.ROW, 128), IndexExpr()), (128, 128))
    result = [
        op(R.ACQUIRE, K.ACQUIRE, E.PRODUCER, buffer=B.K),
        op(R.LOAD_Q, K.COPY, E.PRODUCER, region=q, buffer=B.Q, scope=S.PROLOGUE),
        op(
            R.LOAD_K,
            K.COPY,
            E.PRODUCER,
            (dep(R.QK, L.BUFFER_REUSE),),
            region=kv("K"),
            buffer=B.K,
        ),
        op(
            R.LOAD_V,
            K.COPY,
            E.PRODUCER,
            (dep(R.PV, L.BUFFER_REUSE),),
            region=kv("V"),
            buffer=B.V,
        ),
        op(
            R.QK,
            K.MATMUL,
            E.MATRIX,
            (
                dep(R.LOAD_Q, L.LOAD_Q, P.READY),
                dep(R.LOAD_K, L.LOAD_K, P.READY),
                dep(R.QK, L.PREVIOUS_FRAGMENT),
                dep(R.LOGSUM, L.PREVIOUS_SCORE, P.ISSUE),
                dep(R.CAST, L.PREVIOUS_CAST, P.ISSUE),
            ),
            (R.LOAD_Q, R.LOAD_K),
        ),
    ]

    def compute(role, source0, source1=None, kind=K.ELEMENTWISE, extra=()):
        deps = (
            (dep(source0, L.SOURCE0),)
            + ((dep(source1, L.SOURCE1),) if source1 else ())
            + tuple(extra)
        )
        result.append(
            op(role, kind, E.VECTOR, deps, (source0,) + ((source1,) if source1 else ()))
        )

    for local, shuffle, reduce in (
        (R.MAX0, R.SHFL_MAX0, R.REDUCE_MAX0),
        (R.MAX1, R.SHFL_MAX1, R.REDUCE_MAX1),
    ):
        compute(
            local, R.QK, kind=K.REDUCE_MAX, extra=(dep(R.PV, L.PREVIOUS_PV, P.ISSUE),)
        )
        compute(shuffle, local)
        compute(reduce, shuffle, kind=K.REDUCE_MAX)
    compute(R.MERGE_MAX, R.REDUCE_MAX0, R.REDUCE_MAX1, K.REDUCE_MAX)
    compute(R.SCALE, R.MERGE_MAX)
    compute(R.EXP0, R.SCALE)
    compute(R.EXP1, R.SCALE)
    for local, shuffle, reduce, source in (
        (R.SUM0, R.SHFL_SUM0, R.REDUCE_SUM0, R.EXP0),
        (R.SUM1, R.SHFL_SUM1, R.REDUCE_SUM1, R.EXP1),
    ):
        compute(local, source, kind=K.REDUCE_SUM)
        compute(shuffle, local)
        compute(reduce, shuffle, kind=K.REDUCE_SUM)
    compute(R.MERGE_SUM, R.REDUCE_SUM0, R.REDUCE_SUM1, K.REDUCE_SUM)
    compute(R.LOGSUM, R.MERGE_SUM, R.SCALE)
    compute(R.SCALE_O, R.SCALE, extra=(dep(R.QK, L.NEXT_QK, P.ISSUE),))
    compute(R.CAST, R.EXP0, R.EXP1)
    result.append(
        op(
            R.PV,
            K.MATMUL,
            E.MATRIX,
            (
                dep(R.LOAD_V, L.LOAD_V, P.READY),
                dep(R.CAST, L.CAST),
                dep(R.SCALE_O, L.SCALE_O),
                dep(R.LOGSUM, L.LOGSUM),
                dep(R.PV, L.PREVIOUS_FRAGMENT),
            ),
            (R.LOAD_V, R.CAST, R.SCALE_O),
        )
    )
    result.append(
        op(
            R.NORMALIZE,
            K.ELEMENTWISE,
            E.VECTOR,
            (dep(R.PV, L.WARP_TAIL), dep(R.LOGSUM, L.LOGSUM), dep(R.PV, L.FINAL_ROUND)),
            (R.PV, R.LOGSUM),
            scope=S.EPILOGUE,
        )
    )
    result.append(
        op(
            R.STORE,
            K.STORE,
            E.PRODUCER,
            (dep(R.NORMALIZE, L.NORMALIZATION),),
            (R.NORMALIZE,),
            o,
            scope=S.EPILOGUE,
        )
    )
    result.append(
        op(R.RELEASE, K.RELEASE, E.PRODUCER, (dep(R.PV, L.WARP_TAIL),), buffer=B.V)
    )
    return tuple(result)


def build(algorithm, num_sms):
    algorithm.verify()
    if type(num_sms) is not int or num_sms < 1:
        raise ValueError("num_sms must be positive")
    family = algorithm.family
    dims = algorithm.dimensions
    profile = (
        SMALL_M
        if algorithm.schedule_id in ("small_m_mma", "grouped_small_m_mma")
        else (GEMM if family == "gemm" else FA3)
    )
    d = domain(family, dims, num_sms, profile, algorithm.groups)
    if family == "gemm":
        m, n, k = dims
        tensors = (
            Tensor("A", (m, k), algorithm.dtype, read_only=True),
            Tensor("B", (k, n), algorithm.dtype, read_only=True),
            Tensor("C", (m, n), algorithm.dtype),
        )
    else:
        b, hq, hkv, q, kv = dims
        tensors = (
            Tensor("Q", (b, hq, q, 128), read_only=True),
            Tensor("K", (b, hkv, 128, kv), read_only=True),
            Tensor("V", (b, hkv, 128, kv), read_only=True),
            Tensor("O", (b, hq, q, 128)),
        )
    if algorithm.groups:
        from ..gemm_groups import physical_tensors

        tensors = physical_tensors(algorithm)
    from ....serialization.json_codec import digest

    flow = FlowKernel(
        symbol=algorithm.symbol,
        source_hash=digest(algorithm),
        family=family,
        schedule_id=algorithm.schedule_id
        if profile == SMALL_M
        else f"ref_h100_{family}_v0_1_0",
        profile=profile,
        domain=d,
        tensors=tensors,
        mapping=mapping(family, profile),
        ops=operations(family, d, profile, algorithm.dtype),
    )
    flow.verify()
    return flow
