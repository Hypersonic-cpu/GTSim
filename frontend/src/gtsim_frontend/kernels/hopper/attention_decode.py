"""Tq=1 schedule specialization of the canonical FA3 math contract.

The scheduled body is constant size. Request/head/split partitions hold only
iteration bounds; no execution instructions are expanded here.
"""

from ...ir.algorithm import MathKind, MathOperation
from ...ir.common import Tensor
from ...ir.flow import FlowKernel, FlowOp
from ...ir.profiles import DECODE
from ...ir.schedule import (
    OpRole as R,
    OpKind as K,
    Scope as S,
    ExecutorRole as E,
    Relation as L,
    Phase as P,
    BufferRole as B,
    Dependency,
    ValueId,
    SSAValue,
    InitialValue,
    IndexExpr,
    LoopVar as V,
    SymbolicRegion,
    TileLoop,
    PipelineRegion,
    PipelineStage,
    IterationId,
    TileId,
    ScheduleDomain,
    AttentionPartition,
    LoopState,
    SlotUse,
    WorkGroup,
)
from ...ir.mapping import Mapping
from ...serialization.json_codec import digest


def partitions(algorithm):
    _, hq, hkv, _, _ = algorithm.dimensions
    splits = algorithm.split_count
    return tuple(
        AttentionPartition(
            b, h, h // (hq // hkv), s, length * s // splits, length * (s + 1) // splits
        )
        for b, length in enumerate(algorithm.valid_lengths)
        for h in range(hq)
        for s in range(splits)
    )


def physical_tensors(algorithm):
    batch, hq, hkv, _, _ = algorithm.dimensions
    dtype = algorithm.dtype
    result = [Tensor("Q", (batch * hq, 128), dtype, read_only=True)]
    for b, length in enumerate(algorithm.valid_lengths):
        for h in range(hkv):
            result.extend(
                Tensor(f"{letter}_r{b}_h{h}", (length, 128), dtype, read_only=True)
                for letter in ("K", "V")
            )
    if algorithm.split_count > 1:
        rows = batch * hq * algorithm.split_count
        result.extend(
            (Tensor("Partial", (rows, 128), "fp32"), Tensor("LSE", (rows, 1), "fp32"))
        )
    else:
        result.append(Tensor("O", (batch, hq * 128), dtype))
    return tuple(result)


def instantiate(algorithm, num_sms=132):
    algorithm.verify()
    ps = partitions(algorithm)
    rounds = max((p.end - p.start + DECODE.kv_tile - 1) // DECODE.kv_tile for p in ps)
    stage = PipelineStage(IterationId(IndexExpr(terms=((V.K, 1),))), 1, (rounds,))
    domain = ScheduleDomain(
        algorithm.dimensions,
        num_sms,
        len(ps),
        len(ps),
        1,
        PipelineRegion(
            TileLoop(V.K, rounds), 1, TileId(IndexExpr(terms=((V.CTA, 1),))), stage
        ),
        (TileLoop(V.CTA, len(ps)),),
        partitions=ps,
        workgroups=tuple(
            WorkGroup(
                bindings=(
                    (V.BATCH, p.request),
                    (V.HEAD, p.query_head),
                    (V.KV_HEAD, p.kv_head),
                    (V.KV_START, p.start),
                    (V.SPLIT, p.split),
                ),
                aliases=(
                    ("K", f"K_r{p.request}_h{p.kv_head}"),
                    ("V", f"V_r{p.request}_h{p.kv_head}"),
                ),
                ranges=(("KV", p.start, p.end),),
            )
            for p in ps
        ),
        states=(
            LoopState(R.MERGE_MAX, R.INIT_MAX),
            LoopState(R.LOGSUM, R.INIT_SUM),
            LoopState(R.PV, R.INIT_OUTPUT),
        ),
    )
    b, hq, _, _, _ = algorithm.dimensions
    qrow = IndexExpr(terms=((V.BATCH, hq), (V.HEAD, 1)))
    kvrow = IndexExpr(terms=((V.KV_START, 1), (V.K, DECODE.kv_tile)))
    zero = IndexExpr()
    regions = {
        "Q": SymbolicRegion("Q", (qrow, zero), (1, 128)),
        "K": SymbolicRegion("K", (kvrow, zero), (DECODE.kv_tile, 128), valid_axis="KV"),
        "V": SymbolicRegion("V", (kvrow, zero), (DECODE.kv_tile, 128), valid_axis="KV"),
    }
    ops = []

    def append(
        role, kind, scope, operands=(), shape=(1, 1), math=None, region=None, deps=()
    ):
        ops.append(
            FlowOp(
                role.value,
                role,
                kind,
                scope,
                E.PRODUCER if kind in (K.COPY, K.STORE) else E.VECTOR,
                tuple(operands),
                tuple(deps),
                region,
                slot=SlotUse({R.LOAD_Q: B.Q, R.LOAD_K: B.K, R.LOAD_V: B.V}[role], stage)
                if role in (R.LOAD_Q, R.LOAD_K, R.LOAD_V)
                else None,
                results=()
                if kind == K.STORE
                else (
                    SSAValue(
                        ValueId(role),
                        shape,
                        algorithm.dtype
                        if kind == K.COPY or role in (R.CAST, R.OUTPUT_CAST)
                        else "fp32",
                        axes=("KV", "D")
                        if role in (R.LOAD_K, R.LOAD_V)
                        else ("Q", "KV")
                        if role in (R.QK, R.EXP0, R.CAST)
                        else ("Q", "D")
                        if role
                        in (
                            R.LOAD_Q,
                            R.INIT_OUTPUT,
                            R.SCALE_O,
                            R.PV,
                            R.NORMALIZE,
                            R.OUTPUT_CAST,
                        )
                        else ("Q", "STATE"),
                    ),
                ),
                math=math,
            )
        )

    for name, role, shape, kind in (
        ("m", R.INIT_MAX, (1, 1), MathKind.NEG_INFINITY),
        ("l", R.INIT_SUM, (1, 1), MathKind.ZERO),
        ("acc", R.INIT_OUTPUT, (1, 128), MathKind.ZERO),
    ):
        append(
            role,
            K.ELEMENTWISE,
            S.PROLOGUE,
            shape=shape,
            math=MathOperation(kind, (), name),
        )
    append(R.LOAD_Q, K.COPY, S.PROLOGUE, shape=(1, 128), region=regions["Q"])
    for letter, role, consumer in (("K", R.LOAD_K, R.QK), ("V", R.LOAD_V, R.PV)):
        append(
            role,
            K.COPY,
            S.PIPELINE,
            shape=(DECODE.kv_tile, 128),
            region=regions[letter],
            deps=(Dependency(consumer, P.COMPLETE, L.BUFFER_REUSE),),
        )
    roles = {
        "scores": R.QK,
        "m_next": R.MERGE_MAX,
        "alpha": R.SCALE,
        "P": R.EXP0,
        "l_next": R.LOGSUM,
        "scaled_acc": R.SCALE_O,
        "P16": R.CAST,
        "acc_next": R.PV,
        "normalized": R.NORMALIZE,
        "O": R.OUTPUT_CAST,
    }
    inputs = {
        "Q": ValueId(R.LOAD_Q),
        "K": ValueId(R.LOAD_K),
        "V": ValueId(R.LOAD_V),
        "m": ValueId(
            R.MERGE_MAX, iteration_delta=-1, initial=InitialValue.NEGATIVE_INFINITY
        ),
        "l": ValueId(R.LOGSUM, iteration_delta=-1, initial=InitialValue.ZERO),
        "acc": ValueId(R.PV, iteration_delta=-1, initial=InitialValue.ZERO),
    }
    for math in algorithm.contract.operations:
        if math.output not in roles or (
            algorithm.split_count > 1 and math.output == "O"
        ):
            continue
        role = roles[math.output]
        values = tuple(inputs[name] for name in math.inputs)
        shape = (
            (1, DECODE.kv_tile)
            if math.output in ("scores", "P", "P16")
            else (
                (1, 128)
                if math.output in ("scaled_acc", "acc_next", "normalized", "O")
                else (1, 1)
            )
        )
        scope = S.EPILOGUE if role in (R.NORMALIZE, R.OUTPUT_CAST) else S.PIPELINE
        kind = K.MATMUL if math.kind == MathKind.MATMUL else K.ELEMENTWISE
        append(role, kind, scope, values, shape, math)
        inputs[math.output] = ValueId(role)
    if algorithm.split_count > 1:
        row = IndexExpr(
            terms=(
                (V.BATCH, hq * algorithm.split_count),
                (V.HEAD, algorithm.split_count),
                (V.SPLIT, 1),
            )
        )
        append(
            R.PARTIAL_LSE,
            K.ELEMENTWISE,
            S.EPILOGUE,
            (ValueId(R.MERGE_MAX), ValueId(R.LOGSUM)),
            math=MathOperation(MathKind.STATE_LSE, ("m_next", "l_next"), "lse"),
        )
        append(
            R.STORE_LSE,
            K.STORE,
            S.EPILOGUE,
            (ValueId(R.PARTIAL_LSE),),
            region=SymbolicRegion("LSE", (row, zero), (1, 1)),
        )
        output_region = SymbolicRegion("Partial", (row, zero), (1, 128))
        # Partial outputs retain FP32 instead of rounding before LSE merge.
        output_value = ValueId(R.NORMALIZE)
    else:
        output_region = SymbolicRegion(
            "O",
            (IndexExpr(terms=((V.BATCH, 1),)), IndexExpr(terms=((V.HEAD, 128),))),
            (1, 128),
        )
        output_value = ValueId(R.OUTPUT_CAST)
    append(R.STORE, K.STORE, S.EPILOGUE, (output_value,), region=output_region)
    flow = FlowKernel(
        symbol=algorithm.symbol,
        source_hash=digest(algorithm),
        family="fa3",
        schedule_id="decode_simt",
        profile=DECODE,
        domain=domain,
        tensors=physical_tensors(algorithm),
        mapping=Mapping(
            execute=((E.PRODUCER, "TMA"), (E.MATRIX, "SIMD"), (E.VECTOR, "SIMD")),
            store=tuple((buffer, "SMEM") for buffer in (B.Q, B.K, B.V, B.OUTPUT)),
        ),
        ops=tuple(ops),
    )
    return flow
