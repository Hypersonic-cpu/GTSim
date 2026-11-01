"""Canonical FA3 split-output finalization schedule and mathematical contract."""

from dataclasses import replace
from ...ir.algorithm import (
    AlgorithmContract,
    IndexDomain,
    OperandContract,
    MathOperation,
    MathKind as M,
)
from ...ir.common import Tensor
from ...ir.flow import FlowKernel, FlowOp
from ...ir.profiles import DECODE
from ...ir.schedule import (
    OpRole as R,
    OpKind as K,
    Scope as S,
    ExecutorRole as E,
    BufferRole as B,
    ValueId,
    SSAValue,
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
    WorkGroup,
)
from ...ir.mapping import Mapping
from ...serialization.json_codec import digest


def contract(dimensions, dtype, splits):
    b, hq, _, _, _ = dimensions
    domains = tuple(
        IndexDomain(axis, n)
        for axis, n in (("B", b), ("HQ", hq), ("SPLIT", splits), ("D", 128))
    )
    operands = (
        OperandContract("Partial", ("B", "HQ", "SPLIT", "D"), "fp32"),
        OperandContract("LSE", ("B", "HQ", "SPLIT", "STATE"), "fp32"),
        OperandContract("O", ("B", "HQ", "Q", "D"), dtype, True),
    )
    ops = (
        MathOperation(
            M.MAX_REDUCTION, ("LSE",), "m", ("SPLIT",), output_axes=("Q", "STATE")
        ),
        MathOperation(M.EXP, ("LSE", "m"), "w", output_axes=("SPLIT", "STATE")),
        MathOperation(
            M.SUM_REDUCTION, ("w",), "den", ("SPLIT",), output_axes=("Q", "STATE")
        ),
        MathOperation(
            M.MATMUL, ("w", "Partial"), "numer", ("SPLIT",), output_axes=("Q", "D")
        ),
        MathOperation(
            M.NORMALIZE, ("numer", "den"), "normalized", output_axes=("Q", "D")
        ),
        MathOperation(
            M.CAST, ("normalized",), "O", dtype=dtype, output_axes=("Q", "D")
        ),
    )
    return AlgorithmContract(domains, operands, ops, output_dtype=dtype)


def physical_tensors(algorithm):
    batch, hq, _, _, _ = algorithm.dimensions
    rows = batch * hq * algorithm.split_count
    return (
        Tensor("Partial", (rows, 128), "fp32", read_only=True),
        Tensor("LSE", (rows, 1), "fp32", read_only=True),
        Tensor("O", (batch, hq * 128), algorithm.dtype),
    )


def instantiate(algorithm, num_sms=132):
    algorithm.verify()
    batch, hq, hkv, _, _ = algorithm.dimensions
    splits = algorithm.split_count
    profile = replace(DECODE, kv_tile=splits, mma_n=splits)
    parts = tuple(
        AttentionPartition(b, h, h // (hq // hkv), 0, 0, splits)
        for b in range(batch)
        for h in range(hq)
    )
    stage = PipelineStage(IterationId(IndexExpr(terms=((V.K, 1),))), 1, (1,))
    domain = ScheduleDomain(
        algorithm.dimensions,
        num_sms,
        len(parts),
        len(parts),
        1,
        PipelineRegion(
            TileLoop(V.K, 1), 1, TileId(IndexExpr(terms=((V.CTA, 1),))), stage
        ),
        (TileLoop(V.CTA, len(parts)),),
        partitions=parts,
        workgroups=tuple(
            WorkGroup(
                bindings=((V.BATCH, p.request), (V.HEAD, p.query_head)),
                ranges=(("SPLIT", 0, splits),),
            )
            for p in parts
        ),
    )
    row = IndexExpr(terms=((V.BATCH, hq * splits), (V.HEAD, splits)))
    zero = IndexExpr()
    ops = []
    for name, role, shape, axes in (
        ("Partial", R.LOAD_A, (splits, 128), ("SPLIT", "D")),
        ("LSE", R.LOAD_B, (splits, 1), ("SPLIT", "STATE")),
    ):
        ops.append(
            FlowOp(
                role.value,
                role,
                K.COPY,
                S.PROLOGUE,
                E.PRODUCER,
                region=SymbolicRegion(name, (row, zero), shape),
                results=(SSAValue(ValueId(role), shape, "fp32", axes),),
            )
        )
    producers = {"Partial": R.LOAD_A, "LSE": R.LOAD_B}
    roles = {
        "m": R.MERGE_MAX,
        "w": R.EXP0,
        "den": R.LOGSUM,
        "numer": R.PV,
        "normalized": R.NORMALIZE,
        "O": R.OUTPUT_CAST,
    }
    for math in algorithm.contract.operations:
        role = roles[math.output]
        shape = tuple(
            splits if axis == "SPLIT" else 128 if axis == "D" else 1
            for axis in math.output_axes
        )
        ops.append(
            FlowOp(
                role.value,
                role,
                K.MATMUL if math.kind == M.MATMUL else K.ELEMENTWISE,
                S.PIPELINE,
                E.VECTOR,
                tuple(ValueId(producers[name]) for name in math.inputs),
                results=(SSAValue(ValueId(role), shape, math.dtype, math.output_axes),),
                math=math,
            )
        )
        producers[math.output] = role
    ops.append(
        FlowOp(
            R.STORE.value,
            R.STORE,
            K.STORE,
            S.EPILOGUE,
            E.PRODUCER,
            (ValueId(R.OUTPUT_CAST),),
            region=SymbolicRegion(
                "O",
                (IndexExpr(terms=((V.BATCH, 1),)), IndexExpr(terms=((V.HEAD, 128),))),
                (1, 128),
            ),
        )
    )
    return FlowKernel(
        symbol=algorithm.symbol,
        source_hash=digest(algorithm),
        family="fa3",
        schedule_id="decode_combine",
        profile=profile,
        domain=domain,
        tensors=physical_tensors(algorithm),
        mapping=Mapping(
            execute=((E.PRODUCER, "TMA"), (E.MATRIX, "SIMD"), (E.VECTOR, "SIMD")),
            store=tuple((buffer, "SMEM") for buffer in (B.Q, B.K, B.V, B.OUTPUT)),
        ),
        ops=tuple(ops),
    )
