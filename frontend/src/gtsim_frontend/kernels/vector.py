"""Compact canonical math/Flow plans for supported vector primitives."""

import math
from ..ir.algorithm import (
    AlgorithmContract,
    OperandContract,
    IndexDomain,
    MathOperation,
    MathKind as M,
)
from ..ir.primitives import (
    NormSpec,
    AddSpec,
    SwiGLUSpec,
    RopeSpec,
    ExtractSpec,
    KVUpdateSpec,
    RouterSpec,
    HistogramSpec,
    GatherSpec,
    WeightedSpec,
)
from dataclasses import replace
from ..ir.common import Tensor, require
from ..ir.flow import FlowOp, FlowKernel
from ..ir.profiles import ReferenceProfile
from ..ir.mapping import Mapping
from ..ir.schedule import (
    Scope as S,
    ExecutorRole as E,
    OpKind as K,
    ValueId,
    SSAValue,
    SymbolicRegion,
    IndexExpr,
    LoopVar as V,
    WorkGroup,
    TileLoop,
    PipelineRegion,
    PipelineStage,
    IterationId,
    TileId,
    ScheduleDomain,
    BufferRole,
)
from ..serialization.json_codec import digest


def verify_spec(spec):
    if isinstance(spec, (NormSpec, AddSpec, SwiGLUSpec)):
        require(spec.rows > 0 and spec.width > 0, "vector", "invalid row/width domain")
        if isinstance(spec, NormSpec):
            require(
                spec.epsilon > 0 and math.isfinite(spec.epsilon),
                "rmsnorm",
                "invalid epsilon",
            )
    elif isinstance(spec, RopeSpec):
        require(
            spec.batch > 0
            and spec.heads > 0
            and spec.head_dim > 0
            and spec.head_dim % 2 == 0,
            "rope",
            "invalid head geometry",
        )
        require(
            spec.input_offset >= 0
            and spec.input_offset + spec.heads * spec.head_dim <= spec.input_width,
            "rope",
            "input head region outside projection",
        )
        require(
            len(spec.positions) == spec.batch and all(p >= 0 for p in spec.positions),
            "rope",
            "missing request positions",
        )
        require(
            spec.theta > 0 and math.isfinite(spec.theta) and spec.epsilon > 0,
            "rope",
            "invalid theta/epsilon",
        )
    elif isinstance(spec, (ExtractSpec, KVUpdateSpec)):
        from .memory import verify_spec as verify_memory

        verify_memory(spec)
    elif isinstance(spec, (RouterSpec, HistogramSpec, GatherSpec, WeightedSpec)):
        from .moe import verify_spec as verify_moe

        verify_moe(spec)
    else:
        raise ValueError("unsupported typed vector primitive")


def signature(spec):
    verify_spec(spec)
    if isinstance(spec, (ExtractSpec, KVUpdateSpec)):
        from .memory import signature as memory_signature

        return memory_signature(spec)
    if isinstance(spec, (RouterSpec, HistogramSpec, GatherSpec, WeightedSpec)):
        from .moe import signature as moe_signature

        return moe_signature(spec)
    if isinstance(spec, NormSpec):
        return (
            Tensor("X", (spec.rows, spec.width), "bf16", read_only=True),
            Tensor("Weight", (1, spec.width), "bf16", read_only=True),
            Tensor("Output", (spec.rows, spec.width), "bf16"),
        )
    if isinstance(spec, AddSpec):
        return (
            Tensor("X", (spec.rows, spec.width), "bf16", read_only=True),
            Tensor("Y", (spec.rows, spec.width), "bf16", read_only=True),
            Tensor("Output", (spec.rows, spec.width), "bf16"),
        )
    if isinstance(spec, SwiGLUSpec):
        return (
            Tensor("X", (spec.rows, 2 * spec.width), "bf16", read_only=True),
            Tensor("Output", (spec.rows, spec.width), "bf16"),
        )
    return (
        Tensor("Input", (spec.batch, spec.input_width), "bf16", read_only=True),
        Tensor("Weight", (1, spec.head_dim), "bf16", read_only=True),
        Tensor("Output", (spec.batch * spec.heads, spec.head_dim), "bf16"),
    )


def index(variable, coefficient=1):
    return IndexExpr(terms=((variable, coefficient),))


class Body:
    """Build typed operations, never native instructions or execution nodes."""

    def __init__(self):
        self.ops = []
        self.values = {}

    def load(self, name, tensor, row, column, width, dtype="bf16", height=1, axes=None):
        shape = (height, width)
        axes = axes or (("Q", "D") if width > 1 else ("Q", "STATE"))
        self.ops.append(
            FlowOp(
                name,
                name,
                K.COPY,
                S.PIPELINE,
                E.PRODUCER,
                region=SymbolicRegion(tensor, (row, column), shape),
                results=(SSAValue(ValueId(name), shape, dtype, axes),),
            )
        )
        self.values[name] = self.ops[-1].results[0]
        return ValueId(name)

    def math(
        self,
        name,
        kind,
        inputs=(),
        width=1,
        dtype="fp32",
        constants=(),
        parameter="",
        reduction=(),
    ):
        values = tuple(ValueId(v) if isinstance(v, str) else v for v in inputs)
        axes = ("Q", "D") if width > 1 else ("Q", "STATE")
        reduction_shape = ()
        if reduction:
            source = self.values[values[0].producer]
            reduction_shape = values[0].shape or source.shape
        op = MathOperation(
            kind,
            tuple(str(v.producer) for v in values),
            name,
            reduction,
            dtype,
            axes,
            constants=tuple(constants),
            parameter=parameter,
        )
        self.ops.append(
            FlowOp(
                name,
                name,
                K.ELEMENTWISE,
                S.PIPELINE,
                E.VECTOR,
                values,
                results=(SSAValue(ValueId(name), (1, width), dtype, axes),),
                math=op,
                reduction_shape=reduction_shape,
            )
        )
        self.values[name] = self.ops[-1].results[0]
        return ValueId(name)

    def constant(self, name, value):
        return self.math(name, M.CONSTANT, constants=(("value", float(value)),))

    def cast(self, name, value, width, dtype="fp32"):
        return self.math(name, M.CAST, (value,), width, dtype)

    def store(self, name, value, tensor, row, column, width, extra=()):
        self.ops.append(
            FlowOp(
                name,
                name,
                K.STORE,
                S.EPILOGUE,
                E.PRODUCER,
                (value, *extra),
                region=SymbolicRegion(
                    tensor,
                    (row, column),
                    (1, width),
                    row_index=str(extra[0].producer) if extra else "",
                ),
            )
        )

    def topk(self, name, value, count):
        source = self.values[value.producer]
        values = self.math(
            name, M.TOPK, (value,), count, constants=(("top_k", float(count)),)
        )
        op = self.ops[-1]
        self.ops[-1] = replace(
            op,
            reduction_shape=source.shape,
            results=(
                op.results[0],
                SSAValue(ValueId(name, 1), (1, count), "int32", ("Q", "D")),
            ),
        )
        return values, ValueId(name, 1)

    def norm(self, x, weight, width, epsilon):
        x = self.cast("x_fp32", x, width)
        w = self.cast("w_fp32", weight, width)
        squared = self.math("square", M.MULTIPLY, (x, x), width)
        summed = self.math("sum", M.SUM_REDUCTION, (squared,), reduction=("D",))
        divisor = self.constant("inverse_width", 1 / width)
        mean = self.math("mean", M.MULTIPLY, (summed, divisor))
        eps = self.constant("epsilon", epsilon)
        stable = self.math("variance_epsilon", M.ADD, (mean, eps))
        inverse = self.math("inverse_rms", M.RSQRT, (stable,))
        scaled = self.math("scaled", M.MULTIPLY, (x, inverse), width)
        return self.math("normalized", M.MULTIPLY, (scaled, w), width)


def plan(spec):
    if isinstance(spec, (ExtractSpec, KVUpdateSpec)):
        from .memory import plan as memory_plan

        return memory_plan(spec)
    if isinstance(spec, (RouterSpec, HistogramSpec, GatherSpec, WeightedSpec)):
        from .moe import plan as moe_plan

        return moe_plan(spec)
    ts = signature(spec)
    body = Body()
    z = IndexExpr()
    row = index(V.ROW)
    if isinstance(spec, (NormSpec, AddSpec, SwiGLUSpec)):
        rows, width = spec.rows, spec.width
        groups = tuple(WorkGroup(bindings=((V.ROW, r),)) for r in range(rows))
    else:
        rows, width = spec.batch * spec.heads, spec.head_dim
        groups = tuple(
            WorkGroup(
                bindings=(
                    (V.ROW, b * spec.heads + h),
                    (V.TOKEN, b),
                    (V.CHANNEL, spec.input_offset + h * spec.head_dim),
                ),
                constants=(("position", float(spec.positions[b])),),
            )
            for b in range(spec.batch)
            for h in range(spec.heads)
        )
    if isinstance(spec, NormSpec):
        x = body.load("load_x", "X", row, z, width)
        w = body.load("load_w", "Weight", z, z, width)
        out = body.norm(x, w, width, spec.epsilon)
        body.store(
            "store", body.cast("Output", out, width, "bf16"), "Output", row, z, width
        )
    elif isinstance(spec, AddSpec):
        x = body.cast("x_fp32", body.load("load_x", "X", row, z, width), width)
        y = body.cast("y_fp32", body.load("load_y", "Y", row, z, width), width)
        out = body.math("sum", M.ADD, (x, y), width)
        body.store(
            "store", body.cast("Output", out, width, "bf16"), "Output", row, z, width
        )
    elif isinstance(spec, SwiGLUSpec):
        gate = body.cast("gate_fp32", body.load("gate", "X", row, z, width), width)
        up = body.cast(
            "up_fp32", body.load("up", "X", row, IndexExpr(width), width), width
        )
        negative = body.constant("negative", -1)
        negated = body.math("negative_gate", M.MULTIPLY, (gate, negative), width)
        exponential = body.math("exp_gate", M.EXP, (negated,), width)
        one = body.constant("one", 1)
        denominator = body.math("sigmoid_denominator", M.ADD, (one, exponential), width)
        silu = body.math("silu", M.DIVIDE, (gate, denominator), width)
        out = body.math("product", M.MULTIPLY, (silu, up), width)
        body.store(
            "store", body.cast("Output", out, width, "bf16"), "Output", row, z, width
        )
    else:
        x = body.load("load_head", "Input", index(V.TOKEN), index(V.CHANNEL), width)
        w = body.load("load_weight", "Weight", z, z, width)
        normalized = body.norm(x, w, width, spec.epsilon)
        half = width // 2
        lo = ValueId(normalized.producer, offsets=(0, 0), shape=(1, half))
        hi = ValueId(normalized.producer, offsets=(0, half), shape=(1, half))
        channels = body.math("channels", M.IOTA, width=half)
        coefficient = body.constant(
            "frequency_coefficient", -2 * math.log(spec.theta) / width
        )
        exponent = body.math(
            "frequency_exponent", M.MULTIPLY, (channels, coefficient), half
        )
        frequencies = body.math("frequencies", M.EXP, (exponent,), half)
        position = body.math("position", M.PARAMETER, parameter="position")
        angle = body.math("angle", M.MULTIPLY, (frequencies, position), half)
        sine = body.math("sine", M.SIN, (angle,), half)
        cosine = body.math("cosine", M.COS, (angle,), half)
        lc = body.math("lower_cosine", M.MULTIPLY, (lo, cosine), half)
        hs = body.math("upper_sine", M.MULTIPLY, (hi, sine), half)
        lower = body.math("rotated_lower", M.SUBTRACT, (lc, hs), half)
        hc = body.math("upper_cosine", M.MULTIPLY, (hi, cosine), half)
        ls = body.math("lower_sine", M.MULTIPLY, (lo, sine), half)
        upper = body.math("rotated_upper", M.ADD, (hc, ls), half)
        body.store(
            "store_lower",
            body.cast("Output_lower", lower, half, "bf16"),
            "Output",
            row,
            z,
            half,
        )
        body.store(
            "store_upper",
            body.cast("Output_upper", upper, half, "bf16"),
            "Output",
            row,
            IndexExpr(half),
            half,
        )
    contract = AlgorithmContract(
        (IndexDomain("ROWS", rows), IndexDomain("D", width)),
        tuple(
            OperandContract(t.id, ("ROW", "COL"), t.dtype, not t.read_only) for t in ts
        ),
        tuple(op.math for op in body.ops if op.math is not None),
        output_dtype="bf16",
    )
    return ts, groups, tuple(body.ops), contract, rows, width


def contract_for(spec):
    return plan(spec)[3]


def instantiate(algorithm, num_sms=132):
    algorithm.verify()
    ts, groups, ops, contract, rows, width = plan(algorithm.primitive)
    stage = PipelineStage(IterationId(index(V.K)), 1, (1,))
    domain = ScheduleDomain(
        (rows, width),
        num_sms,
        len(groups),
        len(groups),
        1,
        PipelineRegion(TileLoop(V.K, 1), 1, TileId(index(V.CTA)), stage),
        (TileLoop(V.CTA, len(groups)),),
        workgroups=groups,
        partial_outputs=tuple(t.id for t in ts[2:])
        if isinstance(algorithm.primitive, KVUpdateSpec)
        else (),
    )
    profile = ReferenceProfile(
        "vector",
        q_tile=1,
        kv_tile=width,
        head_dim=width,
        stages=1,
        warps=4,
        consumer_warps=1,
        producer_warp=1,
        warpgroup=1,
        mma_m=1,
        mma_n=width,
    )
    flow = FlowKernel(
        symbol=algorithm.symbol,
        source_hash=digest(algorithm),
        family="vector",
        schedule_id="simt_vector",
        profile=profile,
        domain=domain,
        tensors=ts,
        ops=ops,
        contract=contract,
        mapping=Mapping(
            execute=((E.PRODUCER, "TMA"), (E.VECTOR, "SIMD")),
            store=((BufferRole.OUTPUT, "SMEM"),),
        ),
    )
    flow.verify()
    return flow
