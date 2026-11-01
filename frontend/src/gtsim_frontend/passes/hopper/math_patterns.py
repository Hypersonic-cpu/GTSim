"""Shared SIMT native patterns for typed scalar/vector/reduction math.

Native operand references and intra-pattern edges are explicit. The execution
lowerer does not recover arithmetic from Algorithm parameters or kernel names.
"""

from ...ir.algorithm import MathKind as M, MathOperation
from ...ir.arch import NativeStep, NativePattern
from ...ir.schedule import OpKind


def select_math(op, profile):
    steps = []

    def emit(
        opcode,
        shape,
        external=(),
        internal=(),
        kind=None,
        reduction=(),
        dtype="fp32",
        scale="",
        axes=None,
        result_axes=None,
        constants=None,
    ):
        inputs = tuple(op.math.inputs[index] for index in external) if op.math else ()
        inputs += tuple(f"step{index}" for index in internal)
        math = (
            MathOperation(
                kind,
                inputs,
                f"step{len(steps)}",
                reduction,
                dtype=dtype,
                scale_by_inverse_sqrt=scale,
                constants=op.math.constants if constants is None else tuple(constants),
                parameter=op.math.parameter,
                output_axes=tuple(
                    result_axes if result_axes is not None else op.results[0].axes
                ),
            )
            if kind
            else None
        )
        steps.append(
            NativeStep(
                opcode,
                op.dependencies if not steps else (),
                tuple(shape),
                operands=tuple(op.operands[index] for index in external),
                math=math,
                input_steps=tuple(internal),
                fragment_axes=tuple(
                    axes
                    if axes is not None
                    else (op.math.output_axes[-2:] if op.math else ())
                ),
            )
        )

    if op.kind == OpKind.COPY:
        emit("cp.dram2sram_tma", op.region.shape)
    elif op.kind == OpKind.STORE:
        emit("st.reg2sram", op.region.shape, tuple(range(len(op.operands))))
        emit("cp.sram2dram_tma", op.region.shape, internal=(0,))
    else:
        if op.math is None:
            raise ValueError("SIMT compute requires typed mathematical semantics")
        kind = op.math.kind
        shape = op.results[0].shape
        if kind in (M.ZERO, M.NEG_INFINITY, M.CONSTANT, M.PARAMETER, M.IOTA):
            emit("simd.fill", shape, kind=kind, dtype=op.math.dtype)
        elif kind == M.TOPK:
            count = int(dict(op.math.constants)["top_k"])
            extent = op.reduction_shape[-1]
            maxima = []
            indices = []
            current = None
            for _ in range(count):
                ext = (0,) if current is None else ()
                prior = () if current is None else (current,)
                maxima.append(len(steps))
                emit(
                    "reduce_max",
                    (1, extent),
                    ext,
                    prior,
                    M.MAX_REDUCTION,
                    reduction=("D",),
                    result_axes=("Q", "STATE"),
                )
                indices.append(len(steps))
                emit(
                    "simd.min",
                    (1, extent),
                    ext,
                    (*prior, maxima[-1]),
                    M.ARGMAX,
                    dtype="int32",
                    result_axes=("Q", "STATE"),
                )
                current = len(steps)
                emit(
                    "simd.fill",
                    (1, extent),
                    ext,
                    (*prior, indices[-1]),
                    M.MASK_SELECTED,
                )
            emit(
                "simd.identity",
                (1, 2 * count),
                internal=tuple(maxima + indices),
                kind=M.TOPK_PACK,
            )
        elif kind == M.HISTOGRAM:
            bins = int(dict(op.math.constants)["bins"])
            count = 1
            for n in op.reduction_shape:
                count *= n
            emit("simd.fill", (1, bins), kind=M.ZERO, dtype="int32")
            emit("simd.add", (count, 1), (0,), (0,), M.HISTOGRAM, dtype="int32")
        elif kind == M.PREFIX_SUM:
            extent = op.reduction_shape[-1]
            previous = None
            stride = 1
            while stride < extent:
                ext = (0,) if previous is None else ()
                prior = () if previous is None else (previous,)
                shifted = len(steps)
                emit(
                    "shfl",
                    (1, extent),
                    ext,
                    prior,
                    M.SHIFT_RIGHT,
                    dtype="int32",
                    constants=(("stride", float(stride)),),
                )
                previous = len(steps)
                emit(
                    "simd.add",
                    (1, extent),
                    ext,
                    (*prior, shifted),
                    M.ADD,
                    dtype="int32",
                )
                stride *= 2
            emit(
                "simd.identity",
                (1, extent + 1),
                (0,) if previous is None else (),
                () if previous is None else (previous,),
                M.EXCLUSIVE_SCAN,
                dtype="int32",
            )
        elif kind == M.PACK_VECTOR:
            emit(
                "simd.identity",
                shape,
                tuple(range(len(op.operands))),
                kind=kind,
                dtype=op.math.dtype,
            )
        elif kind == M.MATMUL:
            sequence_axis = (
                "KV" if op.math.reduction_axes == ("D",) else op.math.reduction_axes[0]
            )
            domain = (profile.kv_tile, profile.head_dim)
            emit(
                "simd.mul",
                domain,
                (0, 1),
                kind=M.MULTIPLY,
                axes=(sequence_axis, "D"),
                result_axes=(sequence_axis, "D"),
            )
            emit(
                "reduce_sum",
                domain,
                internal=(0,),
                kind=M.SUM_REDUCTION,
                reduction=op.math.reduction_axes,
                axes=(sequence_axis, "D"),
            )
            if op.math.scale_by_inverse_sqrt:
                emit(
                    "simd.mul",
                    shape,
                    internal=(1,),
                    kind=M.SCALE,
                    scale=op.math.scale_by_inverse_sqrt,
                )
            elif len(op.operands) == 3:
                emit("simd.add", shape, (2,), (1,), M.ADD)
        elif kind in (M.MAX_REDUCTION, M.SUM_REDUCTION):
            emit(
                "reduce_max" if kind == M.MAX_REDUCTION else "reduce_sum",
                op.reduction_shape or (profile.kv_tile, 1),
                (0,),
                kind=kind,
                reduction=op.math.reduction_axes,
                axes=(op.math.reduction_axes[0], "STATE"),
            )
        elif kind == M.STREAMING_MAX:
            emit(
                "reduce_max",
                (1, profile.kv_tile),
                (0,),
                kind=M.MAX_REDUCTION,
                reduction=("KV",),
                axes=("Q", "KV"),
            )
            emit("simd.max", shape, (1,), (0,), M.MAXIMUM)
        elif kind == M.EXP:
            if len(op.operands) == 1:
                emit("sfu.exp", shape, (0,), kind=M.EXP)
            else:
                emit("simd.sub", shape, (0, 1), kind=M.SUBTRACT)
                emit("sfu.exp", shape, internal=(0,), kind=M.EXP)
        elif kind == M.STREAMING_SUM:
            emit(
                "reduce_sum",
                (1, profile.kv_tile),
                (0,),
                kind=M.SUM_REDUCTION,
                reduction=("KV",),
                axes=("Q", "KV"),
            )
            emit("simd.mul", shape, (1, 2), kind=M.MULTIPLY)
            emit("simd.add", shape, internal=(0, 1), kind=M.ADD)
        elif kind == M.RESCALE:
            emit("simd.mul", shape, (0, 1), kind=M.MULTIPLY)
        elif kind == M.NORMALIZE:
            emit("sfu.divide", shape, (0, 1), kind=M.DIVIDE)
        elif kind in (
            M.ADD,
            M.SUBTRACT,
            M.MULTIPLY,
            M.DIVIDE,
            M.MAXIMUM,
            M.RSQRT,
            M.SIN,
            M.COS,
            M.LOG,
        ):
            codes = {
                M.ADD: "simd.add",
                M.SUBTRACT: "simd.sub",
                M.MULTIPLY: "simd.mul",
                M.DIVIDE: "sfu.divide",
                M.MAXIMUM: "simd.max",
                M.RSQRT: "sfu.rsqrt",
                M.SIN: "sfu.sin",
                M.COS: "sfu.cos",
                M.LOG: "sfu.log",
            }
            emit(
                codes[kind],
                shape,
                tuple(range(len(op.operands))),
                kind=kind,
                dtype=op.math.dtype,
            )
        elif kind == M.CAST:
            emit("simd.cast", shape, (0,), kind=M.CAST, dtype=op.math.dtype)
        elif kind == M.STATE_LSE:
            emit("sfu.log", shape, (1,), kind=M.LOG)
            emit("simd.add", shape, (0,), (0,), M.ADD)
        else:
            raise ValueError(f"no shared SIMT pattern for {kind}")
    return NativePattern(op, tuple(steps))
