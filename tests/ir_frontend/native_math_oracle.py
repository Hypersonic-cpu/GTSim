"""Small CPU interpreter for canonical NativeStep math, never simulator output.

Uses ideal real-valued casts to test the algebra independently of BF16 rounding
and timing calibration. Operand axes come from the serialized IR.
"""

from itertools import product
import math
import struct
from gtsim_frontend.ir.algorithm import MathKind as M
from gtsim_frontend.ir.schedule import Scope, OpKind, ValueId


def tensor(axes, shape, fn):
    return (
        tuple(axes),
        tuple(shape),
        {index: fn(index) for index in product(*(range(size) for size in shape))},
    )


def matrix(rows, axes):
    return tensor(axes, (len(rows), len(rows[0])), lambda i: rows[i[0]][i[1]])


def round_dtype(value, dtype):
    """IEEE round-to-nearest-even storage; integer index operations stay exact."""
    if dtype == "int32":
        return int(value)
    value = struct.unpack("f", struct.pack("f", value))[0]
    if dtype == "bf16":
        bits = struct.unpack("I", struct.pack("f", value))[0]
        if math.isfinite(value):
            bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        value = struct.unpack("f", struct.pack("I", bits))[0]
    return value


def step_math(step, inputs, shape, parameters=None, precision=False):
    axes = step.math.output_axes
    extents = {}
    for input_axes, sizes, _ in inputs:
        for axis, size in zip(input_axes, sizes):
            extents[axis] = max(extents.get(axis, 1), size)
    sizes = tuple(
        extents.get(
            axis, 1 if axis in ("Q", "STATE") else shape[i] if i < len(shape) else 1
        )
        for i, axis in enumerate(axes)
    )

    if step.math.kind == M.TOPK_PACK:
        count = int(dict(step.math.constants)["top_k"])
        return tuple(
            tensor(
                ("Q", "D"),
                (1, count),
                lambda i, offset=offset: next(iter(inputs[offset + i[1]][2].values())),
            )
            for offset in (0, count)
        )
    if step.math.kind == M.EXCLUSIVE_SCAN:
        sizes = (1, inputs[0][1][1] + 1)

    def get(value, coords):
        names, dims, data = value
        return data[
            tuple(0 if size == 1 else coords[axis] for axis, size in zip(names, dims))
        ]

    kind = step.math.kind

    def evaluate(index):
        coords = dict(zip(axes, index))
        if kind == M.HISTOGRAM:
            return sum(int(value) == index[-1] for value in inputs[0][2].values())
        if kind == M.ARGMAX:
            selected = next(iter(inputs[1][2].values()))
            return next(
                j for j in range(inputs[0][1][1]) if inputs[0][2][0, j] == selected
            )
        if kind == M.MASK_SELECTED:
            selected = int(next(iter(inputs[1][2].values())))
            return -math.inf if index[-1] == selected else inputs[0][2][index]
        if kind == M.SHIFT_RIGHT:
            stride = int(dict(step.math.constants)["stride"])
            return (
                0 if index[-1] < stride else inputs[0][2][index[0], index[-1] - stride]
            )
        if kind == M.EXCLUSIVE_SCAN:
            return 0 if index[-1] == 0 else inputs[0][2][0, index[-1] - 1]
        if kind == M.PACK_VECTOR:
            return next(iter(inputs[index[-1]][2].values()))
        if kind == M.CONSTANT:
            return dict(step.math.constants)["value"]
        if kind == M.PARAMETER:
            return parameters[step.math.parameter]
        if kind == M.IOTA:
            return float(index[-1])
        if kind == M.ZERO:
            return 0.0
        if kind == M.NEG_INFINITY:
            return -math.inf
        if kind in (M.SUM_REDUCTION, M.MAX_REDUCTION):
            reduction = step.math.reduction_axes
            terms = [
                get(inputs[0], {**coords, **dict(zip(reduction, idx))})
                for idx in product(*(range(extents[axis]) for axis in reduction))
            ]
            return sum(terms) if kind == M.SUM_REDUCTION else max(terms)
        values = [get(value, coords) for value in inputs]
        if kind == M.MULTIPLY:
            return values[0] * values[1]
        if kind == M.ADD:
            return sum(values)
        if kind == M.SUBTRACT:
            return values[0] - values[1]
        if kind == M.DIVIDE:
            return values[0] / values[1]
        if kind == M.MAXIMUM:
            return max(values)
        if kind == M.EXP:
            return math.exp(values[0])
        if kind == M.RSQRT:
            return 1 / math.sqrt(values[0])
        if kind == M.SIN:
            return math.sin(values[0])
        if kind == M.COS:
            return math.cos(values[0])
        if kind == M.LOG:
            return math.log(values[0])
        if kind == M.SCALE:
            return values[0] / math.sqrt(128)
        if kind == M.CAST:
            return values[0]
        raise AssertionError(f"unsupported native oracle math {kind}")

    return tensor(
        axes,
        sizes,
        lambda index: (
            round_dtype(evaluate(index), step.math.dtype)
            if precision
            else evaluate(index)
        ),
    )


def evaluate_partition(native, part, query, keys, values, supplied=None):
    state = {}
    initializers = {s.producer: s.initializer for s in native.domain.states}
    prologue = {p.source.role for p in native.ops if p.source.scope == Scope.PROLOGUE}
    rounds = (
        part.end - part.start + native.profile.kv_tile - 1
    ) // native.profile.kv_tile

    def resolve(value, iteration):
        target = iteration + value.iteration_delta
        if value.producer in prologue:
            return state[value.producer, -1]
        if target < 0:
            return state[initializers[value.producer], -1]
        return state[value.producer, target]

    def pattern(pattern, iteration):
        op = pattern.source
        if op.kind == OpKind.STORE:
            return
        if op.kind == OpKind.COPY:
            if supplied is not None:
                result = matrix(supplied[op.region.tensor], op.results[0].axes)
            elif op.region.tensor == "Q":
                result = matrix([query], op.results[0].axes)
            else:
                start = part.start + iteration * native.profile.kv_tile
                end = min(start + native.profile.kv_tile, part.end)
                rows = keys if op.region.tensor == "K" else values
                result = matrix(rows[start:end], op.results[0].axes)
        else:
            temporary = []
            for step in pattern.steps:
                inputs = [resolve(value, iteration) for value in step.operands]
                inputs.extend(temporary[i] for i in step.input_steps)
                temporary.append(step_math(step, inputs, op.results[0].shape))
            result = temporary[-1]
        state[op.role, iteration] = result

    for p in native.ops:
        if p.source.scope == Scope.PROLOGUE:
            pattern(p, -1)
    for k in range(rounds):
        for p in native.ops:
            if p.source.scope == Scope.PIPELINE:
                pattern(p, k)
    for p in native.ops:
        if p.source.scope == Scope.EPILOGUE:
            pattern(p, rounds - 1)
    from gtsim_frontend.ir.schedule import OpRole as R

    out = state[R.NORMALIZE, rounds - 1][2]
    maximum = next(iter(state[R.MERGE_MAX, rounds - 1][2].values()))
    normalizer = next(iter(state[R.LOGSUM, rounds - 1][2].values()))
    lse = maximum + math.log(normalizer)
    if (R.PARTIAL_LSE, rounds - 1) in state:
        assert (
            abs(next(iter(state[R.PARTIAL_LSE, rounds - 1][2].values())) - lse) < 1e-12
        )
    return [out[0, d] for d in range(128)], lse


def evaluate_vector(native, supplied, precision=False):
    """Interpret native primitive instructions with real tensor/register views."""
    outputs = {
        t.id: [list(row) for row in supplied[t.id]]
        if t.id in native.domain.partial_outputs and t.id in supplied
        else [[None] * t.shape[1] for _ in range(t.shape[0])]
        for t in native.tensors
        if not t.read_only
    }
    for group in native.domain.workgroups:
        state = {}

        def resolve(value):
            stored = state[value.producer]
            if len(stored) == 2:
                stored = stored[value.result]
            axes, shape, data = stored
            if value.shape:
                return tensor(
                    axes,
                    value.shape,
                    lambda i: data[tuple(o + x for o, x in zip(value.offsets, i))],
                )
            return axes, shape, data

        for pattern in native.ops:
            op = pattern.source
            if op.kind == OpKind.COPY:
                row, col = op.region.offsets_at(dict(group.bindings))
                rows, cols = op.region.shape
                name = dict(group.aliases).get(op.region.tensor, op.region.tensor)
                source = supplied[name]
                state[op.role] = matrix(
                    [
                        [
                            round_dtype(v, op.results[0].dtype) if precision else v
                            for v in r[col : col + cols]
                        ]
                        for r in source[row : row + rows]
                    ],
                    op.results[0].axes,
                )
            elif op.kind == OpKind.STORE:
                value = resolve(pattern.steps[0].operands[0])
                row, col = op.region.offsets_at(dict(group.bindings))
                if op.region.row_index:
                    actual = resolve(ValueId(op.region.row_index))
                    assert int(next(iter(actual[2].values()))) == row
                name = dict(group.aliases).get(op.region.tensor, op.region.tensor)
                for i in range(op.region.shape[0]):
                    for j in range(op.region.shape[1]):
                        outputs[name][row + i][col + j] = value[2][i, j]
            else:
                temporary = []
                for step in pattern.steps:
                    inputs = [resolve(v) for v in step.operands]
                    inputs.extend(temporary[i] for i in step.input_steps)
                    temporary.append(
                        step_math(
                            step,
                            inputs,
                            op.results[0].shape,
                            dict(group.constants),
                            precision,
                        )
                    )
                state[op.role] = temporary[-1]
    return outputs
