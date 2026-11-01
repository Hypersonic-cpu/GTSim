"""Canonical extraction and request-specific partial KV append schedules."""

from ..ir.primitives import ExtractSpec, KVUpdateSpec
from ..ir.common import Tensor, require
from ..ir.schedule import WorkGroup, LoopVar as V, IndexExpr
from ..ir.algorithm import AlgorithmContract, OperandContract, IndexDomain


def verify_spec(spec):
    require(
        spec.batch > 0 and spec.heads > 0 and spec.head_dim > 0,
        "memory",
        "invalid request/head geometry",
    )
    if isinstance(spec, ExtractSpec):
        require(
            spec.input_offset >= 0
            and spec.input_offset + spec.heads * spec.head_dim <= spec.input_width,
            "extract",
            "invalid source region",
        )
    else:
        require(
            len(spec.context_lengths) == spec.batch
            and all(n >= 0 for n in spec.context_lengths),
            "kv_append",
            "invalid request append offsets",
        )


def signature(spec):
    if isinstance(spec, ExtractSpec):
        return (
            Tensor("Input", (spec.batch, spec.input_width), "bf16", read_only=True),
            Tensor("Output", (spec.batch * spec.heads, spec.head_dim), "bf16"),
        )
    result = [
        Tensor(name, (spec.batch * spec.heads, spec.head_dim), "bf16", read_only=True)
        for name in ("Key", "Value")
    ]
    result.extend(
        Tensor(f"{name}_r{b}_h{h}", (context + 1, spec.head_dim), "bf16")
        for b, context in enumerate(spec.context_lengths)
        for h in range(spec.heads)
        for name in ("K", "V")
    )
    return tuple(result)


def plan(spec):
    from .vector import Body, index

    verify_spec(spec)
    ts = signature(spec)
    body = Body()
    z = IndexExpr()
    if isinstance(spec, ExtractSpec):
        groups = tuple(
            WorkGroup(
                bindings=(
                    (V.ROW, b * spec.heads + h),
                    (V.TOKEN, b),
                    (V.CHANNEL, spec.input_offset + h * spec.head_dim),
                )
            )
            for b in range(spec.batch)
            for h in range(spec.heads)
        )
        value = body.load(
            "load", "Input", index(V.TOKEN), index(V.CHANNEL), spec.head_dim
        )
        body.store("store", value, "Output", index(V.ROW), z, spec.head_dim)
    else:
        groups = tuple(
            WorkGroup(
                bindings=((V.ROW, b * spec.heads + h), (V.OUTPUT_ROW, context)),
                aliases=(("KCache", f"K_r{b}_h{h}"), ("VCache", f"V_r{b}_h{h}")),
            )
            for b, context in enumerate(spec.context_lengths)
            for h in range(spec.heads)
        )
        key = body.load("load_key", "Key", index(V.ROW), z, spec.head_dim)
        value = body.load("load_value", "Value", index(V.ROW), z, spec.head_dim)
        body.store("append_key", key, "KCache", index(V.OUTPUT_ROW), z, spec.head_dim)
        body.store(
            "append_value", value, "VCache", index(V.OUTPUT_ROW), z, spec.head_dim
        )
    rows = spec.batch * spec.heads
    contract = AlgorithmContract(
        (IndexDomain("ROWS", rows), IndexDomain("D", spec.head_dim)),
        tuple(
            OperandContract(t.id, ("ROW", "COL"), t.dtype, not t.read_only) for t in ts
        ),
        (),
        output_dtype="bf16",
    )
    return ts, groups, tuple(body.ops), contract, rows, spec.head_dim
