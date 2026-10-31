from dataclasses import replace
from ..ir.common import Header, require
from ..ir.profiles import GEMM, FA3
from ..ir.schedule import OpRole, OpKind, Phase, Relation, ExecutorRole, Scope


def verify_domain(flow):
    from ..kernels.hopper.reference.templates import domain

    d = flow.domain
    require(d is not None, flow.symbol, "missing structured loop domain")
    from ..ir.algorithm import Algorithm

    Algorithm(symbol=flow.symbol, family=flow.family, dimensions=d.dimensions).verify()
    require(type(d.num_sms) is int and d.num_sms > 0, flow.symbol, "invalid SM count")
    require(
        flow.profile == (GEMM if flow.family == "gemm" else FA3),
        flow.symbol,
        "unsupported reference profile/depth/tiling",
    )
    require(
        d == domain(flow.family, d.dimensions, d.num_sms),
        flow.symbol,
        "unsupported tile loop, stage generation, tail predicate or pipeline depth",
    )


def verify_operations(flow):
    from ..kernels.hopper.reference.templates import operations

    expected = {op.role: op for op in operations(flow.family, flow.domain)}
    require(
        len({op.id for op in flow.ops}) == len(flow.ops),
        flow.symbol,
        "duplicate display identity",
    )
    require(
        len({op.role for op in flow.ops}) == len(flow.ops)
        and set(expected) == {op.role for op in flow.ops},
        flow.symbol,
        "missing/duplicate/unknown typed operation",
    )
    require(
        tuple(op.role for op in flow.ops) == tuple(expected),
        flow.symbol,
        "unsupported operation ordering in fixed reference schedule",
    )
    seen = set()
    by_role = {op.role: op for op in flow.ops}
    for op in flow.ops:
        require(
            isinstance(op.role, OpRole) and isinstance(op.kind, OpKind),
            op.id,
            "untyped operation identity",
        )
        ref = expected[op.role]
        require(
            op.dependencies == ref.dependencies,
            op.id,
            "unsupported event/dependency transformation in fixed reference schedule",
        )
        require(
            op.slot == ref.slot,
            op.id,
            "unsupported buffer slot/generation reassignment",
        )
        require(
            op.region == ref.region,
            op.id,
            "unsupported tensor region/output coverage transformation",
        )
        require(op.executor == ref.executor, op.id, "unsupported executor assignment")
        require(
            replace(op, id=ref.id) == ref,
            op.id,
            "unsupported SSA operands, scope or operation kind",
        )
        require(
            isinstance(op.executor, ExecutorRole) and isinstance(op.scope, Scope),
            op.id,
            "untyped executor or scope",
        )
        for value in op.operands:
            require(
                value.producer in by_role
                and 0 <= value.result < len(by_role[value.producer].results),
                op.id,
                "missing SSA producer/result",
            )
            require(
                value.iteration_delta < 0 or value.producer in seen,
                op.id,
                "SSA value does not dominate operation",
            )
        seen.add(op.role)
        for dependency in op.dependencies:
            require(
                isinstance(dependency.source, OpRole)
                and isinstance(dependency.phase, Phase)
                and isinstance(dependency.relation, Relation),
                op.id,
                "untyped dependency/event identity",
            )
            require(
                dependency.source in expected, op.id, "missing producer/event dominance"
            )


def verify_tensors(flow):
    from ..ir.algorithm import Algorithm
    from ..ir.program import Program

    dims = flow.domain.dimensions
    if flow.family == "gemm":
        m, n, k = dims
        shapes = {"A": (m, k), "B": (k, n), "C": (m, n)}
        outputs = {"C"}
    else:
        b, hq, hkv, q, kv = dims
        shapes = {
            "Q": (b, hq, q, 128),
            "K": (b, hkv, 128, kv),
            "V": (b, hkv, 128, kv),
            "O": (b, hq, q, 128),
        }
        outputs = {"O"}
    require(
        len(flow.tensors) == len(shapes)
        and {t.id for t in flow.tensors} == set(shapes),
        flow.symbol,
        "missing/duplicate tensor identity",
    )
    for tensor in flow.tensors:
        tensor.verify()
        require(
            tensor.shape == shapes[tensor.id]
            and tensor.dtype == "fp16"
            and tensor.read_only == (tensor.id not in outputs),
            tensor.id,
            "invalid operand region, dtype or output identity",
        )
    Algorithm(symbol=flow.symbol, family=flow.family, dimensions=dims).verify()


def verify_flow(flow):
    from .types import verify_types

    verify_types(flow, location=flow.symbol)
    Header.verify(flow)
    require(flow.family in ("gemm", "fa3"), flow.symbol, "unknown algorithm family")
    require(
        flow.schedule_id == f"ref_h100_{flow.family}_v0_1_0",
        flow.symbol,
        "unknown schedule",
    )
    verify_domain(flow)
    verify_tensors(flow)
    flow.mapping.verify(flow.family)
    verify_operations(flow)
