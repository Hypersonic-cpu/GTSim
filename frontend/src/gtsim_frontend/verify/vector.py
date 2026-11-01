"""Generic verification of canonical vector SSA/math and memory schedules."""

from ..ir.common import require
from ..ir.schedule import OpKind, Relation, Phase, Scope
from ..ir.profiles import resources


def verify_vector(flow):
    require(
        flow.schedule_id == "simt_vector"
        and flow.profile.family == "vector"
        and flow.contract is not None,
        flow.symbol,
        "invalid vector profile/contract",
    )
    require(
        flow.domain is not None
        and bool(flow.domain.workgroups)
        and flow.domain.num_sms > 0,
        flow.symbol,
        "missing vector workgroups",
    )
    require(
        flow.domain.cta_count == len(flow.domain.workgroups)
        and flow.domain.pipeline.loop.count == 1,
        flow.symbol,
        "invalid vector loop domain",
    )
    flow.mapping.verify("vector")
    require(
        resources(flow.profile).shared_bytes <= 228 * 1024,
        flow.symbol,
        "vector shared allocation exceeds target",
    )
    tensors = {t.id: t for t in flow.tensors}
    require(len(tensors) == len(flow.tensors), flow.symbol, "duplicate vector tensor")
    for tensor in flow.tensors:
        tensor.verify()
    require(
        tuple(op.math for op in flow.ops if op.math is not None)
        == flow.contract.operations,
        flow.symbol,
        "scheduled math disagrees with algorithm contract",
    )
    seen = {}
    for op in flow.ops:
        require(
            op.role not in seen
            and op.scope in (Scope.PIPELINE, Scope.EPILOGUE, Scope.PROLOGUE),
            op.id,
            "duplicate producer or scope",
        )
        for operand in op.operands:
            require(
                operand.producer in seen and operand.iteration_delta == 0,
                op.id,
                "SSA producer does not dominate consumer",
            )
            source = seen[operand.producer]
            require(
                0 <= operand.result < len(source.results), op.id, "invalid SSA result"
            )
            shape = source.results[operand.result].shape
            if operand.shape or operand.offsets:
                require(
                    len(shape) == len(operand.shape) == len(operand.offsets)
                    and all(
                        o >= 0 and n > 0 and o + n <= s
                        for o, n, s in zip(operand.offsets, operand.shape, shape)
                    ),
                    op.id,
                    "register view outside producer",
                )
        for dep in op.dependencies:
            require(
                dep.relation == Relation.EXPLICIT
                and dep.source in seen
                and dep.phase in (Phase.ISSUE, Phase.COMPLETE),
                op.id,
                "illegal vector scheduling dependency",
            )
        for result in op.results:
            require(
                len(result.axes) == len(result.shape)
                and all(n > 0 for n in result.shape)
                and result.dtype in ("fp32", "bf16", "int32"),
                op.id,
                "invalid vector SSA shape/axes/dtype",
            )
        if op.math is not None and op.math.parameter:
            for group in flow.domain.workgroups:
                require(
                    len(dict(group.constants)) == len(group.constants)
                    and op.math.parameter in dict(group.constants),
                    op.id,
                    "unbound mathematical group parameter",
                )
        if op.kind in (OpKind.COPY, OpKind.STORE):
            require(
                op.math is None and op.region is not None,
                op.id,
                "memory operation needs a typed region",
            )
            if op.region.row_index:
                require(
                    op.region.row_index in seen
                    and seen[op.region.row_index].results[0].dtype == "int32",
                    op.id,
                    "unbound indirect row index",
                )
            for group in flow.domain.workgroups:
                name = dict(group.aliases).get(op.region.tensor, op.region.tensor)
                require(name in tensors, op.id, "unbound memory operand")
                tensor = tensors[name]
                offsets = op.region.offsets_at(dict(group.bindings))
                require(
                    len(offsets) == len(tensor.shape) == len(op.region.shape) == 2
                    and all(
                        o >= 0 and n > 0 and o + n <= s
                        for o, n, s in zip(offsets, op.region.shape, tensor.shape)
                    ),
                    op.id,
                    "memory region outside logical tensor",
                )
                require(
                    op.kind != OpKind.STORE or not tensor.read_only,
                    op.id,
                    "store to read-only tensor",
                )
        else:
            require(
                op.kind == OpKind.ELEMENTWISE
                and op.math is not None
                and (
                    len(op.results) == 1
                    or (op.math.kind.value == "topk" and len(op.results) == 2)
                ),
                op.id,
                "invalid vector computation",
            )
        seen[op.role] = op
