"""Verify the Tq=1 specialization within canonical structured Flow IR."""

from dataclasses import replace
from ..ir.algorithm import Algorithm
from ..ir.common import require
from ..ir.profiles import DECODE
from ..ir.schedule import Relation, Phase


def verify_attention_flow(flow):
    from ..kernels.hopper.attention_decode import instantiate

    if flow.schedule_id == "decode_combine":
        from ..kernels.hopper.attention_combine import instantiate

    d = flow.domain
    require(
        flow.family == "fa3"
        and d is not None
        and (flow.schedule_id == "decode_combine" or flow.profile == DECODE),
        flow.symbol,
        "invalid decode attention profile",
    )
    require(
        d.num_sms > 0 and bool(d.partitions),
        flow.symbol,
        "missing attention partitions",
    )
    batch = d.dimensions[0]
    lengths = tuple(
        max((part.end for part in d.partitions if part.request == b), default=0)
        for b in range(batch)
    )
    splits = (
        flow.profile.kv_tile
        if flow.schedule_id == "decode_combine"
        else max(part.split for part in d.partitions) + 1
    )
    algorithm = Algorithm(
        symbol=flow.symbol,
        family="fa3",
        dimensions=d.dimensions,
        dtype=flow.tensors[-1].dtype
        if flow.schedule_id == "decode_combine"
        else flow.tensors[0].dtype,
        schedule_id=flow.schedule_id,
        valid_lengths=lengths,
        split_count=splits,
    )
    expected = instantiate(algorithm, d.num_sms)
    require(
        d == expected.domain, flow.symbol, "invalid GQA, split coverage, or loop state"
    )
    require(
        flow.tensors == expected.tensors and flow.mapping == expected.mapping,
        flow.symbol,
        "invalid decode tensor regions or resource mapping",
    )
    require(
        len(flow.ops) == len(expected.ops),
        flow.symbol,
        "missing decode scheduled operations",
    )
    seen = set()
    for op, ref in zip(flow.ops, expected.ops):
        require(
            replace(op, id=ref.id, dependencies=ref.dependencies) == ref,
            op.id,
            "invalid attention math, region, SSA values, or scheduling operation",
        )
        require(
            tuple(dep for dep in op.dependencies if dep.relation != Relation.EXPLICIT)
            == ref.dependencies,
            op.id,
            "missing required attention memory synchronization",
        )
        for dep in op.dependencies:
            if dep.relation == Relation.EXPLICIT:
                require(
                    dep.source in seen and dep.phase in (Phase.ISSUE, Phase.COMPLETE),
                    op.id,
                    "explicit attention dependency must have an earlier producer",
                )
        seen.add(op.role)
