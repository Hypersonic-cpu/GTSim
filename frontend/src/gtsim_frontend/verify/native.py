from dataclasses import replace
from ..ir.schedule import Relation, Phase
from ..ir.common import Header, require
from ..ir.flow import FlowKernel
from ..ir.arch import AsyncProtocol
from ..ir.profiles import resources
from ..passes.hopper.patterns import select, layouts, register_layout


def verify_native(native):
    from .types import verify_types

    verify_types(native, location=native.symbol)
    Header.verify(native)
    require(native.target == "sm90", native.symbol, "unsupported native target")
    flow = FlowKernel(
        symbol=native.symbol,
        contract=native.contract,
        family=native.family,
        schedule_id=native.schedule_id,
        profile=native.profile,
        domain=native.domain,
        tensors=native.tensors,
        mapping=native.mapping,
        ops=tuple(pattern.source for pattern in native.ops),
    )
    flow.verify()
    require(
        native.resources == resources(native.profile),
        native.symbol,
        "invalid native resource allocation",
    )
    require(
        native.layouts == layouts(native.profile),
        native.symbol,
        "invalid shared/register layout",
    )
    require(
        native.registers == register_layout(native.profile),
        native.symbol,
        "invalid register fragment layout",
    )
    require(
        native.async_protocol == AsyncProtocol(),
        native.symbol,
        "unsupported barrier phase, commit/wait group or buffer reuse protocol",
    )
    expected_sync = (
        "streaming-decode-v1"
        if native.schedule_id in ("decode_simt", "decode_combine")
        else "pingpong-cross-wave-v2"
        if native.family == "gemm"
        else "ref-fa3-v1"
    )
    require(
        native.sync_semantics_version == expected_sync,
        native.symbol,
        "invalid native synchronization version",
    )
    seen = set()
    for pattern in native.ops:
        expected = select(pattern.source, native.profile)
        require(
            len(pattern.steps) == len(expected.steps),
            pattern.source.id,
            "native step count mismatch",
        )
        for step, ref in zip(pattern.steps, expected.steps):
            additions = tuple(d for d in step.dependencies if d not in ref.dependencies)
            require(
                all(
                    d.relation == Relation.EXPLICIT
                    and d.source in seen
                    and d.phase in (Phase.ISSUE, Phase.COMPLETE)
                    for d in additions
                ),
                pattern.source.id,
                "illegal explicit native dependency",
            )
            require(
                tuple(d for d in step.dependencies if d not in additions)
                == ref.dependencies,
                pattern.source.id,
                "missing required native synchronization dependency",
            )
        require(
            replace(
                pattern,
                steps=tuple(
                    replace(step, dependencies=ref.dependencies)
                    for step, ref in zip(pattern.steps, expected.steps)
                ),
            )
            == expected,
            pattern.source.id,
            "invalid native instruction family, collective, chunk shape or dependency legalization",
        )

        seen.add(pattern.source.role)
