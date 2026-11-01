from ...ir.arch import ArchKernel
from ...ir.profiles import resources
from ...serialization.json_codec import digest
from .patterns import select, layouts, register_layout


def lower(flow):
    flow.verify()
    result = ArchKernel(
        symbol=flow.symbol,
        contract=flow.contract,
        source_hash=digest(flow),
        family=flow.family,
        sync_semantics_version="streaming-decode-v1"
        if flow.schedule_id in ("decode_simt", "decode_combine")
        else "pingpong-cross-wave-v2"
        if flow.family == "gemm"
        else "ref-fa3-v1",
        schedule_id=flow.schedule_id,
        profile=flow.profile,
        domain=flow.domain,
        tensors=flow.tensors,
        mapping=flow.mapping,
        resources=resources(flow.profile),
        layouts=layouts(flow.profile),
        registers=register_layout(flow.profile),
        ops=tuple(select(op, flow.profile) for op in flow.ops),
    )
    result.verify()
    return result
