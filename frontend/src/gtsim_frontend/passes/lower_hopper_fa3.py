"""Select Hopper patterns without expanding the FA3 loop domain."""

from .hopper.lower import lower as lower_patterns


def lower(flow):
    from ..ir.decode import DecodeFlowKernel, KernelFamily

    if isinstance(flow, DecodeFlowKernel):
        if flow.algorithm.family not in (
            KernelFamily.FA3_DECODE,
            KernelFamily.FA3_COMBINE,
        ):
            raise ValueError("non-FA3 decode Flow must use the GEMM Hopper lowerer")
        from .decode_lowering import lower_native

        return lower_native(flow)
    if flow.family != "fa3":
        raise ValueError("expected fa3 Flow")
    return lower_patterns(flow)
