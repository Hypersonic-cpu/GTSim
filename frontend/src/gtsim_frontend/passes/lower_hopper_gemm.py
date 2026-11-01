"""Select Hopper patterns without expanding the GEMM loop domain."""

from .hopper.lower import lower as lower_patterns


def lower(flow):
    from ..ir.decode import DecodeFlowKernel, KernelFamily

    if isinstance(flow, DecodeFlowKernel):
        if flow.algorithm.family in (KernelFamily.FA3_DECODE, KernelFamily.FA3_COMBINE):
            raise ValueError("FA3 decode Flow must use the FA3 Hopper lowerer")
        from .decode_lowering import lower_native

        return lower_native(flow)
    if flow.family != "gemm":
        raise ValueError("expected gemm Flow")
    return lower_patterns(flow)
