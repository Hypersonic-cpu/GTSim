"""Select Hopper patterns without expanding the GEMM loop domain."""

from .hopper.lower import lower as lower_patterns


def lower(flow):
    if flow.family != "gemm":
        raise ValueError("expected gemm Flow")
    return lower_patterns(flow)
