"""The only full loop/warp-fragment expansion pass."""

from .hopper.expand_gemm import expand as expand_gemm
from .hopper.expand_fa3 import expand as expand_fa3


def lower(arch, num_sms=132):
    arch.verify()
    if arch.family == "vector":
        from .hopper.expand_structured import expand

        return expand(arch)
    if arch.family == "gemm":
        return expand_gemm(arch)
    if arch.family == "fa3":
        return expand_fa3(arch)
    raise ValueError("unsupported native family")
