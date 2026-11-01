"""The only full loop/warp-fragment expansion pass."""

from .hopper.expand_gemm import expand as expand_gemm
from .hopper.expand_fa3 import expand as expand_fa3


def lower(arch, num_sms=132):
    from ..ir.decode import DecodeNativeKernel

    if isinstance(arch, DecodeNativeKernel):
        from .decode_expand import expand as expand_decode

        return expand_decode(arch, num_sms)
    arch.verify()
    if arch.family == "gemm":
        return expand_gemm(arch)
    if arch.family == "fa3":
        return expand_fa3(arch)
    raise ValueError("unsupported native family")
