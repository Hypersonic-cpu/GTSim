"""Physical kernel I/O contracts shared by workload composition and validation."""

from ..ir.common import Tensor


def signature(algorithm):
    if algorithm.family == "vector":
        from .vector import signature as vector

        return vector(algorithm.primitive)
    if algorithm.schedule_id == "decode_simt":
        from .hopper.attention_decode import physical_tensors

        return physical_tensors(algorithm)
    if algorithm.schedule_id == "decode_combine":
        from .hopper.attention_combine import physical_tensors

        return physical_tensors(algorithm)
    if algorithm.groups:
        from .hopper.gemm_groups import physical_tensors

        return physical_tensors(algorithm)
    if algorithm.family == "gemm":
        m, n, k = algorithm.dimensions
        return (
            Tensor("A", (m, k), algorithm.dtype, read_only=True),
            Tensor("B", (k, n), algorithm.dtype, read_only=True),
            Tensor("C", (m, n), algorithm.dtype),
        )
    b, hq, hkv, q, kv = algorithm.dimensions
    return (
        Tensor("Q", (b, hq, q, 128), algorithm.dtype, read_only=True),
        Tensor("K", (b, hkv, 128, kv), algorithm.dtype, read_only=True),
        Tensor("V", (b, hkv, 128, kv), algorithm.dtype, read_only=True),
        Tensor("O", (b, hq, q, 128), algorithm.dtype),
    )
