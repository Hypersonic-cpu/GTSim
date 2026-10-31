"""Compact GEMM scheduled loop-body construction."""

from .templates import build


def instantiate(algorithm, num_sms=132):
    if algorithm.family != "gemm":
        raise ValueError("expected gemm algorithm")
    return build(algorithm, num_sms)
