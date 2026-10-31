"""Compact FA3 scheduled loop-body construction."""

from .templates import build


def instantiate(algorithm, num_sms=132):
    if algorithm.family != "fa3":
        raise ValueError("expected fa3 algorithm")
    return build(algorithm, num_sms)
