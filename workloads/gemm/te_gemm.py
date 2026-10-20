"""Computation-only FP16 GEMM with FP32 reduction and FP16 output."""

from tvm import te


def workload(m=512, n=512, k=512):
    a = te.placeholder((m, k), name="A", dtype="float16")
    b = te.placeholder((k, n), name="B", dtype="float16")
    rk = te.reduce_axis((0, k), name="k")
    accum = te.compute((m, n), lambda i, j: te.sum(
        a[i, rk].astype("float32") * b[rk, j].astype("float32"), axis=rk), name="matmul")
    c = te.compute((m, n), lambda i, j: accum[i, j].astype("float16"), name="C")
    return a, b, c
