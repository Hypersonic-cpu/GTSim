"""Physical bindings for the grouped Small-M MMA baseline.

Row maps are part of the canonical GEMM contract, not a batched-GEMM algorithm.
Only nonempty groups and their explicitly selected weights are represented.
"""

from ...ir.common import Tensor


def physical_tensors(algorithm):
    rows, columns, reduction = algorithm.dimensions
    inputs = max(row for group in algorithm.groups for row in group.input_rows) + 1
    weights = tuple(
        Tensor(group.weight, (reduction, columns), algorithm.dtype, read_only=True)
        for group in algorithm.groups
    )
    return (
        Tensor("A", (inputs, reduction), algorithm.dtype, read_only=True),
        weights[0],
        Tensor("C", (rows, columns), algorithm.dtype),
        *weights[1:],
    )
