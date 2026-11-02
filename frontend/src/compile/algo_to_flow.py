"""Compose algorithm semantics and a symbolic schedule; defer static resolution."""

from dataclasses import replace
from parser import parse_file
from .bind import bind_tensors
from parser import ast as A


def compile_algorithm(
    algorithm_path, schedule_path, parameters, tensor_bindings, tensors
):
    algorithm = parse_file(algorithm_path)
    flow = parse_file(schedule_path)
    (region,) = flow.regions
    mathematics = {op.id: op for op in algorithm.operations}
    represented = {op.implements for op in region.ops if op.implements}
    if represented != set(mathematics):
        raise ValueError("schedule must explicitly implement every algorithm operation")
    arguments = tuple(
        A.StaticBinding(name, str(value)) for name, value in parameters.items()
    )
    region = replace(
        region,
        static=replace(flow.static, arguments=arguments),
        origin=algorithm.symbol,
        ops=tuple(
            replace(op, math=mathematics[op.implements]) if op.implements else op
            for op in region.ops
        ),
    )
    return bind_tensors(region, tensor_bindings)
