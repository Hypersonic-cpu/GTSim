"""Instantiate an explicit Flow schedule for a typed algorithm invocation."""

from dataclasses import replace
from pathlib import Path
from parser import from_module
from .bind import specialize, bind_tensors
from .static import specialize_source


def compile_algorithm(
    algorithm_path, schedule_path, parameters, tensor_bindings, tensors
):
    algorithm_source, algorithm_resolver = specialize_source(
        Path(algorithm_path), parameters
    )
    algorithm = specialize(from_module(algorithm_source), algorithm_resolver)
    source, resolver = specialize_source(Path(schedule_path), parameters)
    flow = specialize(from_module(source), resolver)
    (region,) = flow.regions
    formal_tensors = {t.id: t for t in algorithm.tensors}
    for formal, actual in tensor_bindings.items():
        expected, supplied = formal_tensors[formal], tensors[actual]
        if expected.shape != supplied.shape or expected.dtype != supplied.dtype:
            raise ValueError(f"tensor binding mismatch: {formal} -> {actual}")
    scheduled_tensors = {t.id: t for t in region.tensors}
    for name in tensor_bindings:
        if (
            scheduled_tensors[name].shape != formal_tensors[name].shape
            or scheduled_tensors[name].dtype != formal_tensors[name].dtype
        ):
            raise ValueError(f"schedule tensor differs from algorithm: {name}")
    mathematics = {op.id: op for op in algorithm.operations}
    region = replace(
        region,
        static=flow.static,
        origin=algorithm.symbol,
        ops=tuple(
            replace(op, math=mathematics[op.implements]) if op.implements else op
            for op in region.ops
        ),
    )
    represented = {op.implements for op in region.ops if op.implements}
    if represented != set(mathematics):
        raise ValueError("schedule must explicitly implement every algorithm operation")
    return bind_tensors(region, tensor_bindings)
