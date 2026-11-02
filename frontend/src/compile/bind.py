"""Resolve typed dimensions/indices and bind logical tensor identities."""

from dataclasses import fields, is_dataclass, replace
from parser import ast as A
from .static import EMPTY


def specialize(value, resolver):
    if isinstance(value, tuple):
        return tuple(specialize(v, resolver) for v in value)
    if isinstance(value, A.StaticDeclarations):
        return resolver.residual() if value != EMPTY else EMPTY
    if not is_dataclass(value):
        return value
    updates = {
        f.name: specialize(getattr(value, f.name), resolver) for f in fields(value)
    }
    if isinstance(value, (A.Tensor, A.SSAValue)):
        updates["shape"] = tuple(resolver.scalar(d) for d in value.shape)
    elif isinstance(value, A.IndexExpr):
        updates["expression"] = str(resolver.scalar(value.expression))
    elif isinstance(value, A.TileLoop):
        updates.update(
            (key, resolver.scalar(getattr(value, key)))
            for key in ("start", "stop", "step")
        )
    elif isinstance(value, A.PipelineStage):
        updates["generation_counts_per_wave"] = tuple(
            resolver.scalar(v) for v in value.generation_counts_per_wave
        )
    elif isinstance(value, A.DependencyRule):
        condition = resolver.expression(value.static_if)
        from .static import render

        updates["static_if"] = render(condition)
        updates["producer_bindings"] = tuple(
            (key, str(resolver.scalar(expr))) for key, expr in value.producer_bindings
        )
    elif isinstance(value, A.StaticIf):
        from .static import render

        updates["condition"] = render(resolver.expression(value.condition))
        updates["id"] = updates["condition"].lstrip("%")
    elif isinstance(value, A.FlowRegion):
        updates["dependency_rules"] = tuple(
            rule for rule in updates["dependency_rules"] if rule.static_if != "false"
        )
    return replace(value, **updates)


def bind_tensors(value, bindings):
    if isinstance(value, tuple):
        return tuple(bind_tensors(v, bindings) for v in value)
    if not is_dataclass(value):
        return value
    updates = {
        f.name: bind_tensors(getattr(value, f.name), bindings) for f in fields(value)
    }
    if isinstance(value, A.Tensor):
        updates["id"] = bindings.get(value.id, value.id)
    elif isinstance(value, A.SymbolicRegion):
        updates["tensor"] = bindings.get(value.tensor, value.tensor)
    elif isinstance(value, A.MathOperation):
        updates["inputs"] = tuple(bindings.get(v, v) for v in value.inputs)
        updates["output"] = bindings.get(value.output, value.output)
    return replace(value, **updates)
