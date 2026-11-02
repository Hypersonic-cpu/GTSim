"""Shared static declaration parsing and printing; no IR-specific dispatch."""

from dataclasses import replace
import re
from . import ast as A


def declarations(root):
    from .semantic import statements, match, assignments, split_balanced

    parameters, bindings, predicates, constraints, body = [], [], [], [], []
    for node in statements(root.body):
        if not isinstance(node, str) or not node.startswith("static."):
            body.append(
                node if not isinstance(node, str) else A.Statement(node, "body")
            )
            continue
        if node.startswith("static.param"):
            parsed = re.fullmatch(
                r"static.param\s+%(\w+)\s*:\s*(\w+)(?:\s+range\[([^,]+),\s*([^]]+)\])?",
                node,
            )
            if parsed is None:
                raise ValueError(f"invalid static parameter: {node}")
            name, dtype, lower, upper = parsed.groups()
            parameters.append(A.StaticParameter(name, A.DataType(dtype), lower, upper))
        elif node.startswith("static.let"):
            values = assignments(node)
            if not values:
                raise ValueError("static.let requires a symbol binding")
            bindings.extend(
                A.StaticBinding(name, expression) for name, expression in values
            )
        elif node.startswith("static.pred"):
            name, expression = match(r"static.pred\s+%(\w+)\s*=\s*(.+)", node).groups()
            predicates.append(A.StaticPredicate(name, expression))
        elif node.startswith("static.require"):
            constraints.extend(
                split_balanced(match(r"static.require\s*<(.*)>", node)[1])
            )
        else:
            raise ValueError(f"unsupported static declaration: {node}")
    return replace(root, body=tuple(body)), A.StaticDeclarations(
        tuple(parameters), tuple(bindings), tuple(predicates), tuple(constraints)
    )


def emit(declarations):
    lines = []
    for parameter in declarations.parameters:
        line = f"static.param %{parameter.id} : {parameter.dtype.value}"
        if parameter.lower is not None:
            line += f" range[{parameter.lower}, {parameter.upper}]"
        lines.append(line)
    lines += [
        f"static.let %{value.id} = {value.expression}"
        for value in declarations.bindings
    ]
    lines += [
        f"static.pred %{value.id} = {value.expression}"
        for value in declarations.predicates
    ]
    if declarations.constraints:
        lines.append("static.require <" + ", ".join(declarations.constraints) + ">")
    return ["  " + line for line in lines]
