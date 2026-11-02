"""Shared, restricted static expression resolution and source-AST specialization."""

import ast
from dataclasses import replace
import operator
import re
from pathlib import Path
from parser import ast as A, syntax
from parser.static import declarations

EMPTY = A.StaticDeclarations((), (), (), (), ())
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_FUNCTIONS = {
    "ceildiv": lambda a, b: (a + b - 1) // b,
    "floordiv": operator.floordiv,
    "min": min,
    "max": max,
}


def render(node):
    if isinstance(node, ast.Constant):
        if type(node.value) is bool:
            return "true" if node.value else "false"
        return repr(node.value) if isinstance(node.value, str) else str(node.value)
    if isinstance(node, ast.Name):
        return "%" + node.id
    if isinstance(node, ast.BinOp):
        spelling = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.FloorDiv: "//",
            ast.Mod: "mod",
        }
        return f"({render(node.left)} {spelling[type(node.op)]} {render(node.right)})"
    if isinstance(node, ast.BoolOp):
        return (
            "("
            + (" and " if isinstance(node.op, ast.And) else " or ").join(
                render(v) for v in node.values
            )
            + ")"
        )
    if isinstance(node, ast.UnaryOp):
        return (
            "not " if isinstance(node.op, ast.Not) else "-"
        ) + f"({render(node.operand)})"
    if isinstance(node, ast.Compare):
        spelling = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }
        return (
            "("
            + render(node.left)
            + " "
            + " ".join(
                spelling[type(op)] + " " + render(value)
                for op, value in zip(node.ops, node.comparators)
            )
            + ")"
        )
    if isinstance(node, ast.Call):
        return node.func.id + "(" + ", ".join(render(v) for v in node.args) + ")"
    raise ValueError(f"unsupported static expression: {ast.dump(node)}")


class Resolver:
    def __init__(self, declarations, parameters, functions=None):
        self.functions = {**_FUNCTIONS, **(functions or {})}
        self.context = declarations
        self.values = dict(parameters)
        self.definitions = {
            v.id: v.expression
            for v in (*declarations.bindings, *declarations.predicates)
        }
        for parameter in declarations.parameters:
            value = self.values[parameter.id]
            if (
                parameter.dtype in (A.DataType.INDEX, A.DataType.INT32)
                and type(value) is not int
            ):
                raise ValueError(f"{parameter.id}: expected integer")
            if parameter.lower is not None and value < self.value(parameter.lower):
                raise ValueError(f"{parameter.id}: below declared range")
            if parameter.upper not in (None, "*") and value > self.value(
                parameter.upper
            ):
                raise ValueError(f"{parameter.id}: above declared range")
        for constraint in declarations.constraints:
            if self.value(constraint) is not True:
                raise ValueError(f"unsatisfied static constraint: {constraint}")

    def expression(self, text):
        text = re.sub(r"%([A-Za-z_]\w*)", r"\1", str(text))
        text = re.sub(r"\bmod\b", "%", text)
        text = re.sub(r"\btrue\b", "True", text)
        text = re.sub(r"\bfalse\b", "False", text)
        if " divisible_by " in text:
            left, right = text.split(" divisible_by ", 1)
            text = f"({left}) % ({right}) == 0"
        return self.fold(ast.parse(text, mode="eval").body)

    def value(self, text):
        node = self.expression(text)
        if not isinstance(node, ast.Constant):
            raise ValueError(f"unresolved static expression: {text}")
        return node.value

    def scalar(self, text):
        node = self.expression(text)
        return node.value if isinstance(node, ast.Constant) else render(node)

    def fold(self, node):
        if isinstance(node, ast.Constant):
            return node
        if isinstance(node, ast.Name):
            if node.id in self.values:
                return ast.Constant(self.values[node.id])
            if node.id in self.definitions:
                resolved = self.expression(self.definitions[node.id])
                return resolved if isinstance(resolved, ast.Constant) else node
            return node
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = self.fold(node.left), self.fold(node.right)
            if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
                return ast.Constant(_BINARY[type(node.op)](left.value, right.value))
            return ast.BinOp(left, node.op, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub)):
            value = self.fold(node.operand)
            if isinstance(value, ast.Constant):
                return ast.Constant(
                    not value.value if isinstance(node.op, ast.Not) else -value.value
                )
            return ast.UnaryOp(node.op, value)
        if isinstance(node, ast.BoolOp):
            values = []
            conjunction = isinstance(node.op, ast.And)
            for value in node.values:
                value = self.fold(value)
                if isinstance(value, ast.Constant):
                    if bool(value.value) != conjunction:
                        return ast.Constant(not conjunction)
                else:
                    values.append(value)
            if not values:
                return ast.Constant(conjunction)
            return values[0] if len(values) == 1 else ast.BoolOp(node.op, values)
        if isinstance(node, ast.Compare) and all(
            type(op) in _COMPARE for op in node.ops
        ):
            values = [self.fold(node.left), *(self.fold(v) for v in node.comparators)]
            if all(isinstance(v, ast.Constant) for v in values):
                return ast.Constant(
                    all(
                        _COMPARE[type(op)](left.value, right.value)
                        for op, left, right in zip(node.ops, values, values[1:])
                    )
                )
            return ast.Compare(values[0], node.ops, values[1:])
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and not node.keywords
        ):
            if node.func.id not in (*self.functions, "has_previous_fragment"):
                raise ValueError(f"unsupported static function: {node.func.id}")
            values = [self.fold(v) for v in node.args]
            if node.func.id in self.functions and all(
                isinstance(v, ast.Constant) for v in values
            ):
                return ast.Constant(
                    self.functions[node.func.id](*(v.value for v in values))
                )
            return ast.Call(node.func, values, [])
        raise ValueError(f"unsupported static expression: {ast.dump(node)}")

    def residual(self):
        bindings, predicates = [], []
        for definitions, destination, cls in (
            (self.context.bindings, bindings, A.StaticBinding),
            (self.context.predicates, predicates, A.StaticPredicate),
        ):
            for definition in definitions:
                value = self.expression(definition.expression)
                if not isinstance(value, ast.Constant):
                    destination.append(cls(definition.id, render(value)))
        return A.StaticDeclarations((), tuple(bindings), tuple(predicates), (), ())


def select(nodes, resolver):
    """Select parameter-resolved branches; retain loop-instance-dependent ones."""
    output = []
    active_chain = False
    unresolved = False
    selected = False
    for node in nodes:
        if isinstance(node, A.StaticBranch):
            if node.kind == A.StaticBranchKind.IF:
                active_chain, unresolved, selected = True, False, False
            if not active_chain or selected:
                continue
            condition = (
                resolver.expression(node.condition)
                if node.condition is not None
                else ast.Constant(True)
            )
            if isinstance(condition, ast.Constant) and not condition.value:
                continue
            body = select(node.body, resolver)
            if isinstance(condition, ast.Constant):
                if unresolved:
                    output.append(
                        A.StaticBranch(A.StaticBranchKind.ELSE, None, (), body)
                    )
                else:
                    output.extend(body)
                selected = True
            else:
                kind = A.StaticBranchKind.ELIF if unresolved else A.StaticBranchKind.IF
                output.append(
                    A.StaticBranch(kind, render(condition), node.iterations, body)
                )
                unresolved = True
        elif isinstance(node, A.Operation):
            output.append(replace(node, body=select(node.body, resolver)))
            active_chain = False
        else:
            output.append(node)
            if node.text.strip():
                active_chain = False
    return tuple(output)


def specialize_source(path, parameters):
    """Return a specialized syntax Module and its common resolver."""
    module = syntax.parse(path.read_text())
    root = next(v for v in module.nodes if isinstance(v, A.Operation))
    _, context = declarations(root)
    resolver = Resolver(context, parameters)
    return replace(
        module,
        nodes=tuple(
            replace(v, body=select(v.body, resolver))
            if isinstance(v, A.Operation)
            else v
            for v in module.nodes
        ),
    ), resolver


def source_symbol(path):
    """Read a definition name without prematurely decoding unselected branches."""
    root = next(
        v
        for v in syntax.parse(Path(path).read_text()).nodes
        if isinstance(v, A.Operation)
    )
    return re.search(r"@(\w+)", " ".join(root.header))[1]


def bound_resolver(context, parent):
    """Instantiate a lexical static scope at the final lowering boundary."""
    arguments = {arg.id: parent.value(arg.expression) for arg in context.arguments}
    return Resolver(context, {**parent.values, **arguments})
