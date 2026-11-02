"""Read/write MLIR-like assembly as independent typed frontend AST objects."""

from pathlib import Path
from . import ast
from .ast import ProgIR, AlgoIR, FlowIR, ArchIR, ExecIR
from . import syntax, semantic


class ParseError(ValueError):
    """Invalid assembly with source context."""


def _parse(text):
    try:
        if text.lstrip().startswith("{"):
            from .json_codec import loads

            return loads(text)
        module = syntax.parse(text)
        if any(
            isinstance(node, ast.Statement) and node.text.strip()
            for node in module.nodes
        ):
            raise ValueError("unexpected statement outside an IR operation")
        roots = [node for node in module.nodes if isinstance(node, ast.Operation)]
        if not roots:
            raise ValueError("expected an IR root operation")
        root = roots[0]
        if root.name == "prog.program" and len(roots) == 1:
            return semantic.prog(root)
        if root.name == "algo.kernel" and len(roots) == 1:
            return semantic.algo(root)
        if root.name == "flow.kernel" and len(roots) <= 2:
            if len(roots) == 2 and roots[1].name != "mapping.program":
                raise ValueError("expected mapping.program after flow.kernel")
            return semantic.flow(root, roots[1] if len(roots) == 2 else None)
        raise ValueError(f"unsupported IR root: {root.name}")
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise ParseError(str(error)) from error


def parse(text):
    from .types import check

    try:
        result = _parse(text)
        check(result)
        return result
    except ValueError as error:
        raise ParseError(str(error)) from error


def parse_file(path):
    path = Path(path)
    try:
        return parse(path.read_text(encoding="utf-8"))
    except (OSError, ParseError) as error:
        raise ParseError(f"{path}: {error}") from error


def serialize(ir):
    from .writer import serialize as write

    return write(ir)


def serialize_file(ir, path):
    try:
        Path(path).write_text(serialize(ir), encoding="utf-8")
    except OSError as error:
        raise ParseError(f"{path}: {error}") from error


__all__ = [
    "ProgIR",
    "AlgoIR",
    "FlowIR",
    "ArchIR",
    "ExecIR",
    "ParseError",
    "parse",
    "parse_file",
    "serialize",
    "serialize_file",
]
