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
        return from_module(syntax.parse(text))
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise ParseError(str(error)) from error


def from_module(module):
    """Decode an already parsed/specialized syntax AST without source rewriting."""
    try:
        if any(
            isinstance(node, ast.Statement) and node.text.strip()
            for node in module.nodes
        ):
            raise ValueError("unexpected statement outside an IR operation")
        roots = [node for node in module.nodes if isinstance(node, ast.Operation)]
        if not roots:
            raise ValueError("expected an IR root operation")
        from .static import declarations

        root, static = declarations(roots[0])
        if root.name == "prog.program" and len(roots) == 1:
            return semantic.prog(root, static)
        if root.name == "flow.program" and len(roots) == 1:
            return semantic.flow_program(root, static)
        if root.name == "algo.kernel" and len(roots) == 1:
            return semantic.algo(root, static)
        if root.name == "flow.kernel" and len(roots) <= 2:
            if len(roots) == 2 and roots[1].name != "mapping.program":
                raise ValueError("expected mapping.program after flow.kernel")
            empty = ast.StaticDeclarations((), (), (), (), ())
            region = semantic.flow_region(
                root, roots[1] if len(roots) == 2 else None, empty
            )
            return ast.FlowIR(
                root.header[0].split("@")[1].split()[0],
                static,
                region.tensors,
                (region,),
                None,
                (),
            )
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
