"""Parser and serializer for the serving frontend's MLIR-like text form."""

import re
from .ast import Module, Node, Operation, Statement


def _operation_name(header: str) -> str:
    return header.strip().removesuffix("{").split(None, 1)[0]


def _is_operation(line: str) -> bool:
    return line.lstrip().startswith(("prog.", "algo.", "flow.", "mapping."))


def _statement_opcode(line: str) -> str:
    match = re.search(r"\b(?:prog|algo|flow|mapping|map)\.[\w.]+", line)
    return match.group(0) if match else ""


def _block_header(lines: list[str], index: int):
    first = lines[index]
    if first.rstrip().endswith("{"):
        return (first,), index + 1
    if not _is_operation(first):
        return None

    header = [first]
    for cursor in range(index + 1, len(lines)):
        line = lines[cursor]
        if not line.strip() or _is_operation(line):
            return None
        header.append(line)
        if line.rstrip().endswith("{"):
            return tuple(header), cursor + 1
    return None


def parse(text: str) -> Module:
    """Parse line-oriented operations and brace-delimited regions."""
    if not text:
        return Module("", (), False)

    lines = text.split("\n")
    trailing_newline = lines[-1] == ""
    if trailing_newline:
        lines.pop()

    def read_nodes(index: int, in_region: bool = False):
        nodes: list[Node] = []
        while index < len(lines):
            line = lines[index]
            if line.strip() == "}":
                if not in_region:
                    raise ValueError(f"line {index + 1}: unexpected closing brace")
                return tuple(nodes), index + 1
            block = _block_header(lines, index)
            if block:
                header, index = block
                body, index = read_nodes(index, True)
                indent = header[0][: len(header[0]) - len(header[0].lstrip())]
                nodes.append(
                    Operation(_operation_name(header[0]), header, indent, body)
                )
            else:
                nodes.append(Statement(line, _statement_opcode(line)))
                index += 1
        if in_region:
            raise ValueError("unclosed operation region")
        return tuple(nodes), index

    nodes, _ = read_nodes(0)
    first_operation = next(
        (node for node in nodes if isinstance(node, Operation)), None
    )
    dialect = first_operation.name.split(".", 1)[0] if first_operation else ""
    return Module(dialect, nodes, trailing_newline)


def _emit(node: Node, lines: list[str]) -> None:
    if isinstance(node, Statement):
        lines.append(node.text)
        return

    lines.extend(node.header)
    for child in node.body:
        _emit(child, lines)
    lines.append(f"{node.close_indent}}}")


def serialize(module: Module) -> str:
    lines: list[str] = []
    for node in module.nodes:
        _emit(node, lines)
    text = "\n".join(lines)
    return text + ("\n" if module.trailing_newline else "")
