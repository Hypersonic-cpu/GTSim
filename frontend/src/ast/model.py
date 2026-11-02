"""Minimal AST for the MLIR-like serving IR text form."""

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class Statement:
    text: str
    opcode: str


@dataclass(frozen=True, slots=True)
class Operation:
    name: str
    header: tuple[str, ...]
    close_indent: str
    body: tuple["Node", ...]


Node: TypeAlias = Statement | Operation


@dataclass(frozen=True, slots=True)
class Module:
    dialect: str
    nodes: tuple[Node, ...]
    trailing_newline: bool
