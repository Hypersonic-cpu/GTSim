"""Explicit resource bindings; target and resource names are identifiers."""

from dataclasses import dataclass
from enum import Enum
from .common import MemorySpace
from .schedule import ExecutorRole


class Placement(str, Enum):
    ORDERED = "original"
    ROUND_ROBIN = "round_robin"


@dataclass(frozen=True)
class Mapping:
    target: str
    placement: Placement
    execute: tuple[tuple[ExecutorRole, str], ...]
    store: tuple[tuple[str, MemorySpace], ...]
    routes: tuple[tuple[MemorySpace, MemorySpace, str], ...]
