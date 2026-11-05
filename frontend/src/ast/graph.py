"""Explicit execution graph data; opcode names belong to the selected target."""

from dataclasses import dataclass
from enum import Enum
from .layout import TileMajorLayout
from .common import Header, DataType, MemorySpace, ImmediateValue


class ExecutionResource(str, Enum):
    LOAD = "load"
    STORE = "store"
    MATRIX = "matrix"
    ARITHMETIC = "arithmetic"
    SPECIAL_FUNCTION = "special_function"
    SHUFFLE = "shuffle"


@dataclass(frozen=True)
class KernelTensor:
    id: int
    name: str
    shape: tuple[int | str, ...]
    strides: tuple[int | str, ...]
    dtype: DataType
    tier: MemorySpace
    preloaded: bool
    read_only: bool
    tile_major: TileMajorLayout | None


@dataclass(frozen=True)
class CTA:
    id: int
    sm: int
    warps: int
    shared_bytes: int
    register_bytes: int


@dataclass(frozen=True)
class ArchInstruction:
    id: int
    name: str
    opcode: str
    resource: ExecutionResource
    cta: int
    warp: int
    tensor: int | None
    tile: tuple[int, ...]
    dtype: DataType
    leading_dim: int | None
    after_issue: tuple[int, ...]
    after_complete: tuple[int, ...]
    static_args: tuple[tuple[str, ImmediateValue], ...]


@dataclass(frozen=True)
class FlatKernel(Header):
    target: str
    tensors: tuple[KernelTensor, ...]
    ctas: tuple[CTA, ...]
    ops: tuple[ArchInstruction, ...]
    partial_outputs: tuple[int, ...]
    outputs: tuple[int, ...]
