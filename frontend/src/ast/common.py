"""Shared data types; identifiers and symbolic dimensions are user-defined."""

from dataclasses import dataclass
from enum import Enum


class DataType(str, Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    INT32 = "int32"
    INDEX = "index"


class MemorySpace(str, Enum):
    HBM = "HBM"
    HBF = "HBF"
    SHARED = "SMEM"
    REGISTER = "REGISTER"


class Layout(str, Enum):
    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"


class ConstantValue(str, Enum):
    POSITIVE_INFINITY = "inf"
    NEGATIVE_INFINITY = "negative_infinity"


@dataclass(frozen=True)
class StaticParameter:
    id: str
    dtype: DataType
    lower: str | None
    upper: str | None


@dataclass(frozen=True)
class StaticBinding:
    id: str
    expression: str


@dataclass(frozen=True)
class StaticPredicate:
    id: str
    expression: str


@dataclass(frozen=True)
class StaticDeclarations:
    parameters: tuple[StaticParameter, ...]
    bindings: tuple[StaticBinding, ...]
    predicates: tuple[StaticPredicate, ...]
    constraints: tuple[str, ...]


@dataclass(frozen=True)
class StaticIf:
    """Compiler condition applying to an iteration scope."""

    id: str
    condition: str
    iterations: tuple[str, ...]


@dataclass(frozen=True)
class Constant:
    id: str
    value: int | float | ConstantValue
    dtype: DataType


@dataclass(frozen=True)
class Tensor:
    id: str
    shape: tuple[int | str, ...]
    dtype: DataType
    tier: MemorySpace | None
    read_only: bool
    layout: Layout | None
    external: bool


@dataclass(frozen=True)
class Header:
    symbol: str
    static: StaticDeclarations
