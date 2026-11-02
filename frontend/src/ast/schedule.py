"""Workload-independent loop, SSA, memory, and synchronization structures."""

from dataclasses import dataclass
from enum import Enum
from .common import ConstantValue, DataType


class Phase(str, Enum):
    ISSUE = "issue"
    READY = "ready"
    COMPLETE = "complete"
    RELEASED = "released"


class OpKind(str, Enum):
    ACQUIRE = "acquire"
    COPY = "async_copy"
    MATRIX = "matrix"
    REDUCTION = "reduction"
    ARITHMETIC = "arithmetic"
    SPECIAL_FUNCTION = "special_function"
    STORE = "store_tile"
    AWAIT = "await"
    RELEASE = "release"


class Scope(str, Enum):
    PROLOGUE = "prologue"
    PIPELINE = "pipeline"
    EPILOGUE = "epilogue"


class ExecutorRole(str, Enum):
    PRODUCER = "producer"
    MATRIX = "matrix"
    VECTOR = "vector"


class Organization(str, Enum):
    INDEPENDENT = "independent"
    COOPERATIVE = "cooperative"


class AccessMode(str, Enum):
    READ_ONLY = "read_only"
    WRITE_ONLY = "write_only"
    READ_WRITE = "read_write"


class DependencyKind(str, Enum):
    DATA = "data"
    BUFFER_REUSE = "buffer_reuse"
    LOOP_CARRIED = "loop_carried"
    OUTPUT_ORDER = "output_order"
    ISSUE_ORDER = "issue_order"


@dataclass(frozen=True)
class IndexExpr:
    expression: str


@dataclass(frozen=True)
class TileId:
    index: IndexExpr


@dataclass(frozen=True)
class IterationId:
    index: IndexExpr


@dataclass(frozen=True)
class PipelineStage:
    iteration: IterationId
    modulus: int
    generation_counts_per_wave: tuple[int | str, ...]


@dataclass(frozen=True)
class SlotUse:
    buffer: str
    stage: PipelineStage


@dataclass(frozen=True)
class Dependency:
    source: str
    phase: Phase
    kind: DependencyKind
    label: str


@dataclass(frozen=True)
class SymbolicRegion:
    tensor: str
    offsets: tuple[IndexExpr, ...]
    shape: tuple[int | str, ...]


@dataclass(frozen=True)
class TileLoop:
    id: str
    variable: str
    start: int | str
    stop: int | str
    step: int | str


@dataclass(frozen=True)
class PipelineRegion:
    id: str
    loop: TileLoop
    depth: int
    stage: PipelineStage


@dataclass(frozen=True)
class ScheduleDomain:
    pipeline: PipelineRegion
    loops: tuple[TileLoop, ...]


@dataclass(frozen=True)
class ValueId:
    producer: str
    result: int
    iteration_delta: int
    initial: int | float | ConstantValue | None


@dataclass(frozen=True)
class SSAValue:
    id: ValueId
    shape: tuple[int | str, ...]
    dtype: DataType


@dataclass(frozen=True)
class Executor:
    id: str
    role: ExecutorRole
    organization: Organization


@dataclass(frozen=True)
class Buffer:
    id: str
    slots: int
    access: AccessMode
    bytes_per_slot: int


@dataclass(frozen=True)
class BufferView:
    id: str
    buffer: str
    offset: int
    shape: tuple[int | str, ...]
    dtype: DataType
