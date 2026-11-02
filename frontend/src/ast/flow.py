"""Explicit scheduled operations; no algorithm-family dispatch or presets."""

from dataclasses import dataclass
from .common import Header, Tensor, DataType
from .algorithm import MathOperation
from .profiles import ScheduleConfig
from .schedule import (
    OpKind,
    Scope,
    ExecutorRole,
    SymbolicRegion,
    Dependency,
    SlotUse,
    ScheduleDomain,
    ValueId,
    SSAValue,
    Executor,
    Buffer,
    BufferView,
)
from .mapping import Mapping


@dataclass(frozen=True)
class FlowOp:
    id: str
    kind: OpKind
    scope: Scope
    executor: ExecutorRole
    operands: tuple[ValueId, ...]
    dependencies: tuple[Dependency, ...]
    region: SymbolicRegion | None
    slot: SlotUse | None
    results: tuple[SSAValue, ...]
    math: MathOperation | None
    capabilities: tuple[str, ...]
    destination: str | None
    event: str | None
    cast: DataType | None


@dataclass(frozen=True)
class FlowIR(Header):
    schedule_id: str
    config: ScheduleConfig
    domain: ScheduleDomain
    tensors: tuple[Tensor, ...]
    mapping: Mapping
    ops: tuple[FlowOp, ...]
    symbols: tuple[tuple[str, str, str], ...]
    requirements: tuple[str, ...]
    expressions: tuple[tuple[str, str], ...]
    tile_map: tuple[tuple[str, str], ...]
    executors: tuple[Executor, ...]
    buffers: tuple[Buffer, ...]
    buffer_views: tuple[BufferView, ...]
    mapping_symbol: str | None
