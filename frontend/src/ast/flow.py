"""Explicit scheduled operations; no algorithm-family dispatch or presets."""

from dataclasses import dataclass
from .common import Header, Tensor, DataType, StaticIf, StaticBinding
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
    DependencyRule,
    TileLoop,
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
    implements: str | None
    iterations: tuple[TileLoop, ...]
    work_shape: tuple[int | str, ...]
    guard: str
    static_args: tuple[StaticBinding, ...]


@dataclass(frozen=True)
class FlowRegion(Header):
    schedule_id: str
    config: ScheduleConfig
    domain: ScheduleDomain
    tensors: tuple[Tensor, ...]
    mapping: Mapping
    ops: tuple[FlowOp, ...]
    executors: tuple[Executor, ...]
    buffers: tuple[Buffer, ...]
    buffer_views: tuple[BufferView, ...]
    mapping_symbol: str | None
    dependency_rules: tuple[DependencyRule, ...]
    static_ifs: tuple[StaticIf, ...]

    after: tuple[str, ...]
    origin: str | None


@dataclass(frozen=True)
class FlowIR(Header):
    """Ordered scheduled code; regions provide local resource/symbol scopes."""

    tensors: tuple[Tensor, ...]
    regions: tuple[FlowRegion, ...]
    stream: str | None
    outputs: tuple[str, ...]
