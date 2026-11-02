"""Structured target operations with explicit resource and synchronization data."""

from dataclasses import dataclass
from .common import Header, Tensor, DataType
from .algorithm import MathOperation
from .flow import FlowOp
from .mapping import Mapping
from .profiles import ScheduleConfig, ResourceFootprint
from .schedule import ScheduleDomain, Dependency, TileLoop, ValueId, Phase


@dataclass(frozen=True)
class NativeStep:
    opcode: str
    dependencies: tuple[Dependency, ...]
    fragment_shape: tuple[int | str, ...]
    collective_warps: int
    loops: tuple[TileLoop, ...]
    operands: tuple[ValueId, ...]
    math: MathOperation | None
    input_steps: tuple[int, ...]


@dataclass(frozen=True)
class NativePattern:
    source: FlowOp
    steps: tuple[NativeStep, ...]


@dataclass(frozen=True)
class SharedLayout:
    buffer: str
    slots: int
    layout: str
    bytes_per_slot: int


@dataclass(frozen=True)
class RegisterLayout:
    accumulator_dtype: DataType
    fragment_shape: tuple[int, ...]
    consumer_warps: int
    base_registers_per_thread: int
    per_cta_bytes: int


@dataclass(frozen=True)
class AsyncProtocol:
    mbarrier_phases: int
    commit_groups: int
    wait_group: int
    reuse_phase: Phase


@dataclass(frozen=True)
class ArchIR(Header):
    target: str
    schedule_id: str
    config: ScheduleConfig
    domain: ScheduleDomain
    tensors: tuple[Tensor, ...]
    mapping: Mapping
    resources: ResourceFootprint
    layouts: tuple[SharedLayout, ...]
    registers: RegisterLayout
    async_protocol: AsyncProtocol
    ops: tuple[NativePattern, ...]
