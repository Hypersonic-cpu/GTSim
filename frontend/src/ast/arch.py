"""Structured native instructions: loops, address expressions and phased events."""

from dataclasses import dataclass
from enum import Enum
from .common import Header, Tensor, DataType, StaticIf
from .profiles import ResourceFootprint
from .schedule import (
    ScheduleDomain,
    Dependency,
    TileLoop,
    SymbolicRegion,
    IndexExpr,
    Scope,
    DependencyRule,
    Phase,
)
from .graph import ExecutionResource


class NativeOpcode(str, Enum):
    TMA_LOAD = "cp.dram2sram_tma"
    MATRIX = "wgmma.m64n64k16"
    SHARED_STORE = "st.reg2sram"
    TMA_STORE = "cp.sram2dram_tma"


@dataclass(frozen=True)
class NativeStep:
    opcode: NativeOpcode
    resource: ExecutionResource
    executor: IndexExpr
    loops: tuple[TileLoop, ...]
    region: SymbolicRegion
    dtype: DataType
    leading_dim: IndexExpr | None
    dependencies: tuple[Dependency, ...]
    input_steps: tuple[int, ...]


@dataclass(frozen=True)
class NativePattern:
    id: str
    scope: Scope
    steps: tuple[NativeStep, ...]
    dependencies: tuple[Dependency, ...]


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
class ArchRegion(Header):
    target: str
    domain: ScheduleDomain
    tensors: tuple[Tensor, ...]
    resources: ResourceFootprint
    layouts: tuple[SharedLayout, ...]
    registers: RegisterLayout
    async_protocol: AsyncProtocol
    ops: tuple[NativePattern, ...]
    dependency_rules: tuple[DependencyRule, ...]
    static_ifs: tuple[StaticIf, ...]
    cta_variable: str
    sm: IndexExpr
    after: tuple[str, ...]
    origin: str | None


@dataclass(frozen=True)
class ArchIR(Header):
    tensors: tuple[Tensor, ...]
    regions: tuple[ArchRegion, ...]
    stream: str | None
    outputs: tuple[str, ...]
