"""Structured Hopper program. Full CTA/warp instructions exist only in IR-05."""

from dataclasses import dataclass
from .common import Header, Tensor
from .algorithm import MathOperation, AlgorithmContract
from .flow import FlowOp
from .mapping import Mapping
from .profiles import ReferenceProfile, ResourceFootprint, GEMM
from .schedule import ScheduleDomain, Dependency, BufferRole, TileLoop, ValueId
from .graph import KernelTensor, CTA, ArchInstruction, HOPPER_OPS


@dataclass(frozen=True)
class NativeStep:
    opcode: str
    dependencies: tuple[Dependency, ...]
    fragment_shape: tuple[int, ...] = ()
    collective_warps: int = 1
    chunk_axis: int = -1
    chunk_extent: int = 0
    loops: tuple[TileLoop, ...] = ()
    operands: tuple[ValueId, ...] = ()
    math: MathOperation | None = None
    input_steps: tuple[int, ...] = ()
    fragment_axes: tuple[str, ...] = ()


@dataclass(frozen=True)
class NativePattern:
    source: FlowOp
    steps: tuple[NativeStep, ...]


@dataclass(frozen=True)
class SharedLayout:
    buffer: BufferRole
    slots: int
    layout: str
    bytes_per_slot: int


@dataclass(frozen=True)
class RegisterLayout:
    accumulator_dtype: str
    fragment_shape: tuple[int, int]
    consumer_warps: int
    base_registers_per_thread: int
    per_cta_bytes: int


@dataclass(frozen=True)
class AsyncProtocol:
    mbarrier_phases: int = 2
    commit_groups: int = 1
    wait_group: int = 0
    reuse_phase: str = "consumer_complete"


@dataclass(frozen=True)
class ArchKernel(Header):
    schema_version: int = 2
    target: str = "sm90"
    family: str = "gemm"
    sync_semantics_version: str = "pingpong-cross-wave-v2"
    schedule_id: str = ""
    profile: ReferenceProfile = GEMM
    domain: ScheduleDomain | None = None
    tensors: tuple[Tensor, ...] = ()
    mapping: Mapping = Mapping()
    resources: ResourceFootprint | None = None
    layouts: tuple[SharedLayout, ...] = ()
    registers: RegisterLayout | None = None
    async_protocol: AsyncProtocol = AsyncProtocol()
    ops: tuple[NativePattern, ...] = ()
    contract: AlgorithmContract | None = None

    def verify(self):
        from ..verify.native import verify_native

        verify_native(self)
