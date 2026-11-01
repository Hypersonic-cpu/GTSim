from dataclasses import dataclass
from .common import Header, Tensor
from .algorithm import MathOperation, AlgorithmContract
from .profiles import ReferenceProfile, GEMM
from .schedule import (
    OpRole,
    OpKind,
    Scope,
    ExecutorRole,
    SymbolicRegion,
    Dependency,
    SlotUse,
    ScheduleDomain,
    ValueId,
    SSAValue,
)
from .mapping import Mapping


@dataclass(frozen=True)
class FlowOp:
    id: str
    role: OpRole | str
    kind: OpKind
    scope: Scope
    executor: ExecutorRole
    operands: tuple[ValueId, ...] = ()
    dependencies: tuple[Dependency, ...] = ()
    region: SymbolicRegion | None = None
    slot: SlotUse | None = None
    results: tuple[SSAValue, ...] = ()
    math: MathOperation | None = None
    reduction_shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class FlowKernel(Header):
    schema_version: int = 2
    family: str = "gemm"
    schedule_id: str = ""
    profile: ReferenceProfile = GEMM
    domain: ScheduleDomain | None = None
    tensors: tuple[Tensor, ...] = ()
    mapping: Mapping = Mapping()
    ops: tuple[FlowOp, ...] = ()
    contract: AlgorithmContract | None = None

    def verify(self):
        from ..verify.flow import verify_flow

        verify_flow(self)
