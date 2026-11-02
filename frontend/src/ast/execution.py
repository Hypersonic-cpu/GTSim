"""Executable kernel and program data without compatibility metadata."""

from dataclasses import dataclass
from .graph import FlatKernel, KernelTensor
from .program import Command
from .common import Header


@dataclass(frozen=True)
class ExecRegion:
    id: str
    nodes: tuple[int, ...]
    ctas: tuple[int, ...]
    after: tuple[str, ...]
    outputs: tuple[int, ...]


@dataclass(frozen=True)
class ExecIR(FlatKernel):
    regions: tuple[ExecRegion, ...]
    stream: str | None


@dataclass(frozen=True)
class ExecutionLaunch:
    id: str
    kernel_file: str
    bindings: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ExecProgram(Header):
    streams: tuple[str, ...]
    tensors: tuple[KernelTensor, ...]
    launches: tuple[ExecutionLaunch, ...]
    commands: tuple[Command, ...]
    launch_overhead_cycles: int
