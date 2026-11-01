from dataclasses import dataclass
from .graph import FlatKernel, KernelTensor
from .program import Command
from .common import Header, require


@dataclass(frozen=True)
class ExecKernel(FlatKernel):
    """IR-05 explicit kernel graph with no timing constants or addresses."""

    schema_version: int = 2


@dataclass(frozen=True)
class ExecutionLaunch:
    id: str
    kernel_file: str
    bindings: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ExecProgram(Header):
    streams: tuple[str, ...] = ("compute",)
    tensors: tuple[KernelTensor, ...] = ()
    launches: tuple[ExecutionLaunch, ...] = ()
    commands: tuple[Command, ...] = ()
    launch_overhead_cycles: int = 0

    def verify(self):
        super().verify()
        require(
            len(self.streams) == 1 and bool(self.streams[0]),
            self.symbol,
            "exactly one stream required",
        )
        require(
            type(self.launch_overhead_cycles) is int
            and self.launch_overhead_cycles >= 0,
            self.symbol,
            "invalid launch overhead",
        )
        tensors = {t.id: t for t in self.tensors}
        require(
            len(tensors) == len(self.tensors), self.symbol, "duplicate global tensor"
        )
        launches = {l.id: l for l in self.launches}
        require(len(launches) == len(self.launches), self.symbol, "duplicate launch")
        for l in self.launches:
            require(
                bool(l.kernel_file) and len(dict(l.bindings)) == len(l.bindings),
                l.id,
                "missing graph or duplicate binding",
            )
            require(
                all(t in tensors for _, t in l.bindings), l.id, "dangling global tensor"
            )
        events = set()
        submitted = set()
        for c in self.commands:
            if c.kind == "LaunchKernel":
                require(
                    c.target in launches and c.target not in submitted,
                    self.symbol,
                    "unknown/duplicate launch",
                )
                submitted.add(c.target)
            elif c.kind == "RecordEvent":
                require(
                    bool(c.target) and c.target not in events,
                    self.symbol,
                    "invalid event record",
                )
                events.add(c.target)
            elif c.kind == "WaitEvent":
                require(c.target in events, self.symbol, "wait before record")
            else:
                require(
                    c.kind == "StreamSynchronize" and not c.target,
                    self.symbol,
                    "unknown stream command",
                )
        require(submitted == set(launches), self.symbol, "unsubmitted invocation")
