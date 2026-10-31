from dataclasses import dataclass
from .arch import ArchKernel
from .program import Program
from .common import require


@dataclass(frozen=True)
class ExecKernel(ArchKernel):
    """IR-05 explicit graph; architecture legality retained for materialization."""


@dataclass(frozen=True)
class ExecProgram(Program):
    kernel_files: tuple[tuple[str,str], ...] = ()
    launch_overhead_cycles: int = 0

    def verify(self):
        super().verify()
        require(type(self.launch_overhead_cycles) is int and self.launch_overhead_cycles>=0,self.symbol,"invalid launch overhead")
        files=dict(self.kernel_files)
        require(len(files)==len(self.kernel_files) and set(files)=={k.id for k in self.launches},self.symbol,"missing or duplicate kernel graph")
        require(all(bool(f) for f in files.values()),self.symbol,"empty kernel graph path")
