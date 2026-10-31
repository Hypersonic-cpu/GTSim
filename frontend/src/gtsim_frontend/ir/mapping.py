from dataclasses import dataclass
from .common import require


@dataclass(frozen=True)
class Executor:
    id: str
    role: str
    warps: tuple[int, ...]


@dataclass(frozen=True)
class Buffer:
    id: str
    slots: int
    bytes_per_slot: int


@dataclass(frozen=True)
class Mapping:
    target: str = "sm90"
    execute: tuple[tuple[str,str], ...] = ()
    store: tuple[tuple[str,str], ...] = ()
    routes: tuple[tuple[str,str,str], ...] = (("HBM","SMEM","bulk_copy"),)

    def verify(self, executors, buffers):
        require(self.target == "sm90", "mapping", "only Hopper reference target is supported")
        ex, st = dict(self.execute), dict(self.store)
        require(len(ex) == len(self.execute) and set(ex) == {e.id for e in executors}, "mapping", "missing or duplicate executor binding")
        require(len(st) == len(self.store) and set(st) == {b.id for b in buffers}, "mapping", "missing or duplicate buffer binding")
        capabilities = {"producer":"TMA", "matrix":"WGMMA", "vector":"SIMD"}
        for e in executors:
            require(e.role in capabilities and ex[e.id] == capabilities[e.role], e.id, "unsupported executor capability")
            require(bool(e.warps) and len(set(e.warps)) == len(e.warps) and all(0 <= w < 12 for w in e.warps), e.id, "invalid warp binding")
        for b in buffers:
            require(st[b.id] == "SMEM" and b.slots > 0 and b.bytes_per_slot > 0, b.id, "invalid staging allocation")
        require(self.routes == (("HBM","SMEM","bulk_copy"),), "mapping", "unsupported memory route")
