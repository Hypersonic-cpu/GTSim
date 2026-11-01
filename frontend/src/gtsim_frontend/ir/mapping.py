from dataclasses import dataclass
from .common import require
from .schedule import ExecutorRole, BufferRole


@dataclass(frozen=True)
class Mapping:
    target: str = "sm90"
    placement: str = "original"
    execute: tuple[tuple[ExecutorRole, str], ...] = ()
    store: tuple[tuple[BufferRole, str], ...] = ()
    routes: tuple[tuple[str, str, str], ...] = (("HBM", "SMEM", "bulk_copy"),)

    def verify(self, family, matrix="WGMMA"):
        require(
            self.target == "sm90" and self.placement == "original",
            "mapping",
            "unsupported target or CTA placement",
        )
        if family == "vector":
            require(
                dict(self.execute)
                == {ExecutorRole.PRODUCER: "TMA", ExecutorRole.VECTOR: "SIMD"}
                and self.store == ((BufferRole.OUTPUT, "SMEM"),)
                and self.routes == (("HBM", "SMEM", "bulk_copy"),),
                "mapping",
                "invalid vector execution/storage mapping",
            )
            return
        expected = {ExecutorRole.PRODUCER: "TMA", ExecutorRole.MATRIX: matrix}
        if family == "fa3":
            expected[ExecutorRole.VECTOR] = "SIMD"
        require(
            len(dict(self.execute)) == len(self.execute)
            and dict(self.execute) == expected,
            "mapping",
            "missing/invalid executor capability binding",
        )
        buffers = (
            {BufferRole.AB, BufferRole.OUTPUT}
            if family == "gemm"
            else {BufferRole.Q, BufferRole.K, BufferRole.V, BufferRole.OUTPUT}
        )
        require(
            len(dict(self.store)) == len(self.store)
            and set(dict(self.store)) == buffers
            and all(v == "SMEM" for _, v in self.store),
            "mapping",
            "missing/invalid buffer storage binding",
        )
        require(
            self.routes == (("HBM", "SMEM", "bulk_copy"),),
            "mapping",
            "unsupported memory route",
        )
