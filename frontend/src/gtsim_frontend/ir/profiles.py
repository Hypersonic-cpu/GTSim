"""Reference schedule/resource geometry. Contains no simulator timing values."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceProfile:
    family: str
    bm: int = 128
    bn: int = 256
    bk: int = 64
    q_tile: int = 128
    kv_tile: int = 128
    head_dim: int = 128
    stages: int = 2
    warps: int = 12
    consumer_warps: int = 8
    producer_warp: int = 8
    warpgroup: int = 4
    mma_m: int = 64
    mma_n: int = 64
    mma_k: int = 16
    io_bytes: int = 2
    accumulator_bytes: int = 4
    base_registers_per_thread: int = 32


@dataclass(frozen=True)
class ResourceFootprint:
    warps: int
    shared_bytes: int
    register_bytes: int


GEMM = ReferenceProfile("gemm")
FA3 = ReferenceProfile("fa3")
SMALL_M = ReferenceProfile(
    "gemm",
    bm=16,
    bn=64,
    bk=16,
    warps=4,
    consumer_warps=4,
    producer_warp=0,
    warpgroup=1,
    mma_m=16,
    mma_n=8,
)


def resources(p):
    if p.family == "vector":
        return ResourceFootprint(p.warps, 16384, 32768)
    if p.family == "gemm":
        shared = p.stages * p.io_bytes * p.bk * (p.bm + p.bn) + p.io_bytes * p.bm * p.bn
        registers = p.accumulator_bytes * p.bm * p.bn
    else:
        shared = (
            p.io_bytes * p.q_tile * p.head_dim
            + p.stages * 2 * p.io_bytes * p.kv_tile * p.head_dim
        )
        registers = p.accumulator_bytes * p.q_tile * (p.head_dim + p.kv_tile)
    registers += p.warps * 32 * p.base_registers_per_thread * 4
    return ResourceFootprint(p.warps, shared, registers)


DECODE = ReferenceProfile(
    "fa3",
    q_tile=1,
    kv_tile=256,
    stages=1,
    warps=4,
    consumer_warps=1,
    producer_warp=1,
    warpgroup=1,
    mma_m=1,
    mma_n=256,
    mma_k=128,
)
