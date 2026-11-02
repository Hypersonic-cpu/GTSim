"""Explicit schedule geometry without reference-kernel presets or latencies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduleConfig:
    tile_shape: tuple[int, ...]
    stages: int
    warps: int
    consumer_warps: int
    producer_warp: int
    warpgroup: int
    matrix_shape: tuple[int, ...]


@dataclass(frozen=True)
class ResourceFootprint:
    warps: int
    shared_bytes: int
    register_bytes: int
