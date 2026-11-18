"""Explicit schedule geometry without reference-kernel presets or latencies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduleConfig:
    tile_shape: tuple[int | str, ...]
    stages: int | str
    warps: int | str
    consumer_warps: int | str
    producer_warp: int | str
    warpgroup: int | str
    matrix_shape: tuple[int | str, ...]


@dataclass(frozen=True)
class ResourceFootprint:
    warps: int
    shared_bytes: int
    register_bytes: int
