"""Small typed mathematical parameters; all use canonical Algorithm/Flow/Native IR."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NormSpec:
    rows: int
    width: int
    epsilon: float = 1e-6


@dataclass(frozen=True)
class AddSpec:
    rows: int
    width: int


@dataclass(frozen=True)
class SwiGLUSpec:
    rows: int
    width: int


@dataclass(frozen=True)
class RopeSpec:
    batch: int
    heads: int
    head_dim: int
    input_width: int
    input_offset: int
    positions: tuple[int, ...]
    epsilon: float = 1e-6
    theta: float = 1000000.0


@dataclass(frozen=True)
class ExtractSpec:
    batch: int
    heads: int
    head_dim: int
    input_width: int
    input_offset: int


@dataclass(frozen=True)
class KVUpdateSpec:
    batch: int
    heads: int
    head_dim: int
    context_lengths: tuple[int, ...]


@dataclass(frozen=True)
class RouterSpec:
    batch: int
    experts: int
    top_k: int


@dataclass(frozen=True)
class HistogramSpec:
    batch: int
    experts: int
    top_k: int


@dataclass(frozen=True)
class RouteAssignment:
    token: int
    slot: int
    expert: int


@dataclass(frozen=True)
class GatherSpec:
    batch: int
    width: int
    experts: int
    top_k: int
    assignments: tuple[RouteAssignment, ...]


@dataclass(frozen=True)
class WeightedSpec:
    batch: int
    width: int
    top_k: int
    rows_by_token: tuple[tuple[int, ...], ...]
    grouped: bool


PrimitiveSpec = (
    NormSpec
    | AddSpec
    | SwiGLUSpec
    | RopeSpec
    | ExtractSpec
    | KVUpdateSpec
    | RouterSpec
    | HistogramSpec
    | GatherSpec
    | WeightedSpec
)
