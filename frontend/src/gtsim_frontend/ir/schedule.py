"""Compact typed loop, identity, region, and synchronization vocabulary."""

from dataclasses import dataclass
from enum import Enum
from .common import MatrixGroup


class TailPredicate(str, Enum):
    FULL_TILE = "full_tile"
    TILE_EXISTS = "tile_id < tile_count"


class LoopVar(str, Enum):
    CTA = "cta"
    WAVE = "wave"
    TILE = "tile"
    ROW = "row"
    COLUMN = "column"
    K = "k"
    BATCH = "batch"
    HEAD = "head"
    KV_HEAD = "kv_head"
    WARP = "warp"
    M_HALF = "m_half"
    N_CHUNK = "n_chunk"
    K_FRAGMENT = "k_fragment"
    SPLIT = "split"
    KV_START = "kv_start"
    TOKEN = "token"
    CHANNEL = "channel"
    SLOT = "slot"
    OUTPUT_ROW = "output_row"


class Phase(str, Enum):
    ISSUE = "issue"
    READY = "ready"
    COMPLETE = "complete"
    RELEASED = "released"


class OpKind(str, Enum):
    ACQUIRE = "acquire"
    COPY = "async_copy"
    MATMUL = "matmul_tile"
    REDUCE_MAX = "reduce_max"
    REDUCE_SUM = "reduce_sum"
    ELEMENTWISE = "elementwise"
    STORE = "store_tile"
    AWAIT = "await"
    RELEASE = "release"


class Scope(str, Enum):
    PROLOGUE = "prologue"
    PIPELINE = "pipeline"
    EPILOGUE = "epilogue"


class ExecutorRole(str, Enum):
    PRODUCER = "producer"
    MATRIX = "matrix"
    VECTOR = "vector"


class BufferRole(str, Enum):
    AB = "AB"
    Q = "Q"
    K = "K"
    V = "V"
    OUTPUT = "output"


class OpRole(str, Enum):
    INIT_MAX = "init_max"
    INIT_SUM = "init_sum"
    INIT_OUTPUT = "init_output"
    OUTPUT_CAST = "output_cast"
    PARTIAL_LSE = "partial_lse"
    STORE_LSE = "store_lse"
    ACQUIRE = "acquire"
    RELEASE = "release"
    LOAD_A = "load_A"
    LOAD_B = "load_B"
    MATMUL = "matmul"
    STORE = "store"
    LOAD_Q = "load_Q"
    LOAD_K = "load_K"
    LOAD_V = "load_V"
    QK = "QK"
    PV = "PV"
    NORMALIZE = "normalize"
    MAX0 = "max0"
    SHFL_MAX0 = "shfl_max0"
    REDUCE_MAX0 = "reduce_max0"
    MAX1 = "max1"
    SHFL_MAX1 = "shfl_max1"
    REDUCE_MAX1 = "reduce_max1"
    MERGE_MAX = "merge_max"
    SCALE = "scale"
    EXP0 = "exp0"
    EXP1 = "exp1"
    SUM0 = "sum0"
    SHFL_SUM0 = "shfl_sum0"
    REDUCE_SUM0 = "reduce_sum0"
    SUM1 = "sum1"
    SHFL_SUM1 = "shfl_sum1"
    REDUCE_SUM1 = "reduce_sum1"
    MERGE_SUM = "merge_sum"
    LOGSUM = "logsum"
    SCALE_O = "scale_O"
    CAST = "cast"


class Relation(str, Enum):
    EXPLICIT = "explicit_preceding_producer"
    BUFFER_REUSE = "buffer_reuse"
    PREVIOUS_LOAD = "previous_load"
    LOAD_A = "load_A"
    LOAD_B = "load_B"
    PREVIOUS_FRAGMENT = "previous_fragment"
    PREVIOUS_TILE_STORE = "previous_tile_store"
    WARP_TAIL = "warp_tail"
    FINAL_ROUND = "final_round"
    STAGING = "staging"
    PREVIOUS_CHUNK = "previous_chunk"
    LOAD_Q = "load_Q"
    LOAD_K = "load_K"
    LOAD_V = "load_V"
    PREVIOUS_SCORE = "previous_score"
    PREVIOUS_CAST = "previous_cast"
    PREVIOUS_PV = "previous_PV"
    NEXT_QK = "next_QK"
    SOURCE0 = "source0"
    SOURCE1 = "source1"
    SCALE = "scale"
    CAST = "cast"
    LOGSUM = "logsum"
    SCALE_O = "scale_O"
    NORMALIZATION = "normalization"


@dataclass(frozen=True)
class IndexExpr:
    constant: int = 0
    terms: tuple[tuple[LoopVar | str, int], ...] = ()

    def evaluate(self, bindings):
        return self.constant + sum(
            bindings[var] * coefficient for var, coefficient in self.terms
        )


@dataclass(frozen=True)
class TileId:
    index: IndexExpr


@dataclass(frozen=True)
class IterationId:
    index: IndexExpr


@dataclass(frozen=True)
class PipelineStage:
    iteration: IterationId
    modulus: int
    generation_counts_per_wave: tuple[int, ...]


@dataclass(frozen=True)
class SlotUse:
    buffer: BufferRole
    stage: PipelineStage


@dataclass(frozen=True)
class Dependency:
    source: OpRole | str
    phase: Phase
    relation: Relation


@dataclass(frozen=True)
class SymbolicRegion:
    tensor: str
    offsets: tuple[IndexExpr, ...]
    shape: tuple[int, ...]
    valid_axis: str = ""
    row_index: str = ""

    def offsets_at(self, bindings):
        return tuple(expr.evaluate(bindings) for expr in self.offsets)


@dataclass(frozen=True)
class TileLoop:
    variable: LoopVar
    count: int
    predicate: TailPredicate = TailPredicate.FULL_TILE


@dataclass(frozen=True)
class PipelineRegion:
    loop: TileLoop
    depth: int
    tile: TileId
    stage: PipelineStage


@dataclass(frozen=True)
class AttentionPartition:
    request: int
    query_head: int
    kv_head: int
    split: int
    start: int
    end: int


@dataclass(frozen=True)
class LoopState:
    producer: OpRole | str
    initializer: OpRole


@dataclass(frozen=True)
class WorkGroup:
    bindings: tuple[tuple[LoopVar | str, int], ...] = ()
    aliases: tuple[tuple[str, str], ...] = ()
    ranges: tuple[tuple[str, int, int], ...] = ()
    constants: tuple[tuple[str, float], ...] = ()
    iterations: tuple[tuple[tuple[LoopVar, int], ...], ...] = ()


@dataclass(frozen=True)
class ScheduleDomain:
    dimensions: tuple[int, ...]
    num_sms: int
    cta_count: int
    tile_count: int
    waves: int
    pipeline: PipelineRegion
    loops: tuple[TileLoop, ...]
    order: str = "original"
    partitions: tuple[AttentionPartition, ...] = ()
    states: tuple[LoopState, ...] = ()
    matrix_groups: tuple[MatrixGroup, ...] = ()
    workgroups: tuple[WorkGroup, ...] = ()
    partial_outputs: tuple[str, ...] = ()


class InitialValue(str, Enum):
    NONE = "none"
    ZERO = "zero"
    NEGATIVE_INFINITY = "negative_infinity"


@dataclass(frozen=True)
class ValueId:
    producer: OpRole | str
    result: int = 0
    iteration_delta: int = 0
    initial: InitialValue = InitialValue.NONE
    offsets: tuple[int, ...] = ()
    shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class SSAValue:
    id: ValueId
    shape: tuple[int, ...]
    dtype: str = "fp32"
    axes: tuple[str, ...] = ()
