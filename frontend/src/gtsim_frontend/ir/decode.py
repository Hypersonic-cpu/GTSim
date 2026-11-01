"""Typed IR records for Qwen3 MoE single-token decode kernels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from .common import Header, Tensor, require


class KernelFamily(str, Enum):
    SKINNY_GEMM = "skinny_gemm"
    VECTOR = "vector"
    KV_UPDATE = "kv_update"
    FA3_DECODE = "fa3_decode"
    FA3_COMBINE = "fa3_combine"
    EXPERT_ASSIGNMENT = "expert_assignment"
    EXPERT_GEMM1 = "expert_gemm1"
    SWIGLU = "swiglu"
    EXPERT_GEMM2 = "expert_gemm2"
    WEIGHTED_MOE_SUM = "weighted_moe_sum"


class AccessMode(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


class Pipeline(str, Enum):
    LD = "LD"
    ST = "ST"
    TENSOR = "Tensor"
    SIMD = "SIMD"
    SFU = "SFU"
    SHFL = "SHFL"


class NativeOpcode(str, Enum):
    TMA_LOAD = "cp.dram2sram_tma"
    TMA_STORE = "cp.sram2dram_tma"
    WGMMA = "wgmma.m64n64k16"
    MMA_SYNC = "mma.sync.m16n8k16"
    SIMD_ADD = "simd.add"
    SIMD_SUB = "simd.sub"
    SIMD_MUL = "simd.mul"
    SIMD_FMA = "simd.fma"
    SIMD_MAX = "simd.max"
    SIMD_MIN = "simd.min"
    SIMD_CAST = "simd.cast"
    SIMD_FILL = "simd.fill"
    SIMD_IDENTITY = "simd.identity"
    REDUCE_MAX = "reduce_max"
    MERGE_MAX = "merge_max"
    REDUCE_SUM = "reduce_sum"
    MERGE_SUM = "merge_sum"
    LOCAL_MAX = "local_max"
    LOCAL_SUM = "local_sum"
    EXP2 = "exp2"
    SCORES_SCALE = "scores_scale"
    ACC_O_SCALE = "acc_o_scale"
    ACC_S_CAST = "acc_s_cast"
    ACC_O_DIV = "acc_o_div"
    LOGSUM = "logsum"
    SHFL = "shfl"
    SFU_EXP = "sfu.exp"
    SFU_LOG = "sfu.log"
    SFU_SQRT = "sfu.sqrt"
    SFU_RSQRT = "sfu.rsqrt"
    SFU_DIVIDE = "sfu.divide"
    SFU_SIN = "sfu.sin"
    SFU_COS = "sfu.cos"


class FlowOperator(str, Enum):
    LOAD = "load"
    STORE = "store"
    MATMUL = "matmul"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    FUSED_MULTIPLY_ADD = "fused_multiply_add"
    MAX_REDUCTION = "max_reduction"
    SUM_REDUCTION = "sum_reduction"
    EXPONENTIAL = "exponential"
    LOGARITHM = "logarithm"
    RECIPROCAL_SQRT = "reciprocal_sqrt"
    SIN = "sin"
    COS = "cos"
    NORMALIZE = "normalize"
    ROTARY_EMBEDDING = "rotary_embedding"
    TOPK = "topk"
    EXPERT_HISTOGRAM = "expert_histogram"
    EXPERT_PREFIX_SUM = "expert_prefix_sum"
    EXPERT_GATHER = "expert_gather"
    SILU = "silu"
    WEIGHTED_SUM = "weighted_sum"
    CAST = "cast"
    IDENTITY = "identity"
    FILL = "fill"


class FlowRole(str, Enum):
    LOAD_A = "load_a"
    LOAD_B = "load_b"
    MATMUL = "matmul"
    LOAD_Q = "load_q"
    LOAD_K = "load_k"
    LOAD_V = "load_v"
    QK = "qk"
    SOFTMAX_MAX = "softmax_max"
    SOFTMAX_EXP = "softmax_exp"
    SOFTMAX_SUM = "softmax_sum"
    PV = "pv"
    NORMALIZE = "normalize"
    ASSIGN = "assignment"
    VECTOR = "vector"
    STORE = "store"
    COMBINE = "combine"


class FlowLoopAxis(str, Enum):
    CTA = "cta"
    BATCH = "batch"
    HEAD = "head"
    SPLIT = "split"
    KV_TILE = "kv_tile"
    GEMM_K = "gemm_k"
    GEMM_N = "gemm_n"
    EXPERT = "expert"
    EXPERT_M = "expert_m"
    ROW = "row"


@dataclass(frozen=True)
class TensorUse:
    name: str
    tensor: Tensor
    access: AccessMode
    broadcast_rows: bool = False

    def verify(self):
        require(bool(self.name), "tensor use", "empty local name")
        require(
            isinstance(self.access, AccessMode), self.name, "invalid memory access mode"
        )
        self.tensor.verify()


@dataclass(frozen=True)
class FlowLoop:
    axis: FlowLoopAxis
    count: int
    tile: int = 1
    predicate: str = "tile_exists"

    def verify(self):
        require(self.count > 0 and self.tile > 0, self.axis.value, "invalid loop")
        require(
            self.predicate in ("tile_exists", "valid_rows", "full_tile"),
            self.axis.value,
            "unknown tail predicate",
        )


@dataclass(frozen=True)
class FlowOperation:
    id: int
    role: FlowRole
    operator: FlowOperator
    tensor: str = ""
    dependencies: tuple[int, ...] = ()
    loops: tuple[FlowLoopAxis, ...] = ()
    valid_region: str = ""


@dataclass(frozen=True)
class NativeOperation:
    id: int
    role: FlowRole
    opcode: NativeOpcode
    pipeline: Pipeline
    tensor: str = ""
    fragment_shape: tuple[int, int] = ()
    collective_warps: int = 1
    loop_axes: tuple[FlowLoopAxis, ...] = ()
    dependencies: tuple[int, ...] = ()


@dataclass(frozen=True)
class GemmWork:
    input: str
    weight: str
    output: str
    rows: int
    columns: int
    reduction: int
    tile_m: int = 16
    tile_n: int = 64
    tile_k: int = 16
    dtype: str = "bf16"


@dataclass(frozen=True)
class VectorStage:
    operation: FlowOperator
    elements: int
    repeats: int = 1
    input_indices: tuple[int, ...] = ()
    dependencies: tuple[int, ...] = ()


@dataclass(frozen=True)
class VectorWork:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    rows: int
    columns: int
    tile_rows: int
    tile_columns: int
    stages: tuple[VectorStage, ...]
    row_independent: bool = True
    output_width: int = 0
    epsilon: float = 1e-6


@dataclass(frozen=True)
class QKVTransformWork:
    qkv: str
    q_norm_weight: str
    k_norm_weight: str
    query_output: str
    key_output: str
    value_output: str
    batch: int
    positions: tuple[int, ...]
    query_heads: int = 32
    kv_heads: int = 4
    head_dim: int = 128
    epsilon: float = 1e-6
    rope_theta: float = 1000000.0


@dataclass(frozen=True)
class RouterTopKWork:
    logits: str
    expert_ids: str
    probabilities: str
    routes: tuple[Qwen3RoutingToken, ...]
    batch: int
    experts: int = 128
    top_k: int = 8


@dataclass(frozen=True)
class KVAppend:
    request: int
    head: int
    context_length: int
    new_key: str
    new_value: str
    key_cache: str
    value_cache: str


@dataclass(frozen=True)
class KVUpdateWork:
    appends: tuple[KVAppend, ...]
    head_dim: int = 128


@dataclass(frozen=True)
class AttentionChunk:
    request: int
    split: int
    start: int
    end: int


@dataclass(frozen=True)
class AttentionRequest:
    request: int
    key_caches: tuple[str, ...]
    value_caches: tuple[str, ...]


@dataclass(frozen=True)
class AttentionWork:
    query: str
    output: str
    requests: tuple[AttentionRequest, ...]
    context_lengths: tuple[int, ...]
    split_count: int
    chunks: tuple[AttentionChunk, ...]
    partial_output: str = ""
    partial_lse: str = ""
    query_heads: int = 32
    kv_heads: int = 4
    head_dim: int = 128
    kv_tile: int = 128


@dataclass(frozen=True)
class ExpertAssignment:
    token: int
    slot: int
    expert: int
    probability: float


@dataclass(frozen=True)
class ExpertGroup:
    expert: int
    row_start: int
    assignments: tuple[ExpertAssignment, ...]
    gate_up_weight: str
    down_weight: str


def _verify_assignments(assignments, groups, names, location, phase=None):
    require(bool(assignments), location, "empty expert assignment set")
    keys = [(a.token, a.slot) for a in assignments]
    require(len(set(keys)) == len(keys), location, "duplicate token/slot assignment")
    require(
        all(
            a.token >= 0
            and 0 <= a.slot < 8
            and 0 <= a.expert < 128
            and 0.0 <= a.probability <= 1.0
            for a in assignments
        ),
        location,
        "invalid expert assignment value",
    )
    per_token = {}
    for assignment in assignments:
        per_token.setdefault(assignment.token, []).append(assignment)
    require(
        all(
            len(rows) == 8
            and {a.slot for a in rows} == set(range(8))
            and abs(sum(a.probability for a in rows) - 1.0) <= 1e-5
            for rows in per_token.values()
        ),
        location,
        "each routed token must have 8 unique slots with normalized probabilities",
    )
    grouped = [a for group in groups for a in group.assignments]
    require(
        sorted((a.token, a.slot, a.expert) for a in grouped)
        == sorted((a.token, a.slot, a.expert) for a in assignments),
        location,
        "expert groups disagree with the routing assignment set",
    )
    require(
        len({group.expert for group in groups}) == len(groups)
        and all(group.assignments for group in groups),
        location,
        "empty or duplicate active expert group",
    )
    for group in groups:
        require(
            all(a.expert == group.expert for a in group.assignments),
            location,
            "group contains an assignment for a different expert",
        )
        if phase == "gemm1":
            require(
                group.gate_up_weight in names, location, "missing active Gate/Up weight"
            )
            require(
                names[group.gate_up_weight].tensor.shape == (2048, 1536),
                location,
                "Gate/Up weight shape mismatch",
            )
        if phase == "gemm2":
            require(group.down_weight in names, location, "missing active Down weight")
            require(
                names[group.down_weight].tensor.shape == (768, 2048),
                location,
                "Down weight shape mismatch",
            )


def _groups_from_assignments(assignments):
    grouped = {}
    for assignment in assignments:
        grouped.setdefault(assignment.expert, []).append(assignment)
    return tuple(
        ExpertGroup(expert, 0, tuple(rows), "", "")
        for expert, rows in sorted(grouped.items())
    )


@dataclass(frozen=True)
class AssignmentWork:
    activation: str
    expert_ids: str
    expert_probabilities: str
    sorted_activation: str
    sorted_assignments: str
    sorted_probabilities: str
    assignments: tuple[ExpertAssignment, ...]
    groups: tuple[ExpertGroup, ...]
    hidden_size: int = 2048
    alignment: int = 16


@dataclass(frozen=True)
class ExpertGemmWork:
    variant: str
    phase: str
    input: str
    output: str
    rows: int
    columns: int
    reduction: int
    groups: tuple[ExpertGroup, ...]
    tile_m: int = 16
    tile_n: int = 64
    tile_k: int = 16


@dataclass(frozen=True)
class CombineWork:
    partial_output: str
    partial_lse: str
    output: str
    requests: int
    heads: int
    splits: int
    head_dim: int = 128


@dataclass(frozen=True)
class WeightedSumWork:
    expert_output: str
    expert_probabilities: str
    assignments: str
    output: str
    batch: int
    top_k: int = 8
    hidden_size: int = 2048
    direct_order: bool = False
    grouped_assignments: tuple[ExpertAssignment, ...] = ()


KernelWork = (
    GemmWork
    | VectorWork
    | QKVTransformWork
    | RouterTopKWork
    | KVUpdateWork
    | AttentionWork
    | AssignmentWork
    | ExpertGemmWork
    | CombineWork
    | WeightedSumWork
)


@dataclass(frozen=True)
class DecodeAlgorithm(Header):
    schema_version: int = 3
    family: KernelFamily = KernelFamily.VECTOR
    layer: int = 0
    variant: str = ""
    tensors: tuple[TensorUse, ...] = ()
    work: KernelWork | None = None

    def verify(self):
        super().verify()
        require(self.layer in (0, 1), self.symbol, "invalid layer index")
        require(bool(self.variant), self.symbol, "missing kernel variant")
        require(bool(self.tensors), self.symbol, "empty kernel tensor contract")
        require(
            len({t.name for t in self.tensors}) == len(self.tensors),
            self.symbol,
            "duplicate local tensor",
        )
        for tensor in self.tensors:
            tensor.verify()
        require(self.work is not None, self.symbol, "missing typed kernel semantics")
        names = {t.name: t for t in self.tensors}
        if isinstance(self.work, GemmWork):
            require(
                self.family == KernelFamily.SKINNY_GEMM,
                self.symbol,
                "wrong GEMM family",
            )
            require(
                all(
                    name in names
                    for name in (self.work.input, self.work.weight, self.work.output)
                ),
                self.symbol,
                "missing GEMM tensor",
            )
            require(
                (
                    names[self.work.input].tensor.shape
                    == (self.work.rows, self.work.reduction)
                )
                and (
                    names[self.work.weight].tensor.shape
                    == (self.work.reduction, self.work.columns)
                )
                and (
                    names[self.work.output].tensor.shape
                    == (self.work.rows, self.work.columns)
                ),
                self.symbol,
                "GEMM shape contract mismatch",
            )
            require(
                self.work.tile_m == 16
                and self.work.tile_n in (8, 16, 32, 64)
                and self.work.tile_k == 16,
                self.symbol,
                "unsupported small-M tile",
            )
        elif isinstance(self.work, VectorWork):
            require(
                self.family
                in (
                    KernelFamily.VECTOR,
                    KernelFamily.SWIGLU,
                    KernelFamily.EXPERT_ASSIGNMENT,
                ),
                self.symbol,
                "wrong vector family",
            )
            require(
                self.work.rows > 0
                and self.work.columns > 0
                and self.work.tile_rows > 0
                and self.work.tile_columns > 0,
                self.symbol,
                "invalid vector geometry",
            )
            require(
                bool(self.work.inputs) and bool(self.work.outputs),
                self.symbol,
                "vector work must declare inputs and outputs",
            )
            require(
                all(name in names for name in self.work.inputs + self.work.outputs),
                self.symbol,
                "missing vector tensor",
            )
            require(
                bool(self.work.stages), self.symbol, "empty vector operation sequence"
            )
            for index, stage in enumerate(self.work.stages):
                require(
                    stage.elements > 0 and stage.repeats > 0,
                    self.symbol,
                    "invalid vector stage",
                )
                require(
                    all(
                        0 <= source < len(self.work.inputs)
                        for source in stage.input_indices
                    ),
                    self.symbol,
                    "invalid vector input edge",
                )
                require(
                    all(0 <= dependency < index for dependency in stage.dependencies),
                    self.symbol,
                    "invalid vector stage edge",
                )
            if self.work.output_width:
                require(
                    names[self.work.outputs[0]].tensor.shape
                    == (self.work.rows, self.work.output_width),
                    self.symbol,
                    "vector output shape mismatch",
                )
        elif isinstance(self.work, QKVTransformWork):
            require(
                self.family == KernelFamily.VECTOR,
                self.symbol,
                "wrong QKV transform family",
            )
            require(
                self.work.query_heads == 32
                and self.work.kv_heads == 4
                and self.work.head_dim == 128
                and self.work.batch > 0
                and len(self.work.positions) == self.work.batch,
                self.symbol,
                "invalid Qwen QKV transform geometry",
            )
            required = (
                self.work.qkv,
                self.work.q_norm_weight,
                self.work.k_norm_weight,
                self.work.query_output,
                self.work.key_output,
                self.work.value_output,
            )
            require(
                all(name in names for name in required),
                self.symbol,
                "missing QKV transform tensor",
            )
            require(
                names[self.work.qkv].tensor.shape == (self.work.batch, 5120)
                and names[self.work.q_norm_weight].tensor.shape == (1, 128)
                and names[self.work.k_norm_weight].tensor.shape == (1, 128)
                and names[self.work.query_output].tensor.shape
                == (self.work.batch * 32, 128)
                and names[self.work.key_output].tensor.shape
                == (self.work.batch * 4, 128)
                and names[self.work.value_output].tensor.shape
                == (self.work.batch * 4, 128),
                self.symbol,
                "QKV transform shape contract mismatch",
            )
            require(
                all(position >= 0 for position in self.work.positions),
                self.symbol,
                "invalid RoPE positions",
            )
            require(
                self.work.epsilon == 1e-6 and self.work.rope_theta == 1000000.0,
                self.symbol,
                "Qwen3 RMSNorm/RoPE parameters mismatch",
            )
        elif isinstance(self.work, RouterTopKWork):
            require(
                self.family == KernelFamily.VECTOR,
                self.symbol,
                "wrong Router/TopK family",
            )
            require(
                self.work.experts == 128
                and self.work.top_k == 8
                and self.work.batch > 0,
                self.symbol,
                "invalid Router/TopK geometry",
            )
            require(
                len(self.work.routes) == self.work.batch,
                self.symbol,
                "Router trace batch mismatch",
            )
            require(
                all(
                    name in names
                    for name in (
                        self.work.logits,
                        self.work.expert_ids,
                        self.work.probabilities,
                    )
                ),
                self.symbol,
                "missing Router/TopK tensor",
            )
            require(
                names[self.work.logits].tensor.shape == (self.work.batch, 128)
                and names[self.work.expert_ids].tensor.shape == (self.work.batch, 8)
                and names[self.work.probabilities].tensor.shape == (self.work.batch, 8)
                and names[self.work.expert_ids].tensor.dtype == "int32",
                self.symbol,
                "Router/TopK shape or dtype contract mismatch",
            )
            for token, route in enumerate(self.work.routes):
                require(
                    route.token == token
                    and len(route.expert_ids) == len(route.probabilities) == 8,
                    self.symbol,
                    "invalid Top-8 trace row",
                )
                require(
                    len(set(route.expert_ids)) == 8
                    and all(0 <= e < 128 for e in route.expert_ids),
                    self.symbol,
                    "invalid Top-8 expert IDs",
                )
                require(
                    all(p >= 0 for p in route.probabilities)
                    and abs(sum(route.probabilities) - 1.0) <= 1e-5,
                    self.symbol,
                    "Top-8 probabilities are not renormalized",
                )
        elif isinstance(self.work, KVUpdateWork):
            require(
                self.family == KernelFamily.KV_UPDATE,
                self.symbol,
                "wrong KV update family",
            )
            require(bool(self.work.appends), self.symbol, "empty KV update")
            require(
                self.work.head_dim == 128,
                self.symbol,
                "KV update head dimension mismatch",
            )
            append_slots = {
                (append.request, append.head) for append in self.work.appends
            }
            require(
                len(append_slots) == len(self.work.appends),
                self.symbol,
                "duplicate KV append target",
            )
            require(
                len(append_slots) % 4 == 0
                and append_slots
                == {
                    (request, head)
                    for request in range(len(append_slots) // 4)
                    for head in range(4)
                },
                self.symbol,
                "KV update must append one token to every request/KV head",
            )
            for append in self.work.appends:
                require(
                    append.request >= 0
                    and 0 <= append.head < 4
                    and append.context_length >= 0,
                    self.symbol,
                    "invalid KV append address",
                )
                require(
                    all(
                        name in names
                        for name in (
                            append.new_key,
                            append.new_value,
                            append.key_cache,
                            append.value_cache,
                        )
                    ),
                    self.symbol,
                    "missing KV append tensor",
                )
                require(
                    names[append.key_cache].tensor.shape
                    == names[append.value_cache].tensor.shape
                    == (append.context_length + 1, 128),
                    self.symbol,
                    "KV append cache shape mismatch",
                )
                require(
                    names[append.new_key].tensor.shape
                    == names[append.new_value].tensor.shape
                    and names[append.new_key].tensor.shape[1] == 128
                    and append.request * 4 + append.head
                    < names[append.new_key].tensor.shape[0],
                    self.symbol,
                    "KV source row is outside projected K/V",
                )
        elif isinstance(self.work, AttentionWork):
            require(
                self.family == KernelFamily.FA3_DECODE,
                self.symbol,
                "wrong attention family",
            )
            require(
                self.work.query_heads == 32
                and self.work.kv_heads == 4
                and self.work.head_dim == 128
                and self.work.query_heads % self.work.kv_heads == 0,
                self.symbol,
                "invalid Qwen GQA geometry",
            )
            require(
                len(self.work.context_lengths) == len(self.work.requests),
                self.symbol,
                "context/request count mismatch",
            )
            require(
                all(
                    len(r.key_caches) == len(r.value_caches) == self.work.kv_heads
                    for r in self.work.requests
                ),
                self.symbol,
                "invalid KV-head mapping",
            )
            require(self.work.query in names, self.symbol, "missing attention query")
            batch = len(self.work.requests)
            require(
                names[self.work.query].tensor.shape == (batch * 32, 128),
                self.symbol,
                "query shape must be one 128-d vector per request/head",
            )
            require(
                all(
                    name in names
                    for r in self.work.requests
                    for name in r.key_caches + r.value_caches
                ),
                self.symbol,
                "missing request KV allocation",
            )
            require(
                len(self.work.requests) == len(self.work.context_lengths),
                self.symbol,
                "attention batch mismatch",
            )
            for request, context_length in zip(
                self.work.requests, self.work.context_lengths
            ):
                for name in request.key_caches + request.value_caches:
                    require(
                        names[name].tensor.shape == (context_length + 1, 128),
                        self.symbol,
                        "KV cache region must match that request's effective length",
                    )
            if self.work.split_count > 1:
                require(
                    self.variant == "split_kv"
                    and self.work.partial_output in names
                    and self.work.partial_lse in names,
                    self.symbol,
                    "missing Split-KV partials",
                )
                require(
                    not self.work.output,
                    self.symbol,
                    "Split-KV decode must write partials before reduction",
                )
                require(
                    names[self.work.partial_output].tensor.shape
                    == (batch * 32 * self.work.split_count, 128),
                    self.symbol,
                    "Split-KV partial output shape mismatch",
                )
                require(
                    names[self.work.partial_lse].tensor.shape
                    == (batch * 32 * self.work.split_count, 1),
                    self.symbol,
                    "Split-KV partial logsum shape mismatch",
                )
            else:
                require(
                    self.variant == "non_split"
                    and not self.work.partial_output
                    and not self.work.partial_lse,
                    self.symbol,
                    "invalid non-split contract",
                )
                require(
                    self.work.output in names
                    and names[self.work.output].tensor.shape == (batch, 4096),
                    self.symbol,
                    "non-split attention output must flatten 32 heads per request",
                )
            expected = tuple(length + 1 for length in self.work.context_lengths)
            ranges = {request: [] for request in range(len(expected))}
            split_ids = {request: set() for request in range(len(expected))}
            for chunk in self.work.chunks:
                require(
                    0 <= chunk.request < len(expected)
                    and chunk.start >= 0
                    and chunk.end > chunk.start
                    and chunk.end <= expected[chunk.request],
                    self.symbol,
                    "invalid attention chunk",
                )
                require(
                    0 <= chunk.split < self.work.split_count
                    and chunk.split not in split_ids[chunk.request],
                    self.symbol,
                    "invalid or duplicate attention split ID",
                )
                split_ids[chunk.request].add(chunk.split)
                ranges[chunk.request].append((chunk.start, chunk.end))
            for request, request_ranges in ranges.items():
                ordered = sorted(request_ranges)
                require(
                    bool(ordered)
                    and ordered[0][0] == 0
                    and ordered[-1][1] == expected[request],
                    self.symbol,
                    "attention does not cover effective KV length",
                )
                require(
                    all(a[1] == b[0] for a, b in zip(ordered, ordered[1:])),
                    self.symbol,
                    "attention KV chunks overlap or leave a gap",
                )
                require(
                    len(ordered) == self.work.split_count,
                    self.symbol,
                    "attention split count does not match per-request work",
                )
        elif isinstance(self.work, AssignmentWork):
            require(
                self.family == KernelFamily.EXPERT_ASSIGNMENT,
                self.symbol,
                "wrong assignment family",
            )
            require(
                all(
                    name in names
                    for name in (
                        self.work.activation,
                        self.work.expert_ids,
                        self.work.expert_probabilities,
                        self.work.sorted_activation,
                        self.work.sorted_assignments,
                        self.work.sorted_probabilities,
                    )
                ),
                self.symbol,
                "missing assignment tensor",
            )
            require(
                self.work.hidden_size == 2048 and self.work.alignment == 16,
                self.symbol,
                "invalid grouped assignment contract",
            )
            _verify_assignments(
                self.work.assignments, self.work.groups, names, self.symbol
            )
            assignment_count = len(self.work.assignments)
            require(
                all(
                    a.token < names[self.work.activation].tensor.shape[0]
                    for a in self.work.assignments
                ),
                self.symbol,
                "assignment token index outside activation batch",
            )
            require(
                names[self.work.expert_ids].tensor.shape[1] == 8
                and names[self.work.expert_probabilities].tensor.shape[1] == 8,
                self.symbol,
                "assignment inputs must be Top-8 routes",
            )
            require(
                names[self.work.sorted_activation].tensor.shape
                == (assignment_count, 2048),
                self.symbol,
                "sorted activation shape mismatch",
            )
            require(
                names[self.work.sorted_assignments].tensor.shape
                == (assignment_count, 3)
                and names[self.work.sorted_assignments].tensor.dtype == "int32",
                self.symbol,
                "sorted assignment metadata shape mismatch",
            )
            require(
                names[self.work.sorted_probabilities].tensor.shape
                == (assignment_count, 1),
                self.symbol,
                "sorted probability shape mismatch",
            )
        elif isinstance(self.work, ExpertGemmWork):
            expected_family = (
                KernelFamily.EXPERT_GEMM1
                if self.work.phase == "gemm1"
                else KernelFamily.EXPERT_GEMM2
            )
            require(
                self.family == expected_family, self.symbol, "wrong expert GEMM family"
            )
            require(
                self.work.variant in ("direct", "grouped"),
                self.symbol,
                "unknown expert strategy",
            )
            require(
                self.work.tile_m == 16
                and self.work.tile_n == 64
                and self.work.tile_k == 16,
                self.symbol,
                "unsupported grouped expert tile",
            )
            require(
                self.work.rows > 0
                and self.work.columns > 0
                and self.work.reduction > 0,
                self.symbol,
                "invalid expert GEMM shape",
            )
            require(
                self.work.output in names
                and self.work.input in names
                and bool(self.work.groups),
                self.symbol,
                "missing expert GEMM tensor/group",
            )
            _verify_assignments(
                tuple(a for g in self.work.groups for a in g.assignments),
                self.work.groups,
                names,
                self.symbol,
                self.work.phase,
            )
            assignment_count = sum(len(g.assignments) for g in self.work.groups)
            require(
                self.work.rows == assignment_count,
                self.symbol,
                "expert GEMM row count must equal routed assignments",
            )
            require(
                self.work.phase in ("gemm1", "gemm2"),
                self.symbol,
                "invalid expert GEMM phase",
            )
            expected_shape = (2048, 1536) if self.work.phase == "gemm1" else (768, 2048)
            require(
                (self.work.reduction, self.work.columns) == expected_shape,
                self.symbol,
                "expert GEMM dimensions disagree with Qwen3",
            )
            input_shape = names[self.work.input].tensor.shape
            expected_input_rows = (
                assignment_count
                if self.work.variant == "grouped" or self.work.phase == "gemm2"
                else 1
            )
            require(
                input_shape == (expected_input_rows, self.work.reduction),
                self.symbol,
                "expert GEMM input shape mismatch",
            )
            require(
                names[self.work.output].tensor.shape
                == (assignment_count, self.work.columns),
                self.symbol,
                "expert GEMM output shape mismatch",
            )
            input_rows = names[self.work.input].tensor.shape[0]
            require(
                (self.work.variant == "grouped" and input_rows == assignment_count)
                or (
                    self.work.variant == "direct"
                    and input_rows
                    > max(a.token for g in self.work.groups for a in g.assignments)
                ),
                self.symbol,
                "expert GEMM input rows do not cover routed tokens",
            )
            if self.work.variant == "grouped":
                cursor = 0
                for group in sorted(self.work.groups, key=lambda item: item.row_start):
                    require(
                        group.row_start == cursor,
                        self.symbol,
                        "group row ranges are not compact and ordered",
                    )
                    cursor += len(group.assignments)
                require(
                    cursor == assignment_count,
                    self.symbol,
                    "group rows do not cover the assignments",
                )
            else:
                require(
                    all(
                        len(group.assignments) == 1
                        and group.row_start
                        == group.assignments[0].token * 8 + group.assignments[0].slot
                        for group in self.work.groups
                    ),
                    self.symbol,
                    "direct expert rows must preserve token/slot identity",
                )
        elif isinstance(self.work, CombineWork):
            require(
                self.family == KernelFamily.FA3_COMBINE
                and self.variant == "split_combine",
                self.symbol,
                "wrong Split-KV combine family",
            )
            require(
                all(
                    name in names
                    for name in (
                        self.work.partial_output,
                        self.work.partial_lse,
                        self.work.output,
                    )
                ),
                self.symbol,
                "missing combine tensor",
            )
            require(
                self.work.splits > 1
                and self.work.heads == 32
                and self.work.head_dim == 128,
                self.symbol,
                "invalid combine geometry",
            )
            require(
                names[self.work.partial_output].tensor.shape
                == (
                    self.work.requests * self.work.heads * self.work.splits,
                    self.work.head_dim,
                ),
                self.symbol,
                "partial attention output dimensions mismatch",
            )
            require(
                names[self.work.partial_lse].tensor.shape
                == (self.work.requests * self.work.heads * self.work.splits, 1),
                self.symbol,
                "partial attention LSE dimensions mismatch",
            )
            require(
                names[self.work.output].tensor.shape
                == (self.work.requests, self.work.heads * self.work.head_dim),
                self.symbol,
                "combined attention output dimensions mismatch",
            )
        elif isinstance(self.work, WeightedSumWork):
            require(
                self.family == KernelFamily.WEIGHTED_MOE_SUM
                and self.variant in ("direct", "grouped"),
                self.symbol,
                "wrong weighted-sum family",
            )
            require(
                all(
                    name in names
                    for name in (
                        self.work.expert_output,
                        self.work.expert_probabilities,
                        self.work.assignments,
                        self.work.output,
                    )
                ),
                self.symbol,
                "missing weighted-sum tensor",
            )
            require(
                self.work.batch > 0
                and self.work.top_k == 8
                and self.work.hidden_size == 2048,
                self.symbol,
                "invalid weighted-sum dimensions",
            )
            require(
                names[self.work.expert_output].tensor.shape
                == (self.work.batch * 8, 2048),
                self.symbol,
                "weighted expert output shape mismatch",
            )
            if self.work.direct_order:
                require(
                    names[self.work.expert_probabilities].tensor.shape
                    == (self.work.batch, 8),
                    self.symbol,
                    "direct weighted probability shape mismatch",
                )
                require(
                    names[self.work.assignments].tensor.shape == (self.work.batch, 8),
                    self.symbol,
                    "direct Top-8 route ID shape mismatch",
                )
            else:
                require(
                    names[self.work.expert_probabilities].tensor.shape
                    == (self.work.batch * 8, 1),
                    self.symbol,
                    "grouped weighted probability shape mismatch",
                )
                require(
                    names[self.work.assignments].tensor.shape
                    == (self.work.batch * 8, 3),
                    self.symbol,
                    "grouped assignment map shape mismatch",
                )
                require(
                    len(self.work.grouped_assignments) == self.work.batch * 8,
                    self.symbol,
                    "weighted grouped assignment count mismatch",
                )
                _verify_assignments(
                    self.work.grouped_assignments,
                    _groups_from_assignments(self.work.grouped_assignments),
                    names,
                    self.symbol,
                )
            require(
                names[self.work.output].tensor.shape == (self.work.batch, 2048),
                self.symbol,
                "weighted MoE output shape mismatch",
            )
        else:
            require(False, self.symbol, "unknown typed kernel semantics")


@dataclass(frozen=True)
class DecodeFlowKernel(Header):
    schema_version: int = 3
    algorithm: DecodeAlgorithm | None = None
    loops: tuple[FlowLoop, ...] = ()
    ops: tuple[FlowOperation, ...] = ()

    def verify(self):
        super().verify()
        require(self.algorithm is not None, self.symbol, "missing algorithm IR")
        self.algorithm.verify()
        require(
            self.algorithm.symbol == self.symbol,
            self.symbol,
            "algorithm symbol mismatch",
        )
        require(
            bool(self.loops) and bool(self.ops), self.symbol, "empty structured flow"
        )
        for loop in self.loops:
            loop.verify()
        require(
            tuple(op.id for op in self.ops) == tuple(range(len(self.ops))),
            self.symbol,
            "Flow operation IDs must be dense",
        )
        for op in self.ops:
            require(
                all(
                    0 <= dep < len(self.ops) and dep != op.id for dep in op.dependencies
                ),
                self.symbol,
                "dangling Flow dependency",
            )
        pending = {op.id: set(op.dependencies) for op in self.ops}
        ready = [op_id for op_id, deps in pending.items() if not deps]
        visited = 0
        while ready:
            op_id = ready.pop()
            visited += 1
            for child, deps in pending.items():
                if op_id in deps:
                    deps.remove(op_id)
                    if not deps:
                        ready.append(child)
        require(visited == len(self.ops), self.symbol, "Flow dependency cycle")


@dataclass(frozen=True)
class DecodeNativeKernel(Header):
    schema_version: int = 3
    flow: DecodeFlowKernel | None = None
    target: str = "sm90"
    resource_warps: int = 4
    resource_shared_bytes: int = 32768
    resource_register_bytes: int = 32768
    ops: tuple[NativeOperation, ...] = ()

    def verify(self):
        super().verify()
        require(self.flow is not None, self.symbol, "missing structured Flow IR")
        self.flow.verify()
        require(self.target == "sm90", self.symbol, "unsupported native target")
        require(
            1 <= self.resource_warps <= 32
            and self.resource_shared_bytes > 0
            and self.resource_register_bytes > 0,
            self.symbol,
            "invalid native resources",
        )
        require(
            tuple((op.id, op.role, op.tensor, op.dependencies) for op in self.ops)
            == tuple(
                (op.id, op.role, op.tensor, op.dependencies) for op in self.flow.ops
            ),
            self.symbol,
            "native pattern graph does not match Flow",
        )
        for native, flow in zip(self.ops, self.flow.ops):
            require(
                native.opcode.value in {op.value for op in NativeOpcode},
                self.symbol,
                "unsupported native opcode",
            )


@dataclass(frozen=True)
class Qwen3RoutingToken:
    token: int
    expert_ids: tuple[int, ...]
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class DecodeInvocation:
    id: str
    layer: int
    name: str
    variant: str
    algorithm: DecodeAlgorithm
    bindings: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class DecodeProgram(Header):
    schema_version: int = 3
    model: str = "Qwen3-30B-A3B"
    mode: str = "single_decode_step"
    batch_size: int = 1
    layers: int = 1
    context_lengths: tuple[int, ...] = ()
    routing_seed: int = 0
    routing_trace: tuple[tuple[Qwen3RoutingToken, ...], ...] = ()
    tensors: tuple[Tensor, ...] = ()
    launches: tuple[DecodeInvocation, ...] = ()
    commands: tuple[tuple[str, str], ...] = ()
    streams: tuple[str, ...] = ("compute",)

    def verify(self):
        super().verify()
        require(
            self.model == "Qwen3-30B-A3B" and self.mode == "single_decode_step",
            self.symbol,
            "unsupported decode model/mode",
        )
        require(
            self.batch_size in (1, 8, 16) and self.layers in (1, 2),
            self.symbol,
            "unsupported batch/layer count",
        )
        require(
            len(self.context_lengths) == self.batch_size
            and all(10000 <= n <= 32000 for n in self.context_lengths),
            self.symbol,
            "context lengths must be one valid 10K-32K value per request",
        )
        require(
            len(self.streams) == 1 and bool(self.streams[0]),
            self.symbol,
            "exactly one ordered stream required",
        )
        require(
            len(self.routing_trace) == self.layers
            and all(len(layer) == self.batch_size for layer in self.routing_trace),
            self.symbol,
            "routing trace must have B token routes for every layer",
        )
        for layer, tokens in enumerate(self.routing_trace):
            for index, route in enumerate(tokens):
                require(
                    route.token == index
                    and len(route.expert_ids) == len(route.probabilities) == 8,
                    self.symbol,
                    "invalid top-8 route",
                )
                require(
                    len(set(route.expert_ids)) == 8
                    and all(0 <= e < 128 for e in route.expert_ids),
                    self.symbol,
                    "invalid/duplicate expert ID",
                )
                require(
                    all(p >= 0 for p in route.probabilities)
                    and sum(route.probabilities) > 0,
                    self.symbol,
                    "invalid routing probabilities",
                )
                require(
                    abs(sum(route.probabilities) - 1.0) <= 1e-5,
                    self.symbol,
                    "routing trace probabilities must be renormalized",
                )
        tensors = {t.id: t for t in self.tensors}
        require(
            len(tensors) == len(self.tensors), self.symbol, "duplicate program tensor"
        )
        for tensor in self.tensors:
            tensor.verify()
        ids = {launch.id for launch in self.launches}
        require(
            len(ids) == len(self.launches), self.symbol, "duplicate kernel invocation"
        )
        require(
            len(self.commands) == len(self.launches)
            and all(kind == "LaunchKernel" for kind, _ in self.commands),
            self.symbol,
            "decode stream must launch every kernel in order",
        )
        require(
            tuple(target for _, target in self.commands)
            == tuple(launch.id for launch in self.launches),
            self.symbol,
            "kernel stream order differs from invocations",
        )
        require(bool(self.launches), self.symbol, "empty decode stream")
        for invocation in self.launches:
            require(
                0 <= invocation.layer < self.layers
                and bool(invocation.name)
                and invocation.variant == invocation.algorithm.variant,
                invocation.id,
                "invalid invocation metadata",
            )
            invocation.algorithm.verify()
            binding = dict(invocation.bindings)
            uses = {t.name: t for t in invocation.algorithm.tensors}
            require(
                len(binding) == len(invocation.bindings) and set(binding) == set(uses),
                invocation.id,
                "invalid local tensor bindings",
            )
            for local, global_name in binding.items():
                require(global_name in tensors, invocation.id, "unbound program tensor")
                global_tensor = tensors[global_name]
                local_tensor = uses[local].tensor
                require(
                    (global_tensor.shape, global_tensor.dtype, global_tensor.tier)
                    == (local_tensor.shape, local_tensor.dtype, local_tensor.tier),
                    invocation.id,
                    "local tensor binding metadata mismatch",
                )
        expected_base = [
            "input_rmsnorm",
            "qkv_projection",
            "qkv_norm_rope",
            "kv_cache_update",
            "fa3_decode",
        ]
        require(
            any(
                launch.layer == 0 and launch.name == "fa3_decode"
                for launch in self.launches
            ),
            self.symbol,
            "missing FA3 Decode invocation",
        )
        attention = next(
            invocation.algorithm.work
            for invocation in self.launches
            if invocation.layer == 0 and invocation.name == "fa3_decode"
        )
        if isinstance(attention, AttentionWork) and attention.split_count > 1:
            expected_base.append("fa3_combine")
        expected_base.extend(
            (
                "attention_output_projection",
                "post_attention_residual_rmsnorm",
                "router_projection",
                "router_softmax_topk",
            )
        )
        if self.batch_size > 1:
            expected_base.append("expert_assignment")
        expected_base.extend(
            (
                "expert_gemm1",
                "swiglu",
                "expert_gemm2",
                "weighted_moe_sum",
                "final_residual_add",
            )
        )
        for layer in range(self.layers):
            layer_launches = [
                launch for launch in self.launches if launch.layer == layer
            ]
            require(
                [launch.name for launch in layer_launches] == expected_base,
                self.symbol,
                f"layer {layer} has an incomplete or reordered decode graph",
            )
            cache_names = [
                t.id for t in self.tensors if t.id.startswith(f"l{layer}.kv.")
            ]
            require(
                len(cache_names) == self.batch_size * 4 * 2,
                self.symbol,
                f"layer {layer} does not have independent K/V cache allocations per request/head",
            )
            kv_update = layer_launches[3].algorithm.work
            require(
                isinstance(kv_update, KVUpdateWork)
                and len({a.request for a in kv_update.appends}) == self.batch_size,
                self.symbol,
                f"layer {layer} must append one token per request",
            )
            append_contexts = {
                request: {
                    append.context_length
                    for append in kv_update.appends
                    if append.request == request
                }
                for request in range(self.batch_size)
            }
            require(
                all(
                    lengths == {self.context_lengths[request]}
                    for request, lengths in append_contexts.items()
                ),
                self.symbol,
                f"layer {layer} KV append offsets disagree with request contexts",
            )
            qkv = layer_launches[2].algorithm.work
            require(
                isinstance(qkv, QKVTransformWork)
                and qkv.positions == self.context_lengths,
                self.symbol,
                f"layer {layer} RoPE positions must match request context lengths",
            )
            route_work = next(
                launch.algorithm.work
                for launch in layer_launches
                if launch.name == "router_softmax_topk"
            )
            require(
                isinstance(route_work, RouterTopKWork)
                and route_work.routes == self.routing_trace[layer],
                self.symbol,
                f"layer {layer} Router/TopK does not consume its routing trace",
            )
            expert1 = next(
                launch.algorithm.work
                for launch in layer_launches
                if launch.name == "expert_gemm1"
            )
            expected_assignments = {
                (token.token, slot, expert, probability)
                for token in self.routing_trace[layer]
                for slot, (expert, probability) in enumerate(
                    zip(token.expert_ids, token.probabilities)
                )
            }
            actual_assignments = {
                (
                    assignment.token,
                    assignment.slot,
                    assignment.expert,
                    assignment.probability,
                )
                for group in expert1.groups
                for assignment in group.assignments
            }
            require(
                actual_assignments == expected_assignments,
                self.symbol,
                f"layer {layer} expert GEMM assignments disagree with the trace",
            )
            active_experts = {expert for _, _, expert, _ in expected_assignments}
            expert_weights = {
                t.name
                for t in next(
                    launch for launch in layer_launches if launch.name == "expert_gemm1"
                ).algorithm.tensors
                if t.name.startswith("w1_e")
            }
            require(
                expert_weights == {f"w1_e{expert}" for expert in active_experts},
                self.symbol,
                f"layer {layer} GEMM1 loads inactive or omits active expert weights",
            )
            expert2_launch = next(
                launch for launch in layer_launches if launch.name == "expert_gemm2"
            )
            expert2_weights = {
                tensor.name
                for tensor in expert2_launch.algorithm.tensors
                if tensor.name.startswith("w2_e")
            }
            require(
                expert2_weights == {f"w2_e{expert}" for expert in active_experts},
                self.symbol,
                f"layer {layer} GEMM2 loads inactive or omits active expert weights",
            )
            output_tensor = tensors.get(f"l{layer}.output")
            require(
                output_tensor is not None
                and output_tensor.shape == (self.batch_size, 2048),
                self.symbol,
                f"layer {layer} does not produce [B,2048]",
            )
            if layer == 1:
                first_binding = dict(layer_launches[0].bindings)
                prior_output = dict(
                    next(
                        launch.bindings
                        for launch in self.launches
                        if launch.layer == 0 and launch.name == "final_residual_add"
                    )
                )["output"]
                require(
                    first_binding["x"] == prior_output,
                    self.symbol,
                    "layer 1 must consume layer 0 output",
                )
        root = tensors.get("input.hidden")
        require(
            root is not None
            and root.shape == (self.batch_size, 2048)
            and root.read_only,
            self.symbol,
            "decode input must be a read-only [B,2048] activation",
        )
        if self.layers == 1:
            first_binding = dict(
                next(
                    launch.bindings
                    for launch in self.launches
                    if launch.layer == 0 and launch.name == "input_rmsnorm"
                )
            )
            require(
                first_binding["x"] == "input.hidden",
                self.symbol,
                "layer 0 must consume the decode input",
            )
