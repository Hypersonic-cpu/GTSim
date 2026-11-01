"""The two typed math contracts; readable formulas are printer output only."""

from dataclasses import dataclass, replace
from enum import Enum
from .common import Header, require, MatrixGroup
from .primitives import PrimitiveSpec


class MathKind(str, Enum):
    TOPK = "topk"
    ARGMAX = "argmax"
    MASK_SELECTED = "mask_selected"
    TOPK_PACK = "topk_pack"
    HISTOGRAM = "histogram"
    PREFIX_SUM = "prefix_sum"
    SHIFT_RIGHT = "shift_right"
    EXCLUSIVE_SCAN = "exclusive_scan"
    PACK_VECTOR = "pack_vector"
    CONSTANT = "constant"
    PARAMETER = "parameter"
    IOTA = "iota"
    RSQRT = "rsqrt"
    SIN = "sin"
    COS = "cos"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    LOG = "log"
    MAXIMUM = "maximum"
    SCALE = "scale"
    ZERO = "zero"
    NEG_INFINITY = "negative_infinity"
    MATMUL = "matmul"
    MAX_REDUCTION = "max_reduction"
    SUM_REDUCTION = "sum_reduction"
    EXP = "exp"
    RESCALE = "rescale"
    NORMALIZE = "normalize"
    CAST = "cast"
    STREAMING_MAX = "streaming_max"
    STREAMING_SUM = "streaming_sum"
    STATE_LSE = "state_lse"


@dataclass(frozen=True)
class IndexDomain:
    axis: str
    extent: int


@dataclass(frozen=True)
class OperandContract:
    role: str
    axes: tuple[str, ...]
    dtype: str
    output: bool = False


@dataclass(frozen=True)
class MathOperation:
    kind: MathKind
    inputs: tuple[str, ...]
    output: str
    reduction_axes: tuple[str, ...] = ()
    dtype: str = "fp32"
    output_axes: tuple[str, ...] = ()
    scale_by_inverse_sqrt: str = ""
    constants: tuple[tuple[str, float], ...] = ()
    parameter: str = ""


@dataclass(frozen=True)
class HeadMapping:
    query_heads: int
    kv_heads: int
    query_heads_per_kv_head: int


@dataclass(frozen=True)
class AlgorithmContract:
    domains: tuple[IndexDomain, ...]
    operands: tuple[OperandContract, ...]
    operations: tuple[MathOperation, ...]
    gqa: HeadMapping | None = None
    output_dtype: str = "fp16"
    valid_lengths: tuple[int, ...] = ()
    groups: tuple[MatrixGroup, ...] = ()


def contract_for(family, dims, dtype="fp16", valid_lengths=()):
    if family == "gemm":
        if len(dims) != 3:
            return None
        domains = tuple(
            IndexDomain(axis, extent) for axis, extent in zip(("M", "N", "K"), dims)
        )
        operands = (
            OperandContract("A", ("M", "K"), dtype),
            OperandContract("B", ("K", "N"), dtype),
            OperandContract("C", ("M", "N"), dtype, True),
        )
        ops = (
            MathOperation(MathKind.ZERO, (), "acc"),
            MathOperation(MathKind.MATMUL, ("A", "B", "acc"), "acc", ("K",)),
            MathOperation(MathKind.CAST, ("acc",), "C", dtype=dtype),
        )
        return AlgorithmContract(
            domains,
            operands,
            tuple(replace(op, output_axes=("M", "N")) for op in ops),
            output_dtype=dtype,
        )
    if family != "fa3" or len(dims) != 5:
        return None
    b, hq, hkv, q, kv = dims
    domains = tuple(
        IndexDomain(axis, extent)
        for axis, extent in zip(("B", "HQ", "HKV", "Q", "KV", "D"), dims + (128,))
    )
    operands = (
        OperandContract("Q", ("B", "HQ", "Q", "D"), dtype),
        OperandContract("K", ("B", "HKV", "D", "KV"), dtype),
        OperandContract("V", ("B", "HKV", "D", "KV"), dtype),
        OperandContract("O", ("B", "HQ", "Q", "D"), dtype, True),
    )
    ops = (
        MathOperation(MathKind.NEG_INFINITY, (), "m"),
        MathOperation(MathKind.ZERO, (), "l"),
        MathOperation(MathKind.ZERO, (), "acc"),
        MathOperation(MathKind.MATMUL, ("Q", "K"), "scores", ("D",)),
        MathOperation(MathKind.STREAMING_MAX, ("scores", "m"), "m_next", ("KV",)),
        MathOperation(MathKind.EXP, ("m", "m_next"), "alpha"),
        MathOperation(MathKind.EXP, ("scores", "m_next"), "P"),
        MathOperation(MathKind.STREAMING_SUM, ("P", "alpha", "l"), "l_next", ("KV",)),
        MathOperation(MathKind.RESCALE, ("alpha", "acc"), "scaled_acc"),
        MathOperation(MathKind.MATMUL, ("P", "V", "scaled_acc"), "acc_next", ("KV",)),
        MathOperation(MathKind.NORMALIZE, ("acc_next", "l_next"), "normalized"),
        MathOperation(MathKind.CAST, ("normalized",), "O", dtype=dtype),
    )
    typed = []
    for op in ops:
        if op.output == "acc_next":
            typed.append(
                MathOperation(
                    MathKind.CAST,
                    ("P",),
                    "P16",
                    dtype=dtype,
                    output_axes=("B", "HQ", "Q", "KV"),
                )
            )
            op = replace(op, inputs=("P16", "V", "scaled_acc"))
        axes = (
            ("B", "HQ", "Q", "KV")
            if op.output in ("scores", "P")
            else (
                ("B", "HQ", "Q", "D")
                if op.output in ("acc", "scaled_acc", "acc_next", "normalized", "O")
                else ("B", "HQ", "Q")
            )
        )
        typed.append(
            replace(
                op,
                output_axes=axes,
                scale_by_inverse_sqrt="D" if op.output == "scores" else "",
            )
        )
    ops = tuple(typed)
    return AlgorithmContract(
        domains,
        operands,
        ops,
        HeadMapping(hq, hkv, hq // hkv if hkv > 0 else 0),
        dtype,
        valid_lengths,
    )


@dataclass(frozen=True)
class Algorithm(Header):
    schema_version: int = 2
    family: str = "gemm"
    dimensions: tuple[int, ...] = (512, 512, 512)
    dtype: str = "fp16"
    head_dim: int = 128
    schedule_id: str = "reference"
    valid_lengths: tuple[int, ...] = ()
    groups: tuple[MatrixGroup, ...] = ()
    split_count: int = 1
    primitive: PrimitiveSpec | None = None
    causal: bool = False
    numerical_execution: bool = False
    contract: AlgorithmContract | None = None

    def __post_init__(self):
        if self.contract is None:
            object.__setattr__(
                self,
                "contract",
                self._expected_contract(),
            )

    def _expected_contract(self):
        if self.family == "vector":
            from ..kernels.vector import contract_for as vector_contract

            return vector_contract(self.primitive)
        if self.schedule_id == "decode_combine":
            from ..kernels.hopper.attention_combine import contract

            return contract(self.dimensions, self.dtype, self.split_count)
        return replace(
            contract_for(self.family, self.dimensions, self.dtype, self.valid_lengths),
            groups=self.groups,
        )

    def verify(self):
        from ..verify.types import verify_types

        verify_types(self, location=self.symbol)
        super().verify()
        if self.family == "vector":
            from ..kernels.vector import verify_spec

            require(
                self.schedule_id == "simt_vector" and self.primitive is not None,
                self.symbol,
                "vector specialization requires typed parameters",
            )
            verify_spec(self.primitive)
            require(
                self.contract == self._expected_contract(),
                self.symbol,
                "invalid vector mathematical contract",
            )
            require(
                self.dtype == "bf16"
                and not self.groups
                and not self.valid_lengths
                and self.split_count == 1
                and not self.causal
                and not self.numerical_execution,
                self.symbol,
                "invalid vector algorithm options",
            )
            return
        require(
            self.primitive is None,
            self.symbol,
            "primitive parameters require vector family",
        )
        require(
            self.family in ("gemm", "fa3"), self.symbol, "unknown algorithm template"
        )
        require(
            self.dtype in ("fp16", "bf16")
            and self.head_dim == 128
            and self.schedule_id
            in (
                "reference",
                "small_m_mma",
                "grouped_small_m_mma",
                "decode_simt",
                "decode_combine",
            )
            and (self.schedule_id != "reference" or self.dtype == "fp16"),
            self.symbol,
            "reference supports fp16 and D128 only",
        )
        require(
            all(type(d) is int and d > 0 for d in self.dimensions),
            self.symbol,
            "invalid concrete index domain",
        )
        require(
            not self.causal and not self.numerical_execution,
            self.symbol,
            "unsupported causal/numerical execution",
        )
        if self.schedule_id in ("decode_simt", "decode_combine"):
            require(
                self.family == "fa3" and len(self.dimensions) == 5,
                self.symbol,
                "decode schedule requires attention",
            )
            b, hq, hkv, q, kv = self.dimensions
            require(
                q == 1 and hq % hkv == 0,
                self.symbol,
                "decode requires Tq=1 and valid GQA",
            )
            require(
                len(self.valid_lengths) == b
                and all(0 < n <= kv for n in self.valid_lengths),
                self.symbol,
                "missing per-request valid attention lengths",
            )
            require(
                1 <= self.split_count <= min(self.valid_lengths),
                self.symbol,
                "invalid attention split count",
            )
        elif self.schedule_id == "grouped_small_m_mma":
            require(
                self.family == "gemm"
                and len(self.dimensions) == 3
                and bool(self.groups),
                self.symbol,
                "grouped GEMM requires matrix groups",
            )
            require(
                len({g.identity for g in self.groups}) == len(self.groups)
                and len({g.weight for g in self.groups}) == len(self.groups),
                self.symbol,
                "duplicate group/weight identity",
            )
            outputs = []
            for group in self.groups:
                require(
                    group.identity >= 0
                    and group.weight not in ("A", "C")
                    and bool(group.weight)
                    and bool(group.input_rows)
                    and len(group.input_rows) == len(group.output_rows)
                    and all(row >= 0 for row in group.input_rows),
                    self.symbol,
                    "invalid grouped matrix row mapping",
                )
                outputs.extend(group.output_rows)
            require(
                sorted(outputs) == list(range(self.dimensions[0])),
                self.symbol,
                "grouped outputs must cover each logical row exactly once",
            )
        elif self.schedule_id == "small_m_mma":
            require(
                self.family == "gemm"
                and len(self.dimensions) == 3
                and self.dimensions[0] in (1, 8, 16),
                self.symbol,
                "small-M schedule requires M=1,8,16",
            )
        elif self.family == "gemm":
            require(len(self.dimensions) == 3, self.symbol, "expected M,N,K")
            require(
                all(d % tile == 0 for d, tile in zip(self.dimensions, (128, 256, 64))),
                self.symbol,
                "reference GEMM needs complete BM128 BN256 BK64 tiles",
            )
        else:
            require(len(self.dimensions) == 5, self.symbol, "expected B,HQ,HKV,Q,KV")
            b, hq, hkv, q, kv = self.dimensions
            require(
                hq % hkv == 0 and q % 128 == 0 and kv % 128 == 0,
                self.symbol,
                "invalid GQA or full Q/KV128 geometry",
            )
        if self.schedule_id not in ("decode_simt", "decode_combine"):
            require(
                not self.valid_lengths and self.split_count == 1,
                self.symbol,
                "request partitions require decode schedule",
            )
        require(
            bool(self.groups) == (self.schedule_id == "grouped_small_m_mma"),
            self.symbol,
            "matrix groups require grouped schedule",
        )
        expected = self._expected_contract()
        require(
            self.contract is not None, self.symbol, "missing typed algorithm contract"
        )
        require(
            self.contract.domains == expected.domains
            and self.contract.groups == expected.groups
            and self.contract.valid_lengths == expected.valid_lengths,
            self.symbol,
            "invalid operand index domains",
        )
        require(
            self.contract.operands == expected.operands,
            self.symbol,
            "missing or invalid operand/output roles",
        )
        require(
            self.contract.operations == expected.operations,
            self.symbol,
            "missing or invalid initialization, math operation, reduction axes or output conversion",
        )
        require(
            self.contract.gqa == expected.gqa
            and self.contract.output_dtype == self.dtype,
            self.symbol,
            "invalid GQA mapping or output dtype",
        )

    @property
    def stages(self):
        return tuple(operation.kind.value for operation in self.contract.operations)

    @property
    def computation(self):
        return tuple(
            f"{operation.output} = {operation.kind.value}{operation.inputs} reduce{operation.reduction_axes}"
            for operation in self.contract.operations
        )
