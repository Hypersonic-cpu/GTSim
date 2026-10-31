"""The two typed math contracts; readable formulas are printer output only."""

from dataclasses import dataclass, replace
from enum import Enum
from .common import Header, require


class MathKind(str, Enum):
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


def contract_for(family, dims):
    if family == "gemm":
        if len(dims) != 3:
            return None
        domains = tuple(
            IndexDomain(axis, extent) for axis, extent in zip(("M", "N", "K"), dims)
        )
        operands = (
            OperandContract("A", ("M", "K"), "fp16"),
            OperandContract("B", ("K", "N"), "fp16"),
            OperandContract("C", ("M", "N"), "fp16", True),
        )
        ops = (
            MathOperation(MathKind.ZERO, (), "acc"),
            MathOperation(MathKind.MATMUL, ("A", "B", "acc"), "acc", ("K",)),
            MathOperation(MathKind.CAST, ("acc",), "C", dtype="fp16"),
        )
        return AlgorithmContract(
            domains, operands, tuple(replace(op, output_axes=("M", "N")) for op in ops)
        )
    if family != "fa3" or len(dims) != 5:
        return None
    b, hq, hkv, q, kv = dims
    domains = tuple(
        IndexDomain(axis, extent)
        for axis, extent in zip(("B", "HQ", "HKV", "Q", "KV", "D"), dims + (128,))
    )
    operands = (
        OperandContract("Q", ("B", "HQ", "Q", "D"), "fp16"),
        OperandContract("K", ("B", "HKV", "D", "KV"), "fp16"),
        OperandContract("V", ("B", "HKV", "D", "KV"), "fp16"),
        OperandContract("O", ("B", "HQ", "Q", "D"), "fp16", True),
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
        MathOperation(MathKind.CAST, ("normalized",), "O", dtype="fp16"),
    )
    typed = []
    for op in ops:
        if op.output == "acc_next":
            typed.append(
                MathOperation(
                    MathKind.CAST,
                    ("P",),
                    "P16",
                    dtype="fp16",
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
        domains, operands, ops, HeadMapping(hq, hkv, hq // hkv if hkv > 0 else 0)
    )


@dataclass(frozen=True)
class Algorithm(Header):
    schema_version: int = 2
    family: str = "gemm"
    dimensions: tuple[int, ...] = (512, 512, 512)
    dtype: str = "fp16"
    head_dim: int = 128
    causal: bool = False
    numerical_execution: bool = False
    contract: AlgorithmContract | None = None

    def __post_init__(self):
        if self.contract is None:
            object.__setattr__(
                self, "contract", contract_for(self.family, self.dimensions)
            )

    def verify(self):
        super().verify()
        require(
            self.family in ("gemm", "fa3"), self.symbol, "unknown algorithm template"
        )
        require(
            self.dtype == "fp16" and self.head_dim == 128,
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
        if self.family == "gemm":
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
        expected = contract_for(self.family, self.dimensions)
        require(
            self.contract is not None, self.symbol, "missing typed algorithm contract"
        )
        require(
            self.contract.domains == expected.domains,
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
            self.contract.gqa == expected.gqa and self.contract.output_dtype == "fp16",
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
