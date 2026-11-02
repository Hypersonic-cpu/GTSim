"""Algorithm-level computation classes for a timing compiler."""

from dataclasses import dataclass
from enum import Enum
from .common import Header, Tensor, Constant, DataType


class OperationClass(str, Enum):
    ARITHMETIC = "arithmetic"
    SPECIAL_FUNCTION = "special_function"
    REDUCTION = "reduction"
    MATRIX = "matrix"
    CONVERT = "convert"


class ArithmeticCost(str, Enum):
    SIMPLE = "simple"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


class SpecialFunction(str, Enum):
    TRIGONOMETRIC = "trigonometric"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    ROOT = "root"


class ReductionKind(str, Enum):
    SUM = "sum"
    EXTREMUM = "extremum"


class AlgorithmTrait(str, Enum):
    TILED_MATRIX = "tiled_matrix"
    PRODUCER_CONSUMER_PIPELINE = "producer_consumer_pipeline"


@dataclass(frozen=True)
class MathOperation:
    id: str
    category: OperationClass
    cost: ArithmeticCost | SpecialFunction | ReductionKind | None
    dtype: DataType
    inputs: tuple[str, ...]
    output: str
    reduction_axes: tuple[str, ...]


@dataclass(frozen=True)
class Computation:
    id: str
    result: str
    expression: str


@dataclass(frozen=True)
class AlgorithmStage:
    id: str
    reads: tuple[str, ...]
    after: tuple[str, ...]
    produces: tuple[str, ...]
    writes: tuple[str, ...]


@dataclass(frozen=True)
class AlgoIR(Header):
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    tensors: tuple[Tensor, ...]
    domains: tuple[tuple[str, str, str], ...]
    accumulators: tuple[Constant, ...]
    computations: tuple[Computation, ...]
    operations: tuple[MathOperation, ...]
    stages: tuple[AlgorithmStage, ...]
    traits: tuple[AlgorithmTrait, ...]
    requirements: tuple[str, ...]
