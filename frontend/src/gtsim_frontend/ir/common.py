from dataclasses import dataclass


class VerificationError(ValueError):
    """An invalid IR contract, with an object location in the message."""


def require(condition, location, reason):
    if not condition:
        raise VerificationError(f"{location}: {reason}")


@dataclass(frozen=True)
class Tensor:
    id: str
    shape: tuple[int, ...]
    dtype: str = "fp16"
    tier: str = "HBM"
    read_only: bool = False

    def verify(self):
        require(bool(self.id), "tensor", "missing identity")
        require(
            bool(self.shape) and all(type(d) is int and d > 0 for d in self.shape),
            self.id,
            "dimensions must be concrete positive integers",
        )
        require(
            self.dtype in ("fp16", "bf16", "fp32", "int32"),
            self.id,
            "unsupported dtype",
        )
        require(self.tier == "HBM", self.id, "reference profile requires HBM")


@dataclass(frozen=True)
class TensorRegion:
    tensor: str
    offsets: tuple[int, ...]
    shape: tuple[int, ...]

    def verify(self, tensors):
        require(self.tensor in tensors, self.tensor, "unbound tensor")
        t = tensors[self.tensor]
        require(
            len(self.offsets) == len(self.shape) == len(t.shape),
            self.tensor,
            "region rank mismatch",
        )
        require(
            all(
                type(o) is int and type(s) is int and o >= 0 and s > 0 and o + s <= d
                for o, s, d in zip(self.offsets, self.shape, t.shape)
            ),
            self.tensor,
            "region outside tensor",
        )


@dataclass(frozen=True)
class Event:
    id: str
    producer: str
    phase: str

    def verify(self):
        require(
            self.phase in ("issue", "ready", "complete", "visible", "released"),
            self.id,
            "unknown event phase",
        )


@dataclass(frozen=True)
class SlotHandle:
    buffer: str
    slot: int
    generation: int


@dataclass(frozen=True)
class Header:
    symbol: str
    source_hash: str = ""
    schema_version: int = 1

    def verify(self):
        require(bool(self.symbol), "header", "empty symbol")
        require(
            self.schema_version
            == (
                2
                if type(self).__name__
                in (
                    "FlowKernel",
                    "ArchKernel",
                    "Algorithm",
                    "ExecKernel",
                )
                else 3
                if type(self).__name__
                in (
                    "DecodeProgram",
                    "DecodeAlgorithm",
                    "DecodeFlowKernel",
                    "DecodeNativeKernel",
                )
                else 1
            ),
            self.symbol,
            "unsupported schema version",
        )
        require(
            self.source_hash == ""
            or (
                len(self.source_hash) == 64
                and all(c in "0123456789abcdef" for c in self.source_hash)
            ),
            self.symbol,
            "invalid SHA256",
        )


@dataclass(frozen=True)
class TileValue:
    id: str
    producer: str
    shape: tuple[int, ...]
    dtype: str = "fp32"

    def verify(self):
        require(
            bool(self.id) and bool(self.producer),
            "value",
            "missing identity or producer",
        )
        require(
            bool(self.shape) and all(type(d) is int and d > 0 for d in self.shape),
            self.id,
            "invalid value shape",
        )
        require(self.dtype in ("fp16", "fp32"), self.id, "unsupported value dtype")
