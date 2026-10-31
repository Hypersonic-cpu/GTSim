from dataclasses import dataclass
from .common import Header, require


@dataclass(frozen=True)
class Algorithm(Header):
    family: str = "gemm"
    dimensions: tuple[int, ...] = (512, 512, 512)
    dtype: str = "fp16"
    head_dim: int = 128
    causal: bool = False
    numerical_execution: bool = False

    def verify(self):
        super().verify()
        require(self.family in ("gemm", "fa3"), self.symbol, "unknown algorithm template")
        require(self.dtype == "fp16", self.symbol, "reference profile only supports fp16")
        require(all(type(d) is int and d > 0 for d in self.dimensions), self.symbol, "invalid dimensions")
        require(not self.causal and not self.numerical_execution, self.symbol, "reference timing profile does not execute numerical or causal masking semantics")
        if self.family == "gemm":
            require(len(self.dimensions) == 3, self.symbol, "expected M,N,K")
            require(all(d % tile == 0 for d, tile in zip(self.dimensions, (128,256,64))), self.symbol, "reference GEMM requires complete BM128 BN256 BK64 tiles")
        else:
            require(len(self.dimensions) == 5, self.symbol, "expected batch,Hq,Hkv,Q,KV")
            b, hq, hkv, q, kv = self.dimensions
            require(hq % hkv == 0 and q % 128 == 0 and kv % 128 == 0 and self.head_dim == 128,
                    self.symbol, "reference FA3 requires GQA alignment, full Q/KV128 tiles and D128")

    @property
    def stages(self):
        return (("acquire_A_B", "matrix_accumulate", "output_stage") if self.family == "gemm" else
                ("Q_load", "KV_load", "QK", "online_softmax", "PV", "normalize", "store"))
