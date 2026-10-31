from ..ir.algorithm import Algorithm
from ..ir.common import Tensor
from ..ir.program import Program, Invocation, Command


def build(m=512, n=512, k=512, dtype="fp16"):
    algorithm = Algorithm(symbol="gemm", dimensions=(m,n,k), dtype=dtype)
    p = Program(symbol="gemm_once", tensors=(Tensor("A",(m,k),dtype,read_only=True),
                Tensor("B",(k,n),dtype,read_only=True), Tensor("C",(m,n),dtype)),
                launches=(Invocation("gemm0",algorithm,(("A","A"),("B","B"),("C","C"))),),
                commands=(Command("LaunchKernel","gemm0"),Command("StreamSynchronize")))
    p.verify()
    return p
