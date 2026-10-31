from ..ir.algorithm import Algorithm
from ..ir.common import Tensor
from ..ir.program import Program, Invocation, Command


def build(batch=1, heads_q=1, heads_kv=1, seq_q=128, seq_kv=128, dtype="fp16", head_dim=128):
    algorithm = Algorithm(symbol="fa3",family="fa3",dimensions=(batch,heads_q,heads_kv,seq_q,seq_kv),dtype=dtype,head_dim=head_dim)
    tensors = (Tensor("Q",(batch,heads_q,seq_q,head_dim),dtype,read_only=True),
               Tensor("K",(batch,heads_kv,head_dim,seq_kv),dtype,read_only=True),
               Tensor("V",(batch,heads_kv,head_dim,seq_kv),dtype,read_only=True),
               Tensor("O",(batch,heads_q,seq_q,head_dim),dtype))
    p = Program(symbol="fa3_once",tensors=tensors,
                launches=(Invocation("fa30",algorithm,tuple((n,n) for n in ("Q","K","V","O"))),),
                commands=(Command("LaunchKernel","fa30"),Command("StreamSynchronize")))
    p.verify()
    return p
