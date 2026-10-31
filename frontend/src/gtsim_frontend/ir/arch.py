from dataclasses import dataclass
from .common import Header, Tensor, require

# Exact supported simulator operation families; no latency values in compiler IR.
HOPPER_OPS = {
    "cp.dram2sram_tma":"LD", "cp.sram2dram_tma":"ST", "st.reg2sram":"ST",
    "wgmma.m64n64k16":"Tensor", "local_max":"SIMD", "reduce_max":"SIMD",
    "merge_max":"SIMD", "local_sum":"SIMD", "reduce_sum":"SIMD", "merge_sum":"SIMD",
    "logsum":"SIMD", "acc_o_scale":"SIMD", "acc_s_cast":"SIMD",
    "shfl_max":"SHFL", "shfl_sum":"SHFL", "scores_scale":"SFU", "exp2":"SFU", "acc_o_div":"SFU",
}


@dataclass(frozen=True)
class CTA:
    id: int
    sm: int
    warps: int
    shared_bytes: int
    register_bytes: int


@dataclass(frozen=True)
class ArchInstruction:
    id: int
    name: str
    op: str
    pipeline: str
    cta: int
    warp: int
    tensor: int = 0
    tile: tuple[int, int, int, int] = (0,0,1,1)
    dtype: str = "fp16"
    leading_dim: int = 0
    after_issue: tuple[int, ...] = ()
    after_complete: tuple[int, ...] = ()


@dataclass(frozen=True)
class ArchKernel(Header):
    target: str = "sm90"
    sync_semantics_version: str = "fork-pingpong-v1"
    tensors: tuple[Tensor, ...] = ()
    ctas: tuple[CTA, ...] = ()
    ops: tuple[ArchInstruction, ...] = ()

    def verify(self):
        super().verify()
        require(self.target == "sm90",self.symbol,"unsupported native target")
        ctas={c.id:c for c in self.ctas}
        require(len(ctas)==len(self.ctas),self.symbol,"duplicate CTA")
        for c in self.ctas:
            require(c.id >= 0 and c.sm >= 0 and c.warps == 12 and 0 < c.shared_bytes <= 228*1024 and
                    0 < c.register_bytes <= 256*1024,self.symbol,"invalid reference CTA resources")
        require(tuple(o.id for o in self.ops)==tuple(range(len(self.ops))),self.symbol,"operation IDs must be dense")
        edges={o.id:set(o.after_issue+o.after_complete) for o in self.ops}
        children={o.id:[] for o in self.ops}
        for o in self.ops:
            require(o.op in HOPPER_OPS and HOPPER_OPS[o.op]==o.pipeline,o.name,"unsupported native op or pipeline")
            require(o.cta in ctas and o.cta*12 <= o.warp < o.cta*12+12,o.name,"invalid CTA/warp")
            require(o.dtype in ("fp16","fp32") and len(o.tile)==4 and min(o.tile[:2])>=0 and min(o.tile[2:])>0,o.name,"invalid dtype or tile")
            for dep in edges[o.id]:
                require(dep in edges and dep!=o.id,o.name,"dangling or self dependency")
                children[dep].append(o.id)
        ready=[n for n,e in edges.items() if not e]; visited=0
        while ready:
            n=ready.pop();visited+=1
            for child in children[n]:
                edges[child].discard(n)
                if not edges[child]:ready.append(child)
        require(visited==len(self.ops),self.symbol,"native dependency cycle")
