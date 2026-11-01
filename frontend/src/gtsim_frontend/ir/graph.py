from dataclasses import dataclass
import re
from .common import Header, Tensor, require

# Exact supported simulator operation families; no latency values in compiler IR.
HOPPER_OPS = {
    "cp.dram2sram_tma": "LD",
    "cp.sram2dram_tma": "ST",
    "st.reg2sram": "ST",
    "wgmma.m64n64k16": "Tensor",
    "mma.sync.m16n8k16": "Tensor",
    "local_max": "SIMD",
    "reduce_max": "SIMD",
    "merge_max": "SIMD",
    "local_sum": "SIMD",
    "reduce_sum": "SIMD",
    "merge_sum": "SIMD",
    "logsum": "SIMD",
    "acc_o_scale": "SIMD",
    "acc_s_cast": "SIMD",
    "shfl_max": "SHFL",
    "shfl_sum": "SHFL",
    "scores_scale": "SFU",
    "exp2": "SFU",
    "acc_o_div": "SFU",
    "simd.add": "SIMD",
    "simd.sub": "SIMD",
    "simd.mul": "SIMD",
    "simd.fma": "SIMD",
    "simd.max": "SIMD",
    "simd.min": "SIMD",
    "simd.cast": "SIMD",
    "simd.fill": "SIMD",
    "simd.identity": "SIMD",
    "sfu.exp": "SFU",
    "sfu.log": "SFU",
    "sfu.sqrt": "SFU",
    "sfu.rsqrt": "SFU",
    "sfu.divide": "SFU",
    "sfu.sin": "SFU",
    "sfu.cos": "SFU",
    "shfl": "SHFL",
}


@dataclass(frozen=True)
class KernelTensor:
    id: int
    name: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str = "fp16"
    tier: str = "HBM"
    preloaded: bool = False
    read_only: bool = False


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
    tile: tuple[int, int, int, int] = (0, 0, 1, 1)
    dtype: str = "fp16"
    leading_dim: int = 0
    after_issue: tuple[int, ...] = ()
    after_complete: tuple[int, ...] = ()


@dataclass(frozen=True)
class FlatKernel(Header):
    target: str = "sm90"
    sync_semantics_version: str = "fork-pingpong-v1"
    tensors: tuple[KernelTensor, ...] = ()
    ctas: tuple[CTA, ...] = ()
    ops: tuple[ArchInstruction, ...] = ()
    partial_outputs: tuple[int, ...] = ()
    outputs: tuple[int, ...] = ()

    def verify(self):
        super().verify()
        require(self.target == "sm90", self.symbol, "unsupported native target")
        require(
            len({t.id for t in self.tensors}) == len(self.tensors)
            and len({t.name for t in self.tensors}) == len(self.tensors),
            self.symbol,
            "duplicate kernel tensor",
        )
        for t in self.tensors:
            require(
                type(t.id) is int
                and t.id >= 0
                and bool(t.name)
                and t.dtype in ("fp16", "bf16", "fp32", "int32")
                and t.tier == "HBM"
                and len(t.shape) == len(t.strides) == 2
                and all(type(d) is int and d > 0 for d in t.shape)
                and t.strides == (t.shape[1], 1),
                self.symbol,
                "invalid kernel tensor metadata",
            )
        require(
            len({o.name for o in self.ops}) == len(self.ops),
            self.symbol,
            "duplicate instruction name",
        )
        collectives = {}
        for o in self.ops:
            if o.op == "wgmma.m64n64k16":
                key = (o.cta, o.warp // 4, re.sub(r"_w[0-9]+", "_w", o.name))
                collectives.setdefault(key, []).append(o.warp % 4)
        require(
            all(sorted(lanes) == [0, 1, 2, 3] for lanes in collectives.values()),
            self.symbol,
            "incomplete or duplicate four-warp WGMMA collective",
        )
        ctas = {c.id: c for c in self.ctas}
        require(len(ctas) == len(self.ctas), self.symbol, "duplicate CTA")
        warp_bases = {}
        sm_next_subpartition = {}
        next_warp_id = 0
        for c in self.ctas:
            require(
                c.id >= 0
                and c.sm >= 0
                and 1 <= c.warps <= 32
                and 0 < c.shared_bytes <= 228 * 1024
                and 0 < c.register_bytes <= 256 * 1024,
                self.symbol,
                "invalid reference CTA resources",
            )
            desired = sm_next_subpartition.get(c.sm, 0) % 4
            next_warp_id += (desired + 4 - next_warp_id % 4) % 4
            warp_bases[c.id] = next_warp_id
            next_warp_id += c.warps
            sm_next_subpartition[c.sm] = (
                sm_next_subpartition.get(c.sm, 0) + c.warps
            ) % 4
        require(
            tuple(o.id for o in self.ops) == tuple(range(len(self.ops))),
            self.symbol,
            "operation IDs must be dense",
        )
        require(
            len(set(self.outputs)) == len(self.outputs),
            self.symbol,
            "duplicate output tensor",
        )
        require(
            len(set(self.partial_outputs)) == len(self.partial_outputs)
            and set(self.partial_outputs) <= set(self.outputs),
            self.symbol,
            "invalid partial output contract",
        )
        tensors = {t.id: t for t in self.tensors}
        writes = {o.tensor for o in self.ops if o.op == "cp.sram2dram_tma"}
        require(
            writes == set(self.outputs),
            self.symbol,
            "missing or undeclared output store",
        )
        for output in self.outputs:
            require(output in tensors, self.symbol, "unknown output tensor")
            tensor = tensors[output]
            regions = [
                o.tile
                for o in self.ops
                if o.op == "cp.sram2dram_tma" and o.tensor == output
            ]
            require(
                0
                < sum(r[2] * r[3] for r in regions)
                <= tensor.shape[0] * tensor.shape[1]
                and (
                    output in self.partial_outputs
                    or sum(r[2] * r[3] for r in regions)
                    == tensor.shape[0] * tensor.shape[1]
                ),
                self.symbol,
                "incomplete native output coverage",
            )
            for i, a in enumerate(regions):
                for b in regions[i + 1 :]:
                    require(
                        not (
                            max(a[0], b[0]) < min(a[0] + a[2], b[0] + b[2])
                            and max(a[1], b[1]) < min(a[1] + a[3], b[1] + b[3])
                        ),
                        self.symbol,
                        "overlapping native output stores",
                    )
        edges = {o.id: set(o.after_issue + o.after_complete) for o in self.ops}
        children = {o.id: [] for o in self.ops}
        for o in self.ops:
            require(
                o.op in HOPPER_OPS and HOPPER_OPS[o.op] == o.pipeline,
                o.name,
                "unsupported native op or pipeline",
            )
            require(
                o.cta in ctas
                and warp_bases[o.cta] <= o.warp < warp_bases[o.cta] + ctas[o.cta].warps,
                o.name,
                "invalid CTA/warp",
            )
            require(
                o.dtype in ("fp16", "bf16", "fp32", "int32")
                and len(o.tile) == 4
                and min(o.tile[:2]) >= 0
                and min(o.tile[2:]) > 0,
                o.name,
                "invalid dtype or tile",
            )
            if o.op in ("cp.dram2sram_tma", "cp.sram2dram_tma"):
                require(o.tensor in tensors, o.name, "unbound global-memory tensor")
                tensor = tensors[o.tensor]
                row, col, rows, columns = o.tile
                require(
                    row + rows <= tensor.shape[0]
                    and col + columns <= tensor.shape[1]
                    and o.leading_dim == tensor.strides[0]
                    and o.dtype == tensor.dtype
                    and (o.op != "cp.sram2dram_tma" or not tensor.read_only),
                    o.name,
                    "global-memory type, region, or visibility mismatch",
                )
            for dep in edges[o.id]:
                require(
                    dep in edges and dep != o.id, o.name, "dangling or self dependency"
                )
                children[dep].append(o.id)
        ready = [n for n, e in edges.items() if not e]
        visited = 0
        while ready:
            n = ready.pop()
            visited += 1
            for child in children[n]:
                edges[child].discard(n)
                if not edges[child]:
                    ready.append(child)
        require(visited == len(self.ops), self.symbol, "native dependency cycle")
