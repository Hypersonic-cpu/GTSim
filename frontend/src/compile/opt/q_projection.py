"""FlowIR specialization for the isolated Q projection."""

from dataclasses import dataclass, replace

from parser import ast as A
from ..static import EMPTY, Resolver, bound_resolver


ORDERINGS = ("n_major", "split_major", "n_grouped_2", "n_grouped_4",
             "n_grouped_8", "staggered")
N, K, NUM_SMS = 8192, 4096, 132
BM = 64


class Infeasible(ValueError):
    """A specific static hardware or parameter constraint was violated."""


@dataclass(frozen=True)
class QSchedule:
    bn: int
    bk: int
    split_k: int
    ordering: str
    pf_bn: int = 0
    pf_bk: int = 0
    ahead: int = 0

    def validate(self):
        if self.bn not in (128, 256) or self.bk not in (32, 64, 128) or (
                self.split_k not in (1, 2, 4, 8) or self.ordering not in ORDERINGS):
            raise Infeasible("unsupported Q projection tiling or ordering")
        if N % self.bn or K % (self.split_k * self.bk):
            raise Infeasible("compute tile has incomplete N or split-K extent")
        if self.ahead not in (0, 1, 2, 4) or (self.ahead and not (
                0 < self.pf_bn <= N and 0 < self.pf_bk <= K)):
            raise Infeasible("invalid prefetch dimensions or lookahead")

    @property
    def tasks(self):
        return task_order(N // self.bn, self.split_k, self.ordering)

def task_order(num_n, split, ordering):
    if ordering == "n_major":
        return tuple((s, n) for n in range(num_n) for s in range(split))
    if ordering == "split_major":
        return tuple((s, n) for s in range(split) for n in range(num_n))
    if ordering == "staggered":
        return tuple(((s + n) % split, n) for n in range(num_n) for s in range(split))
    if ordering in ORDERINGS:
        group = int(ordering.rsplit("_", 1)[-1])
        return tuple((s, n) for start in range(0, num_n, group)
                     for s in range(split) for n in range(start, min(start + group, num_n)))
    raise Infeasible(f"unknown ordering {ordering}")


def specialize_q_projection(flow, schedule, hbf):
    schedule.validate()
    if schedule.ahead and not hbf:
        raise Infeasible("explicit prefetch requires HBF weights")
    tasks = schedule.tasks
    if len(set(tasks)) != (N // schedule.bn) * schedule.split_k:
        raise Infeasible("task ordering does not cover each output partial exactly once")
    out_width = 4 if schedule.split_k > 1 else 2
    input_stage = 2 * schedule.bk * (BM + schedule.bn)
    output_stage = BM * schedule.bn * out_width
    shared = 2 * input_stage + output_stage
    registers = 12 * 32 * 32 * 4 + BM * schedule.bn * 4
    consumer_registers = 32 + BM * schedule.bn // (8 * 32)
    if shared > 228 * 1024 or registers > 256 * 1024 or consumer_registers > 255:
        raise Infeasible(f"SM90 resource limit exceeded: SMEM={shared}, registers={registers} B")
    # C++ reference semantics stage the physically padded accumulator and issue
    # global TMA only for the single live row. A loads similarly fetch one row.
    gemm_symbol = "q_gemm_partial" if schedule.split_k > 1 else "q_gemm"
    active = {gemm_symbol, "q_split_reduce"} if schedule.split_k > 1 else {gemm_symbol}
    regions = []
    root = bound_resolver(flow.static, Resolver(EMPTY, {})).materialize_constants()
    for region in flow.regions:
        if region.symbol not in active:
            continue
        args = tuple(arg for arg in region.static.arguments if arg.id != "__program_guard")
        if region.symbol == gemm_symbol:
            values = {
                "split_order": tuple(s for s, _ in tasks),
                "n_order": tuple(n for _, n in tasks),
                "input_stage_bytes": input_stage,
                "output_stage_bytes": output_stage,
                "PfBN": schedule.pf_bn if schedule.ahead else 0,
                "PfBK": schedule.pf_bk if schedule.ahead else 0,
                "PfAheadDist": schedule.ahead,
                "CompOrdering": schedule.ordering,
                "GroupSize": int(schedule.ordering[-1]) if schedule.ordering.startswith(
                    "n_grouped_") else 0,
            }
            args += tuple(A.StaticBinding(name, repr(value)) for name, value in values.items())
        static = replace(region.static, arguments=args)
        resolver = Resolver(static, {**root.values, **{
            arg.id: root.value(arg.expression) for arg in args}})
        config = replace(region.config, tile_shape=tuple(
            resolver.value(value) for value in region.config.tile_shape))
        buffers = tuple(replace(buffer, bytes_per_slot=resolver.value(buffer.bytes_per_slot))
                        for buffer in region.buffers)
        tensors = region.tensors
        ops = region.ops
        if region.symbol == "q_gemm_partial":
            tensors = tuple(replace(t, dtype=A.DataType.FP32) if t.id == "Partial" else t
                            for t in tensors)
            ops = tuple(replace(op, cast=A.DataType.FP32,
                                math=replace(op.math, dtype=A.DataType.FP32))
                        if op.id == "store" else op for op in ops)
        regions.append(replace(
            region, static=static, config=config, buffers=buffers, tensors=tensors, ops=ops,
            static_ifs=tuple(item for item in region.static_ifs
                            if item.condition != "%__program_guard"),
            after=(regions[-1].symbol,) if regions else ()))
    tensors = tuple(
        replace(tensor, tier=A.MemorySpace.HBF) if tensor.id == "W_q" and hbf else tensor
        for tensor in flow.tensors if schedule.split_k > 1 or tensor.id != "Partial"
    )
    return replace(flow, tensors=tensors, regions=tuple(regions))
