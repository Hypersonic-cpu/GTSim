"""IR-05-only loop iteration, instruction emission and phased event resolution."""

from dataclasses import dataclass, replace
from ...ir.common import MatrixGroup
from ...ir.graph import ArchInstruction, HOPPER_OPS, CTA, KernelTensor
from ...ir.execution import ExecKernel
from ...ir.schedule import OpRole as R, Relation as L, Phase as P, LoopVar as V
from ...serialization.json_codec import digest


@dataclass(frozen=True)
class TileContext:
    cta: int
    sm: int
    tile: int
    row: int
    column: int = 0
    batch: int = 0
    head: int = 0
    kv_head: int = 0
    matrix_group: MatrixGroup | None = None
    weight_tensor: int = 1

    def bindings(self, k=0):
        return {
            V.CTA: self.cta,
            V.TILE: self.tile,
            V.ROW: self.row,
            V.COLUMN: self.column,
            V.BATCH: self.batch,
            V.HEAD: self.head,
            V.KV_HEAD: self.kv_head,
            V.K: k,
        }


@dataclass(frozen=True)
class FutureEvent:
    key: tuple[int, int, int]


class InstructionEmitter:
    def __init__(self, native):
        self.native = native
        self.patterns = {pattern.source.role: pattern for pattern in native.ops}
        self.ops = []
        self.pending = []
        self.events = {}
        self.produced = {}

    def emit(
        self,
        role,
        context,
        warp,
        name,
        tensor,
        tile,
        groups,
        dtype="fp16",
        ld=0,
        step=0,
        value_dependencies=(),
    ):
        pattern = self.patterns[role].steps[step]
        complete = list(value_dependencies)
        issue = []
        future = []
        for dependency in pattern.dependencies:
            if dependency.relation == L.EXPLICIT:
                key = (context.cta, dependency.source)
                if key not in self.produced:
                    raise ValueError(
                        "explicit native dependency has no emitted producer"
                    )
                target = issue if dependency.phase == P.ISSUE else complete
                target.extend(self.produced[key])
                continue
            if dependency.relation not in groups:
                raise ValueError(
                    f"{role}: missing event group {dependency.relation.value}"
                )
            target = issue if dependency.phase == P.ISSUE else complete
            for value in groups[dependency.relation]:
                if isinstance(value, FutureEvent):
                    future.append((dependency.phase, value))
                else:
                    target.append(value)
        if self.native.schedule_id in (
            "small_m_mma",
            "grouped_small_m_mma",
        ) and pattern.opcode in (
            "cp.dram2sram_tma",
            "cp.sram2dram_tma",
        ):
            spec = self.native.tensors[tensor]
            row, column, rows, columns = tile
            tile = (
                row,
                column,
                min(rows, spec.shape[0] - row),
                min(columns, spec.shape[1] - column),
            )
            if min(tile[2:]) <= 0:
                raise ValueError("empty masked native memory region")
            dtype = spec.dtype
        nid = len(self.ops)
        self.ops.append(
            ArchInstruction(
                nid,
                name,
                pattern.opcode,
                HOPPER_OPS[pattern.opcode],
                context.cta,
                context.cta * self.native.profile.warps + warp,
                tensor,
                tuple(tile),
                dtype,
                ld,
                tuple(issue),
                tuple(complete),
            )
        )
        key = (context.cta, role)
        # All fragments of a producer must complete before a conservative
        # explicit consumer edge. Retaining the IDs is bounded by the eager DAG.
        self.produced.setdefault(key, []).append(nid)
        self.pending.extend((nid, phase, value.key) for phase, value in future)
        return nid

    def register(self, key, nid):
        self.events[key] = nid

    def finish(self):
        for nid, phase, key in self.pending:
            if key not in self.events:
                raise ValueError(f"unresolved native event {key}")
            op = self.ops[nid]
            field = "after_issue" if phase == P.ISSUE else "after_complete"
            self.ops[nid] = replace(
                op, **{field: getattr(op, field) + (self.events[key],)}
            )
        return tuple(self.ops)


def contexts(native):
    p = native.profile
    d = native.domain
    if d.matrix_groups:
        _, n, _ = d.dimensions
        ids = {t.id: i for i, t in enumerate(native.tensors)}
        identity = 0
        for group in d.matrix_groups:
            for row in range((len(group.input_rows) + p.bm - 1) // p.bm):
                for column in range((n + p.bn - 1) // p.bn):
                    yield TileContext(
                        identity,
                        identity % d.num_sms,
                        identity,
                        row,
                        column,
                        matrix_group=group,
                        weight_tensor=ids[group.weight],
                    )
                    identity += 1
        return
    if native.family == "gemm":
        m, n, k = d.dimensions
        order = [
            (row, column)
            for start in range(0, (n + p.bn - 1) // p.bn, 8)
            for row in range((m + p.bm - 1) // p.bm)
            for column in range(start, min((n + p.bn - 1) // p.bn, start + 8))
        ]
        for cta in range(d.cta_count):
            for tile in range(cta, d.tile_count, d.cta_count):
                row, column = order[tile]
                yield TileContext(cta, cta, tile, row, column)
    else:
        batch, hq, hkv, q, kv = d.dimensions
        blocks = q // p.q_tile
        for cta in range(d.cta_count):
            head_index, row = divmod(cta, blocks)
            b, h = divmod(head_index, hq)
            yield TileContext(
                cta,
                cta % d.num_sms,
                cta,
                row,
                batch=b,
                head=h,
                kv_head=h // (hq // hkv),
            )


def copy_regions(pattern, region, bindings):
    offsets = region.offsets_at(bindings)
    shape = region.shape
    axis = pattern.chunk_axis
    if axis < 0:
        return (offsets + shape,)
    result = []
    for offset in range(0, shape[axis], pattern.chunk_extent):
        position = list(offsets)
        extent = list(shape)
        position[axis] += offset
        extent[axis] = min(pattern.chunk_extent, shape[axis] - offset)
        result.append(tuple(position + extent))
    return tuple(result)


def tensors(native):
    if native.family == "gemm":
        return tuple(
            KernelTensor(
                i,
                t.id,
                t.shape,
                (t.shape[1], 1),
                dtype=t.dtype,
                preloaded=t.read_only if native.domain.matrix_groups else i == 1,
                read_only=t.read_only if native.domain.matrix_groups else i == 1,
            )
            for i, t in enumerate(native.tensors)
        )
    batch, hq, hkv, q, kv = native.domain.dimensions
    p = native.profile
    result = []
    for b in range(batch):
        for h in range(hq):
            for offset, letter in ((0, "Q"), (3, "O")):
                result.append(
                    KernelTensor(
                        (b * hq + h) * 4 + offset,
                        f"{letter}_b{b}_h{h}",
                        (q, p.head_dim),
                        (p.head_dim, 1),
                        preloaded=letter == "Q",
                        read_only=letter == "Q",
                    )
                )
        for h in range(hkv):
            for offset, letter in ((1, "K"), (2, "V")):
                result.append(
                    KernelTensor(
                        (b * hkv + h) * 4 + offset,
                        f"{letter}_b{b}_h{h}",
                        (p.head_dim, kv),
                        (kv, 1),
                    )
                )
    return tuple(result)


def kernel(native, emitter, ctas):
    ts = tensors(native)
    outputs = (
        (2,)
        if native.family == "gemm"
        else tuple(
            (b * native.domain.dimensions[1] + h) * 4 + 3
            for b in range(native.domain.dimensions[0])
            for h in range(native.domain.dimensions[1])
        )
    )
    result = ExecKernel(
        symbol=native.symbol,
        source_hash=digest(native),
        sync_semantics_version=native.sync_semantics_version,
        tensors=ts,
        ctas=tuple(ctas),
        ops=emitter.finish(),
        outputs=outputs,
    )
    result.verify()
    return result


def cta(context, native):
    r = native.resources
    return CTA(context.cta, context.sm, r.warps, r.shared_bytes, r.register_bytes)


def consumer_warps(profile):
    return range(profile.consumer_warps)


def instruction_iterations(step, warp=None):
    """Interpret the native loop dimensions without algorithm-specific nesting."""
    from itertools import product

    dimensions = [
        loop for loop in step.loops if warp is None or loop.variable != V.WARP
    ]
    for coordinates in product(*(range(loop.count) for loop in dimensions)):
        bindings = dict(zip((loop.variable for loop in dimensions), coordinates))
        if warp is not None:
            bindings[V.WARP] = warp
        yield bindings


def buffer_consumers(pattern, iteration, rounds, previous_wave=()):
    """Legalize the reference empty-slot event, including wave parity."""
    depth = pattern.source.slot.stage.modulus
    if iteration >= depth:
        return rounds[iteration - depth]
    if previous_wave and (iteration == 0 or len(previous_wave) % depth == 0):
        prior = max(
            (j for j in range(len(previous_wave)) if j % depth == iteration % depth),
            default=-1,
        )
        if prior >= 0:
            return previous_wave[prior]
    return []
