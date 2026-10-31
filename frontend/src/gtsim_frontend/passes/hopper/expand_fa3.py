"""Expand the structured FA3 native loop body only at the IR-05 boundary."""

from dataclasses import dataclass, field
from ...ir.schedule import OpRole as R, Relation as L, LoopVar as V
from .patterns import SOFTMAX
from .expansion import (
    InstructionEmitter,
    FutureEvent,
    contexts,
    copy_regions,
    consumer_warps,
    cta,
    kernel,
    instruction_iterations,
    buffer_consumers,
)


@dataclass
class AttentionState:
    last_qk: list = field(default_factory=list)
    last_pv: list = field(default_factory=list)
    logsum: list = field(default_factory=list)
    casts: list = field(default_factory=list)
    scales: list = field(default_factory=list)
    qk_rounds: list = field(default_factory=list)
    pv_rounds: list = field(default_factory=list)


def copy_operand(native, emitter, context, role, k=0, consumers=()):
    p = native.profile
    d = native.domain
    pattern = emitter.patterns[role]
    b = context.batch
    h = context.head
    if role == R.LOAD_Q:
        tensor = (b * d.dimensions[1] + h) * 4
        leading = p.head_dim
        groups = {}
        labels = [f"TB{context.cta}_Q_Load{i}_b{b}_h{h}" for i in range(2)]
    else:
        letter = "K" if role == R.LOAD_K else "V"
        tensor = (b * d.dimensions[2] + context.kv_head) * 4 + (
            1 if role == R.LOAD_K else 2
        )
        leading = d.dimensions[4]
        groups = {L.BUFFER_REUSE: list(consumers)}
        labels = [f"TB{context.cta}_{letter}_Load{i}_k{k}" for i in range(2)]
    regions = copy_regions(pattern.steps[0], pattern.source.region, context.bindings(k))
    return [
        emitter.emit(
            role, context, p.producer_warp, label, tensor, region, groups, ld=leading
        )
        for label, region in zip(labels, regions)
    ]


def matrix(native, emitter, context, k, warp, role, loads, state, extra=None):
    p = native.profile
    g = warp // p.warpgroup
    gw = context.cta * p.warps + warp
    label = "SS" if role == R.QK else "RS"
    last = state.last_qk if role == R.QK else state.last_pv
    result = []
    fragment = emitter.patterns[role].steps[0].fragment_shape
    for binding in instruction_iterations(emitter.patterns[role].steps[0], warp):
        step = binding[V.K_FRAGMENT]
        if role == R.QK:
            groups = {
                L.LOAD_Q: loads[0],
                L.LOAD_K: loads[1],
                L.PREVIOUS_FRAGMENT: [last[warp]] if last[warp] is not None else [],
                L.PREVIOUS_SCORE: [state.logsum[warp]] if k > 0 and step == 0 else [],
                L.PREVIOUS_CAST: [state.casts[warp]] if k > 0 and step == 0 else [],
            }
        else:
            groups = {
                L.LOAD_V: loads,
                L.CAST: [state.casts[warp]],
                L.SCALE_O: [state.scales[warp]],
                L.LOGSUM: [state.logsum[warp]],
                L.PREVIOUS_FRAGMENT: [last[warp]] if last[warp] is not None else [],
            }
        nid = emitter.emit(
            role,
            context,
            warp,
            f"TB{context.cta}_WGMMA_{label}_g{g}_w{gw}_k{k}_s{step}",
            3 if role == R.QK else 4,
            (g * p.mma_m, 0, 16, p.mma_n),
            groups,
            dtype="fp32",
        )
        result.append(nid)
        last[warp] = nid
    if role == R.QK:
        emitter.register((context.cta, k, warp), last[warp])
    return result


def softmax(native, emitter, context, k, warp, state):
    p = native.profile
    g = warp // p.warpgroup
    gw = context.cta * p.warps + warp
    values = {R.QK: state.last_qk[warp]}
    for pattern in native.ops:
        role = pattern.source.role
        if role not in SOFTMAX:
            continue
        groups = {}
        for dep in pattern.steps[0].dependencies:
            if dep.relation == L.PREVIOUS_PV:
                group = [state.last_pv[warp]] if k > 0 else []
            elif dep.relation == L.NEXT_QK:
                group = (
                    [FutureEvent((context.cta, k + 1, warp))]
                    if k + 1 < native.domain.pipeline.loop.count
                    else []
                )
            else:
                group = [values[dep.source]]
            groups[dep.relation] = group
        label, _ = SOFTMAX[role]
        shape = pattern.steps[0].fragment_shape
        values[role] = emitter.emit(
            role,
            context,
            warp,
            f"TB{context.cta}_Softmax_{label}_g{g}_w{gw}_k{k}",
            0,
            (g * p.mma_m, 0, *shape),
            groups,
        )
    state.logsum[warp] = values[R.LOGSUM]
    state.scales[warp] = values[R.SCALE_O]
    state.casts[warp] = values[R.CAST]


def epilogue(native, emitter, context, state):
    p = native.profile
    divisions = []
    for warp in consumer_warps(p):
        g = warp // p.warpgroup
        gw = context.cta * p.warps + warp
        divisions.append(
            emitter.emit(
                R.NORMALIZE,
                context,
                warp,
                f"TB{context.cta}_AccO_Div_g{g}_w{gw}",
                0,
                (g * p.mma_m, 0, 32, p.head_dim // 2),
                {
                    L.WARP_TAIL: [state.last_pv[warp]],
                    L.LOGSUM: [state.logsum[warp]],
                    L.FINAL_ROUND: state.pv_rounds[-1],
                },
            )
        )
    stores = []
    for warp in consumer_warps(p):
        g = warp // p.warpgroup
        gw = context.cta * p.warps + warp
        for col in range(p.head_dim // 16):
            stores.append(
                emitter.emit(
                    R.STORE,
                    context,
                    warp,
                    f"TB{context.cta}_StSram_g{g}_w{gw}_c{col}",
                    5,
                    (
                        context.row * p.q_tile
                        + g * p.mma_m
                        + (warp % p.warpgroup) * 16,
                        col * 16,
                        16,
                        16,
                    ),
                    {L.NORMALIZATION: divisions},
                    ld=p.head_dim,
                )
            )
    pattern = emitter.patterns[R.STORE]
    for group, region in enumerate(
        copy_regions(pattern.steps[1], pattern.source.region, context.bindings())
    ):
        emitter.emit(
            R.STORE,
            context,
            p.producer_warp,
            f"TB{context.cta}_Out_Store_g{group}",
            (context.batch * native.domain.dimensions[1] + context.head) * 4 + 3,
            region,
            {L.STAGING: stores},
            ld=p.head_dim,
            step=1,
        )


def expand(native):
    emitter = InstructionEmitter(native)
    ctas = []
    p = native.profile
    for context in contexts(native):
        ctas.append(cta(context, native))
        state = AttentionState(*([None] * p.consumer_warps for _ in range(5)))
        qids = copy_operand(native, emitter, context, R.LOAD_Q)
        for k in range(native.domain.pipeline.loop.count):
            kid = copy_operand(
                native,
                emitter,
                context,
                R.LOAD_K,
                k,
                buffer_consumers(emitter.patterns[R.LOAD_K], k, state.qk_rounds),
            )
            vid = copy_operand(
                native,
                emitter,
                context,
                R.LOAD_V,
                k,
                buffer_consumers(emitter.patterns[R.LOAD_V], k, state.pv_rounds),
            )
            qk = []
            pv = []
            for warp in consumer_warps(p):
                qk.extend(
                    matrix(native, emitter, context, k, warp, R.QK, (qids, kid), state)
                )
                softmax(native, emitter, context, k, warp, state)
                pv.extend(matrix(native, emitter, context, k, warp, R.PV, vid, state))
            state.qk_rounds.append(qk)
            state.pv_rounds.append(pv)
        epilogue(native, emitter, context, state)
    return kernel(native, emitter, ctas)
