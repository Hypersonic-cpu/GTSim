"""The single GEMM full-expansion boundary: structured native → IR-05."""

from dataclasses import dataclass, field
from ...ir.schedule import OpRole as R, Relation as L, LoopVar as V
from .expansion import (
    InstructionEmitter,
    contexts,
    copy_regions,
    consumer_warps,
    cta,
    kernel,
    instruction_iterations,
    buffer_consumers,
)


@dataclass
class WaveState:
    stores: list[int] = field(default_factory=list)
    loads: list[int] = field(default_factory=list)
    compute: list[list[int]] = field(default_factory=list)


def operand_loads(native, emitter, context, k, stage_loads, compute, previous):
    p = native.profile
    d = native.domain
    prefix = f"TB{context.cta}_Tile{context.tile}"
    patterns = emitter.patterns
    groups = {
        L.BUFFER_REUSE: buffer_consumers(
            patterns[R.LOAD_A], k, compute, previous.compute
        ),
        L.PREVIOUS_LOAD: stage_loads[-1] if stage_loads else previous.loads,
    }
    results = []
    for role, tensor, letter, leading in (
        (R.LOAD_A, 0, "A", d.dimensions[2]),
        (R.LOAD_B, 1, "B", d.dimensions[1]),
    ):
        pattern = patterns[role]
        regions = copy_regions(
            pattern.steps[0], pattern.source.region, context.bindings(k)
        )
        ids = []
        for chunk, region in enumerate(regions):
            suffix = f"_c{chunk}" if role == R.LOAD_B else ""
            ids.append(
                emitter.emit(
                    role,
                    context,
                    p.producer_warp,
                    f"{prefix}_TMA_Load{letter}_k{k}{suffix}",
                    tensor,
                    region,
                    groups,
                    ld=leading,
                )
            )
        results.append(ids)
    return results


def matrix_accumulate(native, emitter, context, k, loads, last, previous_stores):
    p = native.profile
    prefix = f"TB{context.cta}_Tile{context.tile}"
    result = []
    step_spec = emitter.patterns[R.MATMUL].steps[0]
    shape = step_spec.fragment_shape
    for binding in instruction_iterations(step_spec):
        warp = binding[V.WARP]
        half = binding[V.M_HALF]
        nc = binding[V.N_CHUNK]
        step = binding[V.K_FRAGMENT]
        chunk = (warp // step_spec.collective_warps) * 2 + nc
        groups = {
            L.LOAD_A: loads[0],
            L.LOAD_B: loads[1],
            L.PREVIOUS_FRAGMENT: [last[warp]] if last[warp] is not None else [],
            L.PREVIOUS_TILE_STORE: previous_stores if k == 0 else [],
        }
        rows_per_warp = shape[0] // step_spec.collective_warps
        region = (
            context.row * p.bm
            + half * shape[0]
            + (warp % step_spec.collective_warps) * rows_per_warp,
            context.column * p.bn + chunk * shape[1],
            rows_per_warp,
            shape[1],
        )
        nid = emitter.emit(
            R.MATMUL,
            context,
            warp,
            f"{prefix}_WGMMA_w{context.cta * p.warps + warp}_k{k}_mh{half}_n{chunk}_ks{step}",
            2,
            region,
            groups,
            dtype="fp32",
        )
        result.append(nid)
        last[warp] = nid
    return result


def epilogue(native, emitter, context, last, compute):
    p = native.profile
    prefix = f"TB{context.cta}_Tile{context.tile}"
    staging = []
    for warp in consumer_warps(p):
        for half in range(p.bm // p.mma_m):
            for col in range(p.bn // 2 // 16):
                region = (
                    context.row * p.bm + half * p.mma_m + (warp % p.warpgroup) * 16,
                    context.column * p.bn
                    + (warp // p.warpgroup) * (p.bn // 2)
                    + col * 16,
                    16,
                    16,
                )
                staging.append(
                    emitter.emit(
                        R.STORE,
                        context,
                        warp,
                        f"{prefix}_StSram_w{context.cta * p.warps + warp}_mh{half}_n{col}",
                        2,
                        region,
                        {L.WARP_TAIL: [last[warp]], L.FINAL_ROUND: compute[-1]},
                    )
                )
    pattern = emitter.patterns[R.STORE]
    stores = []
    for chunk, region in enumerate(
        copy_regions(pattern.steps[1], pattern.source.region, context.bindings())
    ):
        stores.append(
            emitter.emit(
                R.STORE,
                context,
                p.producer_warp,
                f"{prefix}_TMA_StoreC_c{chunk}",
                2,
                region,
                {L.STAGING: staging, L.PREVIOUS_CHUNK: stores[-1:]},
                ld=native.domain.dimensions[1],
                step=1,
            )
        )
    return stores


def expand(native):
    emitter = InstructionEmitter(native)
    ctas = []
    states = {}
    p = native.profile
    for context in contexts(native):
        if context.cta not in states:
            states[context.cta] = WaveState()
            ctas.append(cta(context, native))
        previous = states[context.cta]
        last = [None] * p.consumer_warps
        compute = []
        stage_loads = []
        for k in range(native.domain.pipeline.loop.count):
            loads = operand_loads(
                native, emitter, context, k, stage_loads, compute, previous
            )
            stage_loads.append(loads[0] + loads[1])
            compute.append(
                matrix_accumulate(
                    native, emitter, context, k, loads, last, previous.stores
                )
            )
        states[context.cta] = WaveState(
            epilogue(native, emitter, context, last, compute), stage_loads[-1], compute
        )
    return kernel(native, emitter, ctas)
