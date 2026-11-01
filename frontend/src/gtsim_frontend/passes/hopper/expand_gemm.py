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


def mapped_rows(context, role, regions):
    """Resolve typed logical group rows; coalesce only adjacent actual rows."""
    if context.matrix_group is None or role == R.LOAD_B:
        return regions
    mapping = (
        context.matrix_group.input_rows
        if role == R.LOAD_A
        else context.matrix_group.output_rows
    )
    result = []
    for row, column, rows, columns in regions:
        selected = mapping[row : row + rows]
        for actual in selected:
            if (
                result
                and result[-1][0] + result[-1][2] == actual
                and result[-1][1] == column
                and result[-1][3] == columns
            ):
                start, col, count, width = result[-1]
                result[-1] = (start, col, count + 1, width)
            else:
                result.append((actual, column, 1, columns))
    return tuple(result)


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
        (R.LOAD_B, context.weight_tensor, "B", d.dimensions[1]),
    ):
        pattern = patterns[role]
        regions = copy_regions(
            pattern.steps[0], pattern.source.region, context.bindings(k)
        )
        regions = mapped_rows(context, role, regions)
        ids = []
        for chunk, region in enumerate(regions):
            if native.schedule_id in ("small_m_mma", "grouped_small_m_mma"):
                shape = native.tensors[tensor].shape
                if region[0] >= shape[0] or region[1] >= shape[1]:
                    continue
            suffix = f"_c{chunk}" if role == R.LOAD_B or len(regions) > 1 else ""
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
    if native.schedule_id in ("small_m_mma", "grouped_small_m_mma"):
        return small_m_epilogue(native, emitter, context, last, compute)
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


def small_m_epilogue(native, emitter, context, last, compute):
    """MMA accumulator fragments use the same typed store patterns as WGMMA."""
    p = native.profile
    staging = []
    for binding in instruction_iterations(emitter.patterns[R.MATMUL].steps[0]):
        warp = binding[V.WARP]
        column = (warp * 2 + binding[V.N_CHUNK]) * p.mma_n
        staging.append(
            emitter.emit(
                R.STORE,
                context,
                warp,
                f"tile{context.tile}_stage{column}",
                2,
                (context.row * p.bm, context.column * p.bn + column, p.bm, p.mma_n),
                {L.WARP_TAIL: [last[warp]], L.FINAL_ROUND: compute[-1]},
                dtype=native.tensors[2].dtype,
            )
        )
    stores = []
    pattern = emitter.patterns[R.STORE]
    for store_index, region in enumerate(
        mapped_rows(
            context,
            R.STORE,
            copy_regions(pattern.steps[1], pattern.source.region, context.bindings()),
        )
    ):
        stores.append(
            emitter.emit(
                R.STORE,
                context,
                p.producer_warp,
                f"tile{context.tile}_store{store_index}",
                2,
                region,
                {L.STAGING: staging, L.PREVIOUS_CHUNK: stores[-1:]},
                dtype=native.tensors[2].dtype,
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
