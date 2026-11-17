"""Structured GEMM tiling for the small-M and grouped Qwen schedules.

The pass deliberately operates on FlowIR objects.  It does not rewrite source
text or create alternate schedule files, so the compiler, partitioner, and
native selector all see the same immutable program with one changed geometry.
"""

from dataclasses import asdict, dataclass, replace
import re

from parser import ast as A


BM_CANDIDATES = (16, 32)
BN_CANDIDATES = (64, 128, 256)
BK_CANDIDATES = (16, 32, 64, 128, 256)
STAGES_CANDIDATES = (2,)
SPLIT_K_CANDIDATES = (1,)


@dataclass(frozen=True)
class GemmTiling:
    bm: int = 16
    bn: int = 64
    bk: int = 16
    stages: int = 2
    split_k: int = 1

    def validate(self):
        if self.bm not in BM_CANDIDATES:
            raise ValueError(f"unsupported GEMM BM: {self.bm}")
        if self.bn not in BN_CANDIDATES:
            raise ValueError(f"unsupported GEMM BN: {self.bn}")
        if self.bk not in BK_CANDIDATES:
            raise ValueError(f"unsupported GEMM BK: {self.bk}")
        if self.stages not in STAGES_CANDIDATES:
            raise ValueError(f"unsupported GEMM pipeline stages: {self.stages}")
        if self.split_k not in SPLIT_K_CANDIDATES:
            raise ValueError(f"unsupported GEMM split-K: {self.split_k}")
        return self


@dataclass(frozen=True)
class GemmRegionSummary:
    region: str
    schedule: str
    tile_shape: tuple[int, int, int]
    stages: int
    split_k: int
    cta_extent: int | str
    pipeline_stop: int | str
    load_a_shape: tuple[int | str, ...] | None
    load_b_shape: tuple[int | str, ...] | None
    mma_shape: tuple[int | str, ...] | None
    store_shape: tuple[int | str, ...] | None
    scratch_bytes: int
    shared_bytes: int
    register_bytes: int | None = None

    def json(self):
        return asdict(self)


def scratch_bytes(bm, bn, bk):
    """Return the aligned producer/accumulator scratch footprint.

    The accumulator term is intentionally part of the footprint.  The old
    textual search accounted only for the copy staging area and therefore
    under-reported resource use for wider tiles.
    """

    requested = 2 * bk * (bm + bn) + 4 * bm * bn + 512
    return ((requested + 511) // 512) * 512


def _rewrite_geometry(text, tiling):
    if not isinstance(text, str):
        return text
    bm, bn, bk = tiling.bm, tiling.bn, tiling.bk
    replacements = (
        ("ceildiv(%N,64)", f"ceildiv(%N,{bn})"),
        ("ceildiv(%M,16)", f"ceildiv(%M,{bm})"),
        ("%N divisible_by 64", f"%N divisible_by {bn}"),
        ("%K divisible_by 16", f"%K divisible_by {bk}"),
        ("ceildiv(%K,16)", f"ceildiv(%K,{bk})"),
        ("vceildiv(%RUN_LENGTHS,16)", f"vceildiv(%RUN_LENGTHS,{bm})"),
        ("%k*16", f"%k*{bk}"),
        ("%COL*64", f"%COL*{bn}"),
        ("%BLOCK*16", f"%BLOCK*{bm}"),
        ("min(16,%M-%BLOCK*", f"min({bm},%M-%BLOCK*"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _rewrite_index(index, tiling):
    return replace(index, expression=_rewrite_geometry(index.expression, tiling))


def _rewrite_shape(shape, kind, op_id, tiling):
    """Update the logical tile shape for one canonical GEMM operation."""

    bm, bn, bk = tiling.bm, tiling.bn, tiling.bk
    if op_id == "load_A":
        return ("VALID", bk)
    if op_id == "load_B":
        return (bk, bn)
    if kind == A.OpKind.MATRIX:
        return ("VALID", bn)
    if kind == A.OpKind.STORE:
        return ("VALID", bn)
    return shape


def _rewrite_region(region, kind, op_id, tiling):
    if region is None:
        return None
    shape = _rewrite_shape(region.shape, kind, op_id, tiling)
    return replace(
        region,
        offsets=tuple(_rewrite_index(index, tiling) for index in region.offsets),
        shape=shape,
    )


def _is_small_or_grouped_gemm(region):
    return (
        region.config.warpgroup == 1
        and region.config.matrix_shape == (16, 8, 16)
        and any(op.kind == A.OpKind.MATRIX for op in region.ops)
        and region.config.tile_shape in ((16, 64, 16),)
    )


def _rewrite_region_ir(region, tiling):
    bm, bn, bk = tiling.bm, tiling.bn, tiling.bk
    bindings = tuple(
        replace(binding, expression=_rewrite_geometry(binding.expression, tiling))
        for binding in region.static.bindings
    )
    predicates = tuple(
        replace(predicate, expression=_rewrite_geometry(predicate.expression, tiling))
        for predicate in region.static.predicates
    )
    arguments = tuple(
        replace(argument, expression=_rewrite_geometry(argument.expression, tiling))
        for argument in region.static.arguments
    )
    static = replace(
        region.static,
        bindings=bindings,
        predicates=predicates,
        constraints=tuple(
            _rewrite_geometry(constraint, tiling)
            for constraint in region.static.constraints
        ),
        arguments=arguments,
    )
    pipeline_loop = replace(
        region.domain.pipeline.loop,
        start=_rewrite_geometry(region.domain.pipeline.loop.start, tiling),
        stop=_rewrite_geometry(region.domain.pipeline.loop.stop, tiling),
        step=_rewrite_geometry(region.domain.pipeline.loop.step, tiling),
    )
    pipeline = replace(
        region.domain.pipeline,
        loop=pipeline_loop,
        depth=tiling.stages,
        stage=replace(
            region.domain.pipeline.stage,
            modulus=tiling.stages,
            iteration=replace(
                region.domain.pipeline.stage.iteration,
                index=_rewrite_index(
                    region.domain.pipeline.stage.iteration.index, tiling
                ),
            ),
            generation_counts_per_wave=tuple(
                _rewrite_geometry(value, tiling)
                for value in region.domain.pipeline.stage.generation_counts_per_wave
            ),
        ),
    )
    loops = tuple(
        replace(
            loop,
            start=_rewrite_geometry(loop.start, tiling),
            stop=_rewrite_geometry(loop.stop, tiling),
            step=_rewrite_geometry(loop.step, tiling),
        )
        for loop in region.domain.loops
    )
    ops = []
    for op in region.ops:
        op_region = _rewrite_region(op.region, op.kind, op.id, tiling)
        results = tuple(
            replace(
                result,
                shape=(tiling.bm, tiling.bn)
                if op.kind == A.OpKind.MATRIX
                else _rewrite_shape(result.shape, op.kind, op.id, tiling),
            )
            for result in op.results
        )
        ops.append(
            replace(
                op,
                region=op_region,
                results=results,
                work_shape=tuple(
                    _rewrite_geometry(value, tiling) for value in op.work_shape
                ),
                guard=_rewrite_geometry(op.guard, tiling),
                static_args=tuple(
                    replace(arg, expression=_rewrite_geometry(arg.expression, tiling))
                    for arg in op.static_args
                ),
            )
        )
    buffers = tuple(
        replace(buffer, bytes_per_slot=scratch_bytes(bm, bn, bk))
        if buffer.id == "scratch"
        else buffer
        for buffer in region.buffers
    )
    return replace(
        region,
        static=static,
        config=replace(region.config, tile_shape=(bm, bn, bk), stages=tiling.stages),
        domain=replace(region.domain, pipeline= pipeline, loops=loops),
        ops=tuple(ops),
        buffers=buffers,
    )


def apply_gemm_tiling(flow, bm, bn, bk, stages=2):
    """Apply one legal small-M/grouped GEMM geometry to a FlowIR.

    The function has the intentionally narrow public signature used by the
    experiment runner.  Split-K is fixed to one in this experiment.
    """

    tiling = GemmTiling(bm, bn, bk, stages, 1).validate()
    return replace(
        flow,
        regions=tuple(
            _rewrite_region_ir(region, tiling)
            if _is_small_or_grouped_gemm(region)
            else region
            for region in flow.regions
        ),
    )


def _operation_shape(region, op_id):
    for op in region.ops:
        if op.id == op_id:
            return op.region.shape if op.region is not None else None
    return None


def summarize_gemm_flow(flow, arch=None, split_k=1):
    """Create a JSON-ready structural summary before native expansion."""

    arch_by_symbol = {region.symbol: region for region in arch.regions} if arch else {}
    result = []
    for region in flow.regions:
        if not any(op.kind == A.OpKind.MATRIX for op in region.ops):
            continue
        arch_region = arch_by_symbol.get(region.symbol)
        scratch = next(
            (buffer.bytes_per_slot for buffer in region.buffers if buffer.id == "scratch"),
            0,
        )
        resources = getattr(arch_region, "resources", None)
        result.append(
            GemmRegionSummary(
                region.symbol,
                region.schedule_id,
                tuple(region.config.tile_shape),
                region.config.stages,
                split_k,
                next(
                    (loop.stop for loop in region.domain.loops if loop.id == "cta"),
                    "%CTAS",
                ),
                region.domain.pipeline.loop.stop,
                _operation_shape(region, "load_A"),
                _operation_shape(region, "load_B"),
                _operation_shape(region, "matmul"),
                _operation_shape(region, "store"),
                scratch,
                resources.shared_bytes if resources else scratch,
                resources.register_bytes if resources else None,
            )
        )
    return tuple(result)


def validate_gemm_flow(
    flow, resolver, shared_limit=227 * 1024, register_limit=65536 * 4
):
    """Reject geometry errors before a simulator process is launched."""

    summaries = summarize_gemm_flow(flow)
    if not summaries:
        raise ValueError("no small-M/grouped GEMM regions were found")
    for region, summary in zip(
        (r for r in flow.regions if any(o.kind == A.OpKind.MATRIX for o in r.ops)),
        summaries,
    ):
        n = next(
            (
                resolver.value(binding.expression)
                for binding in region.static.arguments
                if binding.id == "N"
            ),
            None,
        )
        k = next(
            (
                resolver.value(binding.expression)
                for binding in region.static.arguments
                if binding.id == "K"
            ),
            None,
        )
        if n is None or k is None:
            raise ValueError(f"{region.symbol}: unresolved GEMM dimensions")
        bm, bn, bk = summary.tile_shape
        if n % bn:
            raise ValueError(f"{region.symbol}: N={n} is not divisible by BN={bn}")
        if k % bk:
            raise ValueError(f"{region.symbol}: K={k} is not divisible by BK={bk}")
        if summary.scratch_bytes > shared_limit:
            raise ValueError(
                f"{region.symbol}: shared footprint {summary.scratch_bytes} exceeds "
                f"the {shared_limit}-byte CTA limit"
            )
        if summary.register_bytes is not None and summary.register_bytes > register_limit:
            raise ValueError(
                f"{region.symbol}: register footprint {summary.register_bytes} exceeds "
                f"the {register_limit}-byte SM limit"
            )
        for shape_name, shape in (
            ("load_A", summary.load_a_shape),
            ("load_B", summary.load_b_shape),
            ("mma", summary.mma_shape),
            ("store", summary.store_shape),
        ):
            if shape is None:
                raise ValueError(f"{region.symbol}: missing {shape_name} shape")
        for op in region.ops:
            if op.region is None:
                continue
            if any(
                isinstance(value, int) and value > 256 for value in op.region.shape
            ):
                raise ValueError(f"{region.symbol}: {op.id} exceeds the TMA box limit")
    return summaries


def summaries_json(summaries):
    return [item.json() for item in summaries]


def validate_native_gemm(arch):
    """Check the lowered Native IR invariants shared by every candidate."""

    for region in arch.regions:
        matrices = [pattern for pattern in region.ops if any(
            step.opcode == A.NativeOpcode.MMA for step in pattern.steps
        )]
        if not matrices:
            continue
        for pattern in matrices:
            for step in pattern.steps:
                if step.opcode in (A.NativeOpcode.TMA_LOAD, A.NativeOpcode.TMA_STORE):
                    if step.region is None:
                        raise ValueError(f"{region.symbol}: TMA step has no region")
                    for extent in step.region.shape:
                        if isinstance(extent, int) and not 0 < extent <= 256:
                            raise ValueError(
                                f"{region.symbol}: illegal TMA extent {extent}"
                            )
                if step.opcode == A.NativeOpcode.MMA:
                    if step.region is None or step.region.shape != (16, 8):
                        raise ValueError(
                            f"{region.symbol}: illegal MMA fragment shape "
                            f"{step.region.shape if step.region else None}"
                        )
                    k_loops = [loop for loop in step.loops if loop.id == "k_fragment"]
                    if len(k_loops) != 1 or not isinstance(k_loops[0].stop, int):
                        raise ValueError(
                            f"{region.symbol}: missing explicit K-fragment loop"
                        )
                    if k_loops[0].stop <= 0:
                        raise ValueError(f"{region.symbol}: empty K-fragment loop")
        if region.resources.shared_bytes > 227 * 1024:
            raise ValueError(f"{region.symbol}: native shared resource overflow")
        if region.resources.register_bytes > 65536 * 4:
            raise ValueError(f"{region.symbol}: native register resource overflow")
