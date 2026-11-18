"""Instantiate packed attention schedules for the Hopper native backend."""

from dataclasses import replace

from parser import ast as A
from ..bind import bind_data_types
from ..static import Resolver


def specialize_attention(region, root, tensors, inputs, outputs):
    packed = region.origin == "fa3_gqa_decode_split"
    combine = region.origin == "fa3_gqa_combine"
    if not (packed or combine):
        return region

    arguments = {arg.id: root.value(arg.expression) for arg in region.static.arguments}
    # Guarded alternatives still need physical geometry for lowering.
    active = arguments.get("__program_guard", True)
    target_name = root.value(region.mapping.target) if region.mapping.target.startswith("%") else region.mapping.target
    if target_name != "sm90":
        raise ValueError("packed attention requires a modeled SM90 target")
    if packed:
        dtypes = {tensors[name].dtype for name in inputs}
        if len(dtypes) != 1 or not dtypes <= {A.DataType.BF16, A.DataType.FP16}:
            raise ValueError("SM90 packed attention needs homogeneous BF16 or FP16 Q/K/V")
        (dtype,) = dtypes
        b, hq, hkv, d, r, s, tile, stages = (
            arguments[k] for k in ("B", "HQ", "HKV", "D", "R", "S", "TILE", "STAGES")
        )
        contexts = arguments["CONTEXTS"]
        if active and (
            min(b, hq, hkv, d, r, tile, stages) <= 0 or hq % hkv
            or r > hq // hkv or len(contexts) != b
            or any(c < 0 for c in contexts) or not 2 <= s <= min(contexts) + 1
        ):
            raise ValueError("invalid GQA geometry or split count (packed Split-KV requires nonempty splits and S >= 2)")
        if active and (d % 8 or d > 256 or tile > 256):
            raise ValueError("SM90 packed attention TMA needs aligned D and box extents <= 256")
        mm, mn, mk = 64, 64, 16
        physical_m = (r + mm - 1) // mm * mm
        physical_d = (d + mk - 1) // mk * mk
        k_rows = (tile + mn - 1) // mn * mn
        v_rows = (tile + mk - 1) // mk * mk
        v_columns = (d + mn - 1) // mn * mn

        def aligned(size):
            return (size + 127) // 128 * 128

        # Permanent Q, separately reusable K and V stage slots, and
        # separately allocated synchronization state. The packed FP32
        # accumulators and probability fragments are accounted in registers.
        target = dict(
            WARPS=8, CONSUMER_WARPS=4, PRODUCER_WARP=4, WARPGROUP=4,
            MMA_M=mm, MMA_N=mn, MMA_K=mk, QKV_DTYPE=dtype.value,
            Q_BYTES=aligned(physical_m * physical_d * 2),
            K_BYTES=aligned(k_rows * physical_d * 2),
            V_BYTES=aligned(v_rows * v_columns * 2),
            BARRIER_BYTES=aligned(16 * (2 * stages + 1)),
        )
    else:
        d, s = arguments["D"], arguments["S"]
        dtype = tensors[outputs[0]].dtype
        if dtype not in (A.DataType.FP16, A.DataType.BF16, A.DataType.FP32):
            raise ValueError("unsupported packed attention output dtype")
        target = dict(OUTPUT_DTYPE=dtype.value, COMBINE_BYTES=4 * s * (d + 1))

    declarations = replace(region.static, parameters=(), constraints=())
    resolver = Resolver(declarations, {**root.values, **arguments, **target})
    bound = bind_data_types(region, resolver)
    config = bound.config
    config = replace(
        config,
        tile_shape=tuple(resolver.value(v) for v in config.tile_shape),
        matrix_shape=tuple(resolver.value(v) for v in config.matrix_shape),
        **{name: resolver.value(getattr(config, name)) for name in (
            "stages", "warps", "consumer_warps", "producer_warp", "warpgroup"
        )},
    )
    buffers = tuple(replace(b, slots=resolver.value(b.slots),
                            bytes_per_slot=resolver.value(b.bytes_per_slot))
                    for b in bound.buffers)
    pipe = bound.domain.pipeline
    stage = replace(pipe.stage, modulus=config.stages)
    ops = tuple(replace(op, slot=replace(op.slot, stage=stage)) if op.slot else op
                for op in bound.ops)
    if active and sum(b.slots * b.bytes_per_slot for b in buffers) > 227 * 1024:
        raise ValueError("packed attention exceeds the modeled SM90 shared memory limit")
    return replace(
        bound, config=config, buffers=buffers, ops=ops,
        mapping=replace(bound.mapping, target=target_name),
        static_ifs=bound.static_ifs if active else tuple(
            replace(guard, condition="false") if not guard.iterations else guard
            for guard in bound.static_ifs
        ),
        domain=replace(bound.domain, pipeline=replace(pipe, depth=config.stages, stage=stage)),
        static=replace(bound.static, bindings=bound.static.bindings + tuple(
            A.StaticBinding(name, repr(value)) for name, value in target.items()
        )),
    )
