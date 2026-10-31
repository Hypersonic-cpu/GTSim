"""Shared, operation-level Hopper instruction selection and synchronization."""

from ...ir.arch import NativeStep, NativePattern, SharedLayout, RegisterLayout
from ...ir.profiles import resources
from ...ir.schedule import (
    OpKind as K,
    OpRole as R,
    Relation as L,
    Phase as P,
    Dependency,
    BufferRole as B,
    TileLoop,
    LoopVar as V,
)

SOFTMAX = {
    R.MAX0: ("LocalMax_r0", "local_max"),
    R.MAX1: ("LocalMax_r1", "local_max"),
    R.SHFL_MAX0: ("ShflMax_r0", "shfl_max"),
    R.SHFL_MAX1: ("ShflMax_r1", "shfl_max"),
    R.REDUCE_MAX0: ("ReduceMax_r0", "reduce_max"),
    R.REDUCE_MAX1: ("ReduceMax_r1", "reduce_max"),
    R.MERGE_MAX: ("MergeMax", "merge_max"),
    R.SCALE: ("Scale", "scores_scale"),
    R.EXP0: ("Exp2_r0", "exp2"),
    R.EXP1: ("Exp2_r1", "exp2"),
    R.SUM0: ("LocalSum_r0", "local_sum"),
    R.SUM1: ("LocalSum_r1", "local_sum"),
    R.SHFL_SUM0: ("ShflSum_r0", "shfl_sum"),
    R.SHFL_SUM1: ("ShflSum_r1", "shfl_sum"),
    R.REDUCE_SUM0: ("ReduceSum_r0", "reduce_sum"),
    R.REDUCE_SUM1: ("ReduceSum_r1", "reduce_sum"),
    R.MERGE_SUM: ("MergeSum", "merge_sum"),
    R.LOGSUM: ("Logsum", "logsum"),
    R.SCALE_O: ("AccOScale", "acc_o_scale"),
    R.CAST: ("Cast", "acc_s_cast"),
}


def softmax_shape(role, profile):
    fragment_rows = profile.mma_m // profile.warpgroup
    rows = fragment_rows // 2
    if role in (R.MAX0, R.MAX1, R.SUM0, R.SUM1):
        return (rows, profile.kv_tile)
    if role in (R.EXP0, R.EXP1):
        return (fragment_rows, profile.kv_tile // 2)
    if role == R.SCALE:
        return (rows, fragment_rows)
    if role == R.LOGSUM:
        return (rows, rows)
    if role == R.SCALE_O:
        return (fragment_rows * 2, profile.head_dim // 2)
    if role == R.CAST:
        return (rows, profile.kv_tile // 2)
    return (rows, profile.warpgroup)


def select(op, profile):
    deps = op.dependencies
    if op.kind == K.COPY:
        axis, extent = (
            (-1, 0)
            if op.role == R.LOAD_A
            else ((1, profile.mma_n) if op.role == R.LOAD_B else (0, profile.mma_m))
        )
        steps = (
            NativeStep("cp.dram2sram_tma", deps, chunk_axis=axis, chunk_extent=extent),
        )
    elif op.kind == K.MATMUL:
        loops = (TileLoop(V.WARP, profile.consumer_warps),)
        if profile.family == "gemm":
            loops += (
                TileLoop(V.M_HALF, profile.bm // profile.mma_m),
                TileLoop(V.N_CHUNK, 2),
                TileLoop(V.K_FRAGMENT, profile.bk // profile.mma_k),
            )
        else:
            loops += (
                TileLoop(
                    V.K_FRAGMENT,
                    profile.kv_tile
                    * profile.head_dim
                    // (profile.mma_n * profile.mma_k),
                ),
            )
        steps = (
            NativeStep(
                "wgmma.m64n64k16",
                deps,
                (profile.mma_m, profile.mma_n, profile.mma_k),
                profile.warpgroup,
                loops=loops,
            ),
        )
    elif op.kind == K.STORE:
        staging = tuple(d for d in deps if d.relation != L.PREVIOUS_CHUNK)
        final = (Dependency(R.STORE, P.COMPLETE, L.STAGING),) + tuple(
            d for d in deps if d.relation == L.PREVIOUS_CHUNK
        )
        steps = (
            NativeStep("st.reg2sram", staging, (16, 16)),
            NativeStep(
                "cp.sram2dram_tma",
                final,
                chunk_axis=1 if profile.family == "gemm" else 0,
                chunk_extent=64,
            ),
        )
    elif op.role == R.NORMALIZE:
        steps = (NativeStep("acc_o_div", deps, (32, profile.head_dim // 2)),)
    elif op.kind == K.ACQUIRE:
        steps = (NativeStep("mbarrier.wait.empty", deps),)
    elif op.kind == K.RELEASE:
        steps = (NativeStep("mbarrier.arrive.empty", deps),)
    elif op.role in SOFTMAX:
        _, opcode = SOFTMAX[op.role]
        shape = softmax_shape(op.role, profile)
        steps = (NativeStep(opcode, deps, shape),)
    else:
        raise ValueError(f"{op.id}: unsupported typed Flow operation {op.kind}")
    return NativePattern(op, steps)


def layouts(profile):
    if profile.family == "gemm":
        return (
            SharedLayout(
                B.AB,
                profile.stages,
                "row_major",
                profile.io_bytes * profile.bk * (profile.bm + profile.bn),
            ),
            SharedLayout(
                B.OUTPUT, 1, "row_major", profile.io_bytes * profile.bm * profile.bn
            ),
        )
    return (
        SharedLayout(
            B.Q, 1, "row_major", profile.io_bytes * profile.q_tile * profile.head_dim
        ),
        SharedLayout(
            B.K,
            profile.stages,
            "row_major",
            profile.io_bytes * profile.kv_tile * profile.head_dim,
        ),
        SharedLayout(
            B.V,
            profile.stages,
            "row_major",
            profile.io_bytes * profile.kv_tile * profile.head_dim,
        ),
        SharedLayout(
            B.OUTPUT,
            1,
            "reuse_input_after_PV",
            profile.io_bytes * profile.q_tile * profile.head_dim,
        ),
    )


def register_layout(profile):
    return RegisterLayout(
        "fp32",
        (profile.mma_m // profile.warpgroup, profile.mma_n),
        profile.consumer_warps,
        profile.base_registers_per_thread,
        resources(profile).register_bytes,
    )
