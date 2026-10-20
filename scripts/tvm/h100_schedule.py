"""Explicit Hopper Scheduled TIR used by the bounded GTSim frontend.

PTXNamespace performs the pinned TVM's instruction/operand selection. This is
a schedule for graph extraction, not a hardware-tested CUDA implementation.
The synchronization projection and resource policy are versioned together.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GemmSchedule:
    bm: int = 128
    bn: int = 256
    bk: int = 64

    @property
    def baseline(self):
        return (self.bm, self.bn, self.bk) == (128, 256, 64)

    @property
    def shared_bytes(self):
        return 4 * self.bk * (self.bm + self.bn) + 2 * self.bm * self.bn

    @property
    def register_bytes(self):
        return 4 * self.bm * self.bn + 12 * 32 * 32 * 4

    def validate(self, m, n, k, sms, hardware=None):
        if min(m, n, k, sms) <= 0 or max(m, n, k, sms) > 2**31 - 1:
            raise ValueError("dimensions/SM count must be positive int32 values")
        if self.bm not in (64, 128) or self.bn not in (128, 256) or self.bk not in (32, 64, 128):
            raise ValueError("supported tiles: BM={64,128}, BN={128,256}, BK={32,64,128}")
        if m % self.bm or n % self.bn or k % self.bk:
            raise ValueError("M, N and K must divide BM, BN and BK exactly")
        if (m // self.bm) * (n // self.bn) > 2**31 - 1:
            raise ValueError("output tile count overflows int32")
        # The ISA selection below uses complete m64n64k16 collectives, a
        # 128-column slice per warpgroup, and 64-column TMA B/C boxes.
        if self.bm % 64 or self.bn % 128 or self.bk % 16:
            raise ValueError("unsupported WGMMA/TMA layout combination")
        limits = hardware or {"shared_memory_per_cta_bytes": 227 * 1024,
                              "shared_memory_per_sm_bytes": 228 * 1024,
                              "register_file_per_sm_bytes": 65536 * 4,
                              "max_warps_per_sm": 64, "max_concurrent_blocks_per_sm": 4}
        if self.shared_bytes > min(limits["shared_memory_per_cta_bytes"], limits["shared_memory_per_sm_bytes"]):
            raise ValueError(f"shared-memory footprint {self.shared_bytes} exceeds capacity")
        if self.register_bytes > min(limits["register_file_per_sm_bytes"], 12 * 32 * 255 * 4):
            raise ValueError(f"register footprint {self.register_bytes} exceeds capacity")
        if limits["max_warps_per_sm"] < 12 or limits["max_concurrent_blocks_per_sm"] < 1:
            raise ValueError("hardware cannot admit a 12-warp CTA")


def scheduled_gemm(tvm, m, n, k, schedule=GemmSchedule(), sms=132):
    """Construct native TVM IR, with symbolic CTA/warp/tile access expressions.

    No Tile IR nodes, edges, simulator calls, or manual fixture are involved.
    The native operations are deliberately selected rather than inferred from
    a generic GEMM. Unknown schedules must fail in the companion lowerer.
    """
    from tvm.backend.cuda import op as cuda
    from tvm.backend.cuda.ptx import PTXNamespace

    schedule.validate(m, n, k, sms)
    t = tvm.tirx
    ptx = PTXNamespace()
    bm, bn, bk = schedule.bm, schedule.bn, schedule.bk
    rows, columns = m // bm, n // bn
    grid = min(sms, rows * columns)
    stages_k = k // bk
    a = t.decl_buffer((m, k), "float16", name="A")
    b = t.decl_buffer((k, n), "float16", name="B")
    c = t.decl_buffer((m, n), "float16", name="C")
    sa = t.decl_buffer((2, bm, bk), "float16", scope="shared", name="stage_a", data_alignment=128)
    sb = t.decl_buffer((2, bn // 64, bk, 64), "float16", scope="shared", name="stage_b", data_alignment=128)
    sc = t.decl_buffer((bn // 64, bm, 64), "float16", scope="shared", name="stage_c", data_alignment=128)
    full = t.decl_buffer((2,), "uint64", scope="shared", name="full")
    empty = t.decl_buffer((2,), "uint64", scope="shared", name="empty")
    # Accumulator layout: one 64x64 collective has 32 FP32 registers/thread.
    acc = t.decl_buffer((bm // 64, 2, 32), "float32", scope="local", name="acc")
    da = t.decl_buffer((1,), "uint64", scope="local", name="desc_a")
    db = t.decl_buffer((1,), "uint64", scope="local", name="desc_b")
    map_allocations = [t.tvm_stack_alloca("tensormap", 1) for _ in range(3)]
    maps = [t.Var(name, allocation.ty) for name, allocation in
            zip(("map_a", "map_b", "map_c"), map_allocations)]
    bx, tx, wave, kk, chunk, mh, nc, ks, ri = [t.Var(name, "int32") for name in
        ("bx", "tx", "wave", "kk", "chunk", "mh", "nc", "ks", "ri")]
    slot = kk % 2
    tile = wave * grid + bx
    group_start = tile // (rows * 8) * 8
    group_width = t.Min(8, columns - group_start)
    tile_m = (tile % (rows * 8)) // group_width
    tile_n = group_start + (tile % (rows * 8)) % group_width
    row, col = tile_m * bm, tile_n * bn
    wg = tx // 128
    full_generation = wave * ((stages_k + 1 - slot) // 2) + kk // 2

    def seq(items):
        return items[0] if len(items) == 1 else t.SeqStmt(items)

    def ev(call):
        return t.Evaluate(call)

    def loop(var, extent, body):
        return t.For(var, 0, extent, t.ForKind.SERIAL, body)

    def when(condition, body):
        return t.IfThenElse(condition, body, None)

    def arrive(buffer, index):
        return ev(ptx.mbarrier.arrive.shared.b64(t.address_of(buffer[index])))

    init = when(tx == 256, seq([
        loop(chunk, 2, seq([
            ev(ptx.mbarrier.init.shared.b64(t.address_of(full[chunk]), t.const(1, "uint32"))),
            ev(ptx.mbarrier.init.shared.b64(t.address_of(empty[chunk]), t.const(bn // 128, "uint32"))),
        ])),
    ]))
    # A native tensor map records global shape/strides and box extents. These
    # operands, not a BufferLoad heuristic, define each TMA access region.
    host = []
    for descriptor, allocation, tensor, box in zip(maps, map_allocations, (a, b, c), ((bm, bk), (bk, 64), (bm, 64))):
        host += [t.Bind(descriptor, allocation),
                 ev(t.tensormap_encode_tiled(descriptor, tensor.data, tensor.shape[1], tensor.shape[0],
                     tensor.shape[1] * 2, box[1], box[0], 1, 1, descriptor_dtype="float16", rank=2))]
    # The baseline's cross-wave buffer-zero rule is preserved literally. Tile
    # variants use the reference's general persistent buffer-consumer policy.
    need_empty = (kk >= 2) | ((wave > 0) & ((kk == 0) if schedule.baseline else (kk < 2)))
    load = seq([
        when(need_empty, ev(cuda.cuda_mbarrier_wait(t.address_of(empty[slot]), (full_generation - 1) % 2))),
        ev(ptx.mbarrier.arrive.expect_tx.shared.b64(t.address_of(full[slot]), t.const(2 * bk * (bm + bn), "uint32"))),
        ev(ptx["cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"](
            t.address_of(sa[slot, 0, 0]), maps[0], kk * bk, row, t.address_of(full[slot]))),
        loop(chunk, bn // 64, ev(ptx["cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"](
            t.address_of(sb[slot, chunk, 0, 0]), maps[1], col + chunk * 64, kk * bk, t.address_of(full[slot])))),
    ])
    mma = ptx["wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16"](
        *[acc[mh, nc, index] for index in range(32)], da[0], db[0], True, 1, 1, 0, 1)
    compute = seq([
        ev(cuda.cuda_mbarrier_wait(t.address_of(full[slot]), full_generation % 2)),
        ev(ptx.wgmma.fence.sync.aligned()),
        loop(mh, bm // 64, loop(nc, 2, loop(ks, bk // 16, seq([
            ev(cuda.cuda_wgmma_encode_matrix_descriptor(t.address_of(da[0]),
                t.address_of(sa[slot, mh * 64, ks * 16]), bk // 8, bk, 0)),
            ev(cuda.cuda_wgmma_encode_matrix_descriptor(t.address_of(db[0]),
                t.address_of(sb[slot, wg * 2 + nc, ks * 16, 0]), 8, 64, 0)),
            ev(mma),
        ])))),
        ev(ptx.wgmma.commit_group.sync.aligned()),
        ev(ptx.wgmma.wait_group.sync.aligned(0)),
        when(tx % 128 == 0, arrive(empty, slot)),
    ])
    # Physical register -> output coordinates follow TVM's pinned Hopper test:
    # each warp has 16 rows; groups of four lanes supply adjacent FP16 pairs.
    r = mh * 64 + (tx % 128 // 32) * 16 + tx % 32 // 4 + (ri % 4 // 2) * 8
    out_col = wg * 128 + nc * 64 + ri // 4 * 8 + tx % 4 * 2 + ri % 2
    stores = loop(mh, bm // 64, loop(nc, 2, loop(ri, 32,
        t.BufferStore(sc, t.Cast("float16", acc[mh, nc, ri]), [out_col // 64, r, out_col % 64]))))
    output = seq([
        ev(ptx.fence.proxy.async_.shared__cta()),
        loop(chunk, bn // 64, ev(ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
            maps[2], col + chunk * 64, row, t.address_of(sc[chunk, 0, 0])))),
        ev(ptx.cp.async_.bulk.commit_group()),
        ev(ptx.cp.async_.bulk.wait_group(0)),
    ])
    tile_body = seq([
        when(tx < bn, loop(mh, bm // 64, loop(nc, 2, loop(ri, 32,
            t.BufferStore(acc, t.const(0, "float32"), [mh, nc, ri]))))),
        loop(kk, stages_k, seq([when(tx == 256, load), when(tx < bn, compute)])),
        ev(cuda.cuda_cta_sync()),
        when(tx < bn, stores), ev(cuda.cuda_cta_sync()),
        when(tx == 256, output), ev(cuda.cuda_cta_sync()),
    ])
    body = seq([
                when(tx >= bn, ev(ptx.setmaxnreg.dec.sync.aligned.u32(32))),
                when(tx < bn, ev(ptx.setmaxnreg.inc.sync.aligned.u32(bm + 32))),
                init, ev(ptx.fence.proxy.async_.shared__cta()), ev(cuda.cuda_cta_sync()),
                loop(wave, (rows * columns + grid - 1) // grid, when(tile < rows * columns, tile_body))])
    body = t.For(tx, 0, 384, t.ForKind.THREAD_BINDING, body,
                 thread_binding=t.IterVar((0, 384), tx, t.IterVar.ThreadIndex, "threadIdx.x"))
    body = seq([t.AllocBuffer(buffer) for buffer in (sa, sb, sc, full, empty, acc, da, db)] + [body])
    body = t.For(bx, 0, grid, t.ForKind.THREAD_BINDING, body,
                 thread_binding=t.IterVar((0, grid), bx, t.IterVar.ThreadIndex, "blockIdx.x"))
    func = t.PrimFunc([a, b, c], seq(host + [body]))
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a", "max_num_threads": 1024, "thread_warp_size": 32})
    for key, value in {"global_symbol": "gemm", "target": target,
                       "gtsim.schedule": "h100_persistent_v1", "gtsim.sms": sms,
                       "gtsim.tensor_tiers": ["HBM", "HBM", "HBM"],
                       "gtsim.register_overhead_per_thread": 32}.items():
        func = func.with_attr(key, value)
    # The generator reuses convenient symbolic loop variables in disjoint
    # scopes. Let the pinned compiler give each binding its own SSA identity.
    func = t.transform.ConvertSSA()(tvm.IRModule({"gemm": func}))["gemm"]
    if not t.analysis.verify_ssa(func) or not t.analysis.verify_well_formed(func):
        raise ValueError("generated Hopper schedule failed TVM IR verification")
    return func
