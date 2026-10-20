"""Strict schedule recognition and native Hopper TIR -> version 1 Tile IR.

Only the complete h100_persistent_v1 pattern is accepted. Matching the whole
program makes synchronization, descriptor shapes/layout, offsets and instruction
selection preconditions, rather than hints that a generic GEMM may ignore.
"""

from functools import cache

from h100_schedule import GemmSchedule, scheduled_gemm
from tile_ir import Operation, TileIR


def evaluate(tvm, expr, context):
    """Evaluate the bounded integer access expressions at the extraction point."""
    t = tvm.tirx
    if isinstance(expr, t.IntImm):
        return int(expr)
    if isinstance(expr, t.Var):
        if str(expr.name) not in context:
            raise ValueError(f"unbound symbolic index {expr.name}")
        return context[str(expr.name)]
    binary = {t.Add: lambda a, b: a + b, t.Sub: lambda a, b: a - b,
              t.Mul: lambda a, b: a * b, t.FloorDiv: lambda a, b: a // b,
              t.FloorMod: lambda a, b: a % b, t.Min: min, t.Max: max,
              t.LT: lambda a, b: a < b}
    for kind, function in binary.items():
        if isinstance(expr, kind):
            return function(evaluate(tvm, expr.a, context), evaluate(tvm, expr.b, context))
    raise ValueError(f"unsupported symbolic access expression: {type(expr).__name__}")


def statements(tvm, stmt):
    """Walk statements only; expressions remain available on their owners."""
    t = tvm.tirx
    yield stmt
    if isinstance(stmt, t.SeqStmt):
        for child in stmt.seq:
            yield from statements(tvm, child)
    elif isinstance(stmt, (t.For, t.AttrStmt)):
        yield from statements(tvm, stmt.body)
    elif isinstance(stmt, t.IfThenElse):
        yield from statements(tvm, stmt.then_case)
        if stmt.else_case is not None:
            yield from statements(tvm, stmt.else_case)


def recognize(tvm, func):
    t = tvm.tirx
    if not isinstance(func, t.PrimFunc) or len(func.params) != 3:
        raise ValueError("expected a three-buffer Hopper PrimFunc")
    if not func.attrs or func.attrs.get("gtsim.schedule") != "h100_persistent_v1":
        raise ValueError("unsupported TVM schedule; expected explicit h100_persistent_v1")
    tiers = list(func.attrs.get("gtsim.tensor_tiers", []))
    if len(tiers) != 3 or any(tier not in ("HBM", "HBF") for tier in tiers):
        raise ValueError("explicit HBM/HBF tensor devices are required")
    for buffer, tier in zip(func.params, tiers):
        if not t.is_buffer_var(buffer) or len(buffer.shape) != 2 or str(buffer.dtype) != "float16":
            raise ValueError("only three rank-2 FP16 buffers are supported")
        if buffer.scope() != "global." + tier.lower() or buffer.strides or int(buffer.elem_offset) != 0:
            raise ValueError("compact buffers must declare matching global.hbm/global.hbf address spaces at offset zero")
    a, b, c = func.params
    m, k = [int(x) for x in a.shape]
    if tuple(int(x) for x in b.shape)[0] != k:
        raise ValueError("inconsistent reduction dimensions")
    n = int(b.shape[1])
    if tuple(int(x) for x in c.shape) != (m, n):
        raise ValueError("inconsistent output dimensions")
    nodes = list(statements(tvm, func.body))
    allocations = {str(node.buffer.name): node.buffer for node in nodes if isinstance(node, t.AllocBuffer)}
    if not all(name in allocations for name in ("stage_a", "stage_b", "stage_c", "acc")):
        raise ValueError("missing explicit shared stages/accumulator allocation")
    sa, sb = allocations["stage_a"], allocations["stage_b"]
    if len(sa.shape) != 3 or len(sb.shape) != 4:
        raise ValueError("unsupported shared-stage ranks")
    schedule = GemmSchedule(int(sa.shape[1]), int(sb.shape[1]) * 64, int(sa.shape[2]))
    sms = int(func.attrs["gtsim.sms"])
    schedule.validate(m, n, k, sms)
    expected = scheduled_gemm(tvm, m, n, k, schedule, sms, tiers)
    try:
        tvm.ir.assert_structural_equal(func, expected)
    except ValueError as error:
        raise ValueError("unsupported or altered h100_persistent_v1 TIR: instruction, region, binding, layout, or synchronization mismatch") from error
    calls = [node.value for node in nodes if isinstance(node, t.Evaluate) and isinstance(node.value, tvm.ir.Call)]
    native = sorted({call.op.name for call in calls if isinstance(call.op, tvm.ir.Op)})
    required = {"tirx.ptx.cp_async_bulk_tensor_g2s_cluster", "tirx.ptx.cp_async_bulk_tensor_s2g",
                "tirx.ptx.wgmma_f16_ss", "tirx.cuda.mbarrier_wait"}
    if not required.issubset(native):
        raise ValueError(f"missing selected native Hopper instructions: {required - set(native)}")
    bindings = {str(node.thread_binding.thread_tag): int(node.extent) for node in nodes
                if isinstance(node, t.For) and node.thread_binding is not None}
    tensor_maps = {str(call.args[0].name): call for call in calls if call.op.name == "tirx.tensormap_encode_tiled"}
    loads = {str(call.args[1].name): call for call in calls if call.op.name == "tirx.ptx.cp_async_bulk_tensor_g2s_cluster"}
    stores = [call for call in calls if call.op.name == "tirx.ptx.cp_async_bulk_tensor_s2g"]
    shared_store = next(node for node in nodes if isinstance(node, t.BufferStore) and str(node.buffer.name) == "stage_c")
    extents = {str(node.loop_var.name): int(node.extent) for node in nodes if isinstance(node, t.For)}
    wave_loop = next(node for node in nodes if isinstance(node, t.For) and str(node.loop_var.name) == "wave")
    native_pattern = {"wave_guard": wave_loop.body.condition, "a": loads["map_a"], "b": loads["map_b"], "c": stores[0],
                      "maps": tensor_maps, "store": shared_store, "extents": extents}
    register_calls = [call for call in calls if call.op.name == "tirx.ptx.setmaxnreg"]
    budgets = {call.args[1].value: int(call.args[0]) for call in register_calls}
    if set(budgets) != {"inc", "dec"}:
        raise ValueError("missing explicit warpgroup register redistribution")
    register_bytes = 4 * (schedule.bn * budgets["inc"] + (bindings["threadIdx.x"] - schedule.bn) * budgets["dec"])
    if register_bytes != schedule.register_bytes:
        raise ValueError("native setmaxnreg requirements differ from resource metadata")
    return m, n, k, schedule, sms, {"native_operations": native, "thread_bindings": bindings,
        "shared_shapes": {key: [int(x) for x in allocations[key].shape] for key in ("stage_a", "stage_b", "stage_c")},
        "schedule": "h100_persistent_v1", "tensor_tiers": tiers,
        "register_budgets_per_thread": budgets,
        "synchronization": "two-stage full/empty mbarriers; WGMMA commit/wait; CTA output join; TMA store issue order",
        "tvm_verified": ["ConvertSSA", "verify_ssa", "verify_well_formed"]}, native_pattern


def lower_reference_gemm(tvm, func):
    m, n, k, schedule, sms, evidence, native = recognize(tvm, func)
    bm, bn, bk = schedule.bm, schedule.bn, schedule.bk
    # Native tensor-map shapes/strides, bindings and every instruction operand
    # have been matched above. Expand that recognized pattern, not arbitrary
    # BufferLoads or a call to the manual/reference graph implementation.
    ir = TileIR()
    for ident, (name, shape) in enumerate((("A", (m, k)), ("B", (k, n)), ("C", (m, n)))):
        ir.tensor(ident, name, shape, (shape[1], 1), tier=evidence["tensor_tiers"][ident])
    def access(tensor, context):
        # TMA 2-D coordinates and box dimensions are in inner/outer order.
        call, descriptor = native[tensor], native["maps"]["map_" + tensor]
        offset = 1 if tensor == "c" else 2
        column = evaluate(tvm, call.args[offset], context)
        row = evaluate(tvm, call.args[offset + 1], context)
        return row, column, int(descriptor.args[6]), int(descriptor.args[5])

    for cta in range(evidence["thread_bindings"]["blockIdx.x"]):
        ir.cta(cta, cta, schedule.shared_bytes, schedule.register_bytes)
        previous_wave_stores = []
        for wave in range(native["extents"]["wave"]):
            context = {"bx": cta, "wave": wave, "kk": 0, "chunk": 0, "tx": 0, "mh": 0, "nc": 0, "ri": 0}
            if not evaluate(tvm, native["wave_guard"], context):
                continue
            tile = evaluate(tvm, native["wave_guard"].a, context)
            prefix = f"TB{cta}_Tile{tile}"

            # The matched native output indices are independent of kk. Reuse
            # each per-tile coordinate evaluation across reduction iterations.
            @cache
            def output_region(warp, half, n_chunk, register=0, rows=16, columns=64):
                local = dict(context, tx=warp * 32, mh=half, nc=n_chunk, ri=register)
                physical_chunk, row, column = [evaluate(tvm, index, local) for index in native["store"].indices]
                global_row, global_column, _, _ = access("c", dict(local, chunk=physical_chunk))
                return global_row + row, global_column + column, rows, columns

            def emit(suffix, operation, warp, tensor, region, complete=(), issue=(), dtype="fp16"):
                return ir.add(Operation(prefix + suffix, operation, cta, warp, tensor, region,
                                        dtype, list(complete), list(issue)))

            stage_loads, stage_mma = [], []
            warp_last = [None] * (bn // 32)
            for stage in range(native["extents"]["kk"]):
                context["kk"] = stage
                # mbarrier.empty is signaled after every consumer group's wait0;
                # reuse waits for the complete stage from two iterations earlier.
                consumed = stage_mma[stage - 2] if stage >= 2 else []
                if stage == 0:
                    # The selected TIR ends each tile with bulk.wait_group(0)
                    # followed by a CTA barrier. Every next-wave thread waits
                    # for ALL prior TMA output completions, not merely issue.
                    consumed = previous_wave_stores
                previous_issues = stage_loads[-1] if stage else []
                loads = [emit(f"_TMA_LoadA_k{stage}", "cp.dram2sram_tma", 8, 0,
                              access("a", context), consumed, previous_issues)]
                loads.extend(emit(f"_TMA_LoadB_k{stage}_c{chunk}", "cp.dram2sram_tma", 8, 1,
                                  access("b", dict(context, chunk=chunk)), consumed, previous_issues)
                             for chunk in range(bn // 64))
                stage_loads.append(loads)
                fragments = []
                for warp in range(bn // 32):
                    for half in range(native["extents"]["mh"]):
                        for nc in range(native["extents"]["nc"]):
                            column_chunk = warp // 4 * 2 + nc
                            for inner_k in range(native["extents"]["ks"]):
                                predecessors = loads + ([] if warp_last[warp] is None else [warp_last[warp]])
                                ident = emit(f"_WGMMA_w{cta * 12 + warp}_k{stage}_mh{half}_n{column_chunk}_ks{inner_k}",
                                    "wgmma.m64n64k16", warp, 2,
                                    output_region(warp, half, nc),
                                    predecessors, dtype="fp32")
                                warp_last[warp] = ident
                                fragments.append(ident)
                stage_mma.append(fragments)
            # A collective m64n64k16 instruction becomes four 16x64 fragments in
            # canonical per-warp order. The CTA join gates every output store.
            staged = []
            for warp in range(bn // 32):
                for half in range(bm // 64):
                    for part in range(8):
                        staged.append(emit(f"_StSram_w{cta * 12 + warp}_mh{half}_n{part}", "st.reg2sram", warp, 2,
                            output_region(warp, half, part // 4, (part % 4) * 8, columns=16),
                            [warp_last[warp]] + stage_mma[-1]))
            prior = []
            output = []
            for chunk in range(bn // 64):
                ident = emit(f"_TMA_StoreC_c{chunk}", "cp.sram2dram_tma", 8, 2,
                             access("c", dict(context, chunk=chunk)), staged, prior)
                prior = [ident]
                output.append(ident)
            previous_wave_stores = output
    evidence.update({"dimensions": [m, n, k], "tile": [bm, bn, bk], "sms": sms,
                     "shared_bytes": schedule.shared_bytes, "register_bytes": schedule.register_bytes,
                     "nodes": len(ir.operations), "waves": native["extents"]["wave"],
                     "output_tiles": (m // bm) * (n // bn),
                     "cross_wave_sync": "bulk.wait_group(0) + CTA barrier: all prior output completions gate next tile"})
    work = sum(16 * 64 * 16 for op in ir.operations if op.op == "wgmma.m64n64k16")
    if work != m * n * k:
        raise ValueError("lowered WGMMA fragments do not preserve GEMM arithmetic work")
    output = [op.tile for op in ir.operations if op.op == "cp.sram2dram_tma"]
    regions = {(tile[0], tile[1]) for tile in output}
    expected = {(row, column) for row in range(0, m, bm) for column in range(0, n, 64)}
    if regions != expected or len(regions) != len(output) or any(tile[2:] != (bm, 64) for tile in output):
        raise ValueError("native TMA output regions do not cover C exactly once")
    evidence["multiply_accumulates"] = work
    return ir, evidence
