"""Hopper reference matrix/copy primitives, expressed as structured Native IR."""

from parser import ast as A


def index(expression):
    return A.IndexExpr(str(expression))


def dimension(value):
    return index("%" + value if isinstance(value, str) else value)


def loop(name, extent):
    return A.TileLoop(name, name, 0, extent, 1)


def region(source, row, column, shape):
    return A.SymbolicRegion(source.tensor, (index(row), index(column)), shape)


def _select(op, flow, operand_positions):
    p = flow.config
    bm, bn, bk = p.tile_shape
    mm, mn, mk = p.matrix_shape
    if (mm, mn, mk, p.warpgroup) != (64, 64, 16, 4):
        raise ValueError("unsupported Hopper reference matrix geometry")
    warp_loops = (loop("executor", p.consumer_warps), loop("m_fragment", bm // mm))
    steps = []
    if op.kind == A.OpKind.COPY:
        source = op.region
        r, c = (v.expression for v in source.offsets)
        columns = source.shape[1]
        chunked = operand_positions[op.id] == 1
        loops = (loop("chunk", f"ceildiv({columns},{mn})"),) if chunked else ()
        width = mn if chunked else columns
        steps.append(
            A.NativeStep(
                A.NativeOpcode.TMA_LOAD,
                A.ExecutionResource.LOAD,
                index(p.producer_warp),
                loops,
                region(
                    source,
                    r,
                    f"({c}) + %chunk*{mn}" if chunked else c,
                    (source.shape[0], width),
                ),
                op.results[0].dtype,
                dimension(
                    next(t for t in flow.tensors if t.id == source.tensor).shape[1]
                ),
                op.dependencies,
                (),
                (),
                "true",
                op.static_args,
            )
        )
    elif op.kind == A.OpKind.MATRIX:
        r, c = (v.expression for v in op.region.offsets)
        columns_per_group = bn // (p.consumer_warps // p.warpgroup)
        loops = warp_loops + (
            loop("n_fragment", columns_per_group // mn),
            loop("k_fragment", bk // mk),
        )
        shape = (mm // p.warpgroup, mn)
        steps.append(
            A.NativeStep(
                A.NativeOpcode.MATRIX,
                A.ExecutionResource.MATRIX,
                index("%executor"),
                loops,
                region(
                    op.region,
                    f"({r}) + %m_fragment*{mm} + (%executor mod {p.warpgroup})*{shape[0]}",
                    f"({c}) + floordiv(%executor,{p.warpgroup})*{columns_per_group} + %n_fragment*{mn}",
                    shape,
                ),
                op.results[0].dtype,
                None,
                op.dependencies,
                (),
                (),
                "true",
                op.static_args,
            )
        )
    elif op.kind == A.OpKind.STORE:
        r, c = (v.expression for v in op.region.offsets)
        columns_per_group = bn // (p.consumer_warps // p.warpgroup)
        previous_chunk = {
            rule.id
            for rule in flow.dependency_rules
            if rule.selection == A.Selection.ALL
            and any(
                d.label == rule.id and d.phase == A.Phase.ISSUE for d in op.dependencies
            )
        }
        staging = tuple(d for d in op.dependencies if d.label not in previous_chunk)
        final = tuple(d for d in op.dependencies if d.label in previous_chunk)
        steps.append(
            A.NativeStep(
                A.NativeOpcode.SHARED_STORE,
                A.ExecutionResource.STORE,
                index("%executor"),
                warp_loops + (loop("n_fragment", columns_per_group // 16),),
                region(
                    op.region,
                    f"({r}) + %m_fragment*{mm} + (%executor mod {p.warpgroup})*16",
                    f"({c}) + floordiv(%executor,{p.warpgroup})*{columns_per_group} + %n_fragment*16",
                    (16, 16),
                ),
                op.cast,
                None,
                staging,
                (),
                (),
                "true",
                op.static_args,
            )
        )
        tensor = next(t for t in flow.tensors if t.id == op.region.tensor)
        steps.append(
            A.NativeStep(
                A.NativeOpcode.TMA_STORE,
                A.ExecutionResource.STORE,
                index(p.producer_warp),
                (loop("chunk", f"ceildiv({bn},{mn})"),),
                region(op.region, r, f"({c}) + %chunk*{mn}", (bm, mn)),
                op.cast,
                dimension(tensor.shape[1]),
                final,
                (0,),
                (),
                "true",
                op.static_args,
            )
        )
    elif op.kind not in (A.OpKind.ACQUIRE, A.OpKind.RELEASE):
        raise ValueError(f"no Hopper instruction pattern for {op.kind.value}")
    return A.NativePattern(op.id, op.scope, tuple(steps), op.dependencies)


def select(op, flow, positions):
    from dataclasses import replace

    if op.kind == A.OpKind.MATRIX and op.region is None:
        return A.NativePattern(
            op.id,
            op.scope,
            (
                A.NativeStep(
                    A.NativeOpcode.MATRIX,
                    A.ExecutionResource.MATRIX,
                    index("%executor"),
                    op.iterations,
                    None,
                    op.results[0].dtype,
                    None,
                    op.dependencies,
                    (),
                    op.work_shape,
                    op.guard,
                    op.static_args,
                ),
            ),
            op.dependencies,
        )
    if "bulk_copy_exact" in op.capabilities:
        from .vector import select_vector

        return select_vector(op, flow)
    if op.kind in (A.OpKind.ARITHMETIC, A.OpKind.SPECIAL_FUNCTION, A.OpKind.REDUCTION):
        from .vector import select_vector

        return select_vector(op, flow)
    if not any(item.kind == A.OpKind.MATRIX for item in flow.ops) and op.kind in (
        A.OpKind.COPY,
        A.OpKind.STORE,
    ):
        from .vector import select_vector

        return select_vector(op, flow)
    if flow.config.warpgroup == 1:
        from .skinny import select_skinny

        return select_skinny(op, flow, positions)
    pattern = _select(op, flow, positions)
    return replace(
        pattern,
        steps=tuple(
            replace(step, loops=op.iterations + step.loops, guard=op.guard)
            for step in pattern.steps
        ),
    )


def lower_region(flow):
    p = flow.config
    matrices = [op for op in flow.ops if op.kind == A.OpKind.MATRIX]
    positions = {
        operand.producer: position
        for op in matrices
        for position, operand in enumerate(op.operands[:2])
    }
    shared = sum(buffer.slots * buffer.bytes_per_slot for buffer in flow.buffers)
    base_registers = 32
    bm, bn, _ = p.tile_shape
    registers = p.warps * 32 * base_registers * 4 + bm * bn * 4
    cta_variable = flow.domain.loops[0].variable
    from dataclasses import replace

    patterns = tuple(select(op, flow, positions) for op in flow.ops)
    tensors = {tensor.id: tensor for tensor in flow.tensors}
    patterns = tuple(
        replace(
            pattern,
            steps=tuple(
                replace(
                    step, leading_dim=dimension(tensors[step.region.tensor].shape[1])
                )
                if step.opcode in (A.NativeOpcode.TMA_LOAD, A.NativeOpcode.TMA_STORE)
                and step.leading_dim is None
                else step
                for step in pattern.steps
            ),
        )
        for pattern in patterns
    )
    return A.ArchRegion(
        flow.symbol,
        flow.static,
        flow.mapping.target,
        flow.domain,
        flow.tensors,
        A.ResourceFootprint(p.warps, shared, registers),
        tuple(
            A.SharedLayout(b.id, b.slots, "row_major", b.bytes_per_slot)
            for b in flow.buffers
        ),
        A.RegisterLayout(
            A.DataType.FP32,
            (p.matrix_shape[0] // p.warpgroup, p.matrix_shape[1]),
            p.consumer_warps,
            base_registers,
            registers,
        ),
        A.AsyncProtocol(p.stages, 1, 0, A.Phase.COMPLETE),
        patterns,
        flow.dependency_rules,
        flow.static_ifs,
        cta_variable,
        index(
            "%"
            + cta_variable
            + (
                " mod %NUM_SMS"
                if flow.mapping.placement == A.Placement.ROUND_ROBIN
                else ""
            )
        ),
        flow.after,
        flow.origin,
    )
