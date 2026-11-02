"""Render independent AST fields to assembly; no original source text is used."""

from . import ast as A


def enum(value):
    return value.value if hasattr(value, "value") else value


def names(values, prefix):
    return ", ".join(prefix + v for v in values)


def tensor(value, layout=False, symbolic=False):
    shape = "x".join(
        ("%" if symbolic and isinstance(d, str) else "") + str(d) for d in value.shape
    )
    return (
        f"tensor<{shape}x{enum(value.dtype)}"
        + (f", {enum(value.layout)}" if layout and value.layout else "")
        + ">"
    )


def index(value):
    return value.expression


def region(value):
    return f"@{value.tensor}[{', '.join(index(x) for x in value.offsets)}]<{'x'.join(map(str, value.shape))}>"


def operand(value):
    return (
        "%"
        + enum(value.producer)
        + (".prev" if value.iteration_delta < 0 else "")
        + (
            f"<{'zero' if value.initial == 0 else enum(value.initial)}>"
            if value.initial is not None
            else ""
        )
    )


def flowop(op):
    role = op.id
    deps = (
        " after["
        + ", ".join(
            f"@{enum(d.source)}.{enum(d.phase)}:{d.label}" for d in op.dependencies
        )
        + "]"
        if op.dependencies
        else ""
    )
    reads = "reads[" + ", ".join(operand(v) for v in op.operands) + "]"
    capabilities = (
        " requires<" + ", ".join(op.capabilities) + ">" if op.capabilities else ""
    )
    event = f"!{op.event} = " if op.event else ""
    slot = (
        f"%{op.slot.buffer}.s({op.slot.stage.iteration.index.expression} mod {op.slot.stage.modulus})"
        if op.slot
        else ""
    )
    if op.kind == A.OpKind.ACQUIRE:
        return f"flow.acquire @{op.id} scope<{enum(op.scope)}> slots[{slot}]{deps}"
    if op.kind == A.OpKind.RELEASE:
        return f"flow.release @{op.id} slots[{slot}]{deps}"
    if op.kind == A.OpKind.COPY:
        result = op.results[0]
        ty = "x".join(map(str, result.shape)) + "x" + enum(result.dtype)
        return (
            f"%{role}, "
            + event
            + f"flow.load_async @{op.id} {region(op.region)} to {op.destination} by %producer{deps}{capabilities} : tensor<{ty}>"
        )
    if op.kind == A.OpKind.MATRIX:
        result = op.results[0]
        ty = "x".join(map(str, result.shape)) + "x" + enum(result.dtype)
        return (
            f"%{role}, "
            + event
            + f"flow.matmul @{op.id} {reads} on_tile<{region(op.region)}> by %matrix{deps}{capabilities} : tensor<{ty}>"
        )
    return (
        event
        + f"flow.store_async @{op.id} {reads}"
        + (f" cast<{enum(op.cast)}>" if op.cast else "")
        + f" to {region(op.region)} by %producer{deps}{capabilities}"
    )


def prog(ir):
    lines = [
        f"prog.program @{ir.symbol} params("
        + ", ".join(f"%{n}: {enum(ty)}" for n, ty in ir.parameters)
        + ") {"
    ]
    for t in ir.tensors:
        lines.append(
            f"  prog.tensor @{t.id} : {tensor(t, symbolic=True)} storage<{enum(t.tier)}>"
            + (" external" if t.external else "")
            + (" readonly" if t.read_only else "")
        )
    launches = {v.id: v for v in ir.launches}
    for stream in ir.streams:
        lines.append(f"  prog.stream @{stream} ordered {{")
        for command in ir.commands:
            if command.kind == A.CommandKind.LAUNCH:
                inv = launches[command.target]
                lhs = names(inv.outputs, "%") + ", " + names(inv.events, "!")
                lines += [
                    f"    {lhs} = prog.launch @{inv.id}",
                    f"        callee<@{inv.callee}> implementation<@{inv.implementation}>",
                    f"        inputs[{names(inv.inputs, '@')} ] outputs[{names(inv.outputs, '@')} ]".replace(
                        " ]", "]"
                    ),
                    "        attrs{"
                    + ", ".join(f"{k}={v}" for k, v in inv.attributes)
                    + "}",
                ]
            else:
                lines.append(
                    f"    prog.synchronize @{command.target} after[{names(command.after, '!')}]"
                )
        lines.append("  }")
    lines += [
        f"  prog.return {names(ir.returns, '@')} after[{names(ir.return_after, '!')}]",
        "}",
    ]
    return "\n".join(lines) + "\n"


def algo(ir):
    outputs = {
        match[0].lstrip("%")
        for computation in ir.computations
        for result in (computation.result,)
        if (match := result.split("["))
    }
    accumulators = {constant.id for constant in ir.accumulators}
    inputs = [t for t in ir.tensors if t.id not in outputs and t.id not in accumulators]
    outs = [t for t in ir.tensors if t.id in outputs and t.id not in accumulators]
    sig = lambda ts: ", ".join(f"%{t.id}: {tensor(t)}" for t in ts)
    lines = [
        f"algo.kernel @{ir.symbol}({sig(inputs)})",
        f"    -> ({sig(outs)}) {{",
        "  algo.domain "
        + ", ".join(f"%{n} in [{lo}, {hi})" for n, lo, hi in ir.domains),
    ]
    tensors = {t.id: t for t in ir.tensors}
    lines += [
        f"  algo.accumulator @{constant.id} : {tensor(tensors[constant.id])} init<{'zero' if constant.value == 0 else enum(constant.value)}>"
        for constant in ir.accumulators
    ]
    for computation in ir.computations:
        name, result, meaning = (
            computation.id,
            computation.result,
            computation.expression,
        )
        lines += [
            f"  algo.compute @{name} result<{result}>",
            f"      semantics<{meaning}>",
        ]
    for op in ir.operations:
        lines.append(
            f"  algo.op @{op.id} category<{enum(op.category)}> cost<{enum(op.cost) if op.cost is not None else 'none'}> dtype<{enum(op.dtype)}> inputs[{names(op.inputs, '%')}] output[%{op.output}] reduction[{', '.join(op.reduction_axes)}]"
        )
    for stage in ir.stages:
        name, reads, after, produces, writes = (
            stage.id,
            stage.reads,
            stage.after,
            stage.produces,
            stage.writes,
        )
        line = f"  algo.stage @{name}"
        for label, values, prefix in [
            ("reads", reads, "%"),
            ("after", after, "@"),
            ("produces", produces, "%"),
            ("writes", writes, "%"),
        ]:
            if values:
                line += f" {label}[{names(values, prefix)}]"
        lines.append(line)
    lines += [
        "  algo.traits <" + ", ".join(enum(trait) for trait in ir.traits) + ">",
        "  algo.require <" + ", ".join(ir.requirements) + ">",
        "}",
    ]
    return "\n".join(lines) + "\n"


def flow(ir):
    lines = [
        f"flow.kernel @{ir.symbol} schedule<{ir.schedule_id}> target<{ir.mapping.target}> {{"
    ]
    lines += [f"  flow.symbol %{name} range[{lo}, {hi}]" for name, lo, hi in ir.symbols]
    lines += ["  flow.require <" + ", ".join(ir.requirements) + ">"]
    lines += [
        f"  flow.tensor @{t.id} : {tensor(t, layout=True, symbolic=True)}"
        for t in ir.tensors
    ]
    lines += [
        "  flow.domain "
        + "\n      ".join(f"%{n} = {expr}" for n, expr in ir.expressions)
    ]
    p = ir.config
    lines += [
        f"  flow.schedule_config tile({'x'.join(map(str, p.tile_shape))}) stages({p.stages})",
        f"      warps({p.warps}) consumer_warps({p.consumer_warps})",
        f"      producer_warp({p.producer_warp}) warpgroup({p.warpgroup})",
        f"      matrix_tile({'x'.join(map(str, p.matrix_shape))})",
    ]
    lines += [
        f"  flow.loop @{loop.id} %{loop.variable} = {loop.start} to {loop.stop} step {loop.step}"
        for loop in ir.domain.loops
    ]
    lines += [
        "  flow.tile_map "
        + "\n      ".join(f"%{n} = {expr}" for n, expr in ir.tile_map)
    ]
    lines += [
        f"  flow.executor %{executor.id} role<{enum(executor.role)}> organization<{enum(executor.organization)}>"
        for executor in ir.executors
    ]
    lines += [
        f"  flow.buffer %{buffer.id} slots({buffer.slots}) access<{enum(buffer.access)}> : bytes_per_slot<{buffer.bytes_per_slot}>"
        for buffer in ir.buffers
    ]
    lines += [
        f"  flow.buffer_view %{view.id} of %{view.buffer} offset<{view.offset}> : buffer<{'x'.join(map(str, view.shape))}x{enum(view.dtype)}>"
        for view in ir.buffer_views
    ]
    pipe = ir.domain.pipeline
    lines += [
        f"  flow.pipeline @{pipe.id} %{pipe.loop.variable} = {pipe.loop.start} to {pipe.loop.stop} step {pipe.loop.step} depth({pipe.depth})",
        "      generation_counts_per_wave["
        + ", ".join(map(str, pipe.stage.generation_counts_per_wave))
        + "] {",
    ]
    lines += ["    " + flowop(op) for op in ir.ops if op.scope == A.Scope.PIPELINE]
    lines += (
        ["  }"]
        + ["  " + flowop(op) for op in ir.ops if op.scope != A.Scope.PIPELINE]
        + ["}"]
    )
    if ir.mapping_symbol:
        m = ir.mapping
        lines += [
            "",
            f"mapping.program @{ir.mapping_symbol} flow<@{ir.symbol}>",
            f"    target<{m.target}> placement<{enum(m.placement)}> {{",
        ]
        for executor, unit in m.execute:
            name = next(value.id for value in ir.executors if value.role == executor)
            cap = "bulk_copy" if executor == A.ExecutorRole.PRODUCER else "matrix.mma"
            lines.append(f"  map.execute @{name} capability<{cap}> -> @{unit}")
        lines += [
            f"  map.store @{buffer} -> @{enum(storage)}" for buffer, storage in m.store
        ]
        lines += [
            f"  map.route @input_route capability<{cap}> @{enum(src)} -> @{enum(dst)} via<@tma_route>"
            for src, dst, cap in m.routes
        ] + ["}"]
    return "\n".join(lines) + "\n"


def serialize(ir):
    if isinstance(ir, A.ProgIR):
        return prog(ir)
    if isinstance(ir, A.AlgoIR):
        return algo(ir)
    if isinstance(ir, A.FlowIR):
        return flow(ir)
    from .json_codec import dumps

    return dumps(ir)
