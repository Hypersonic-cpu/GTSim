"""Expand canonical partitioned native loops and SSA edges only at IR-05.

Arithmetic is described by NativeStep records. This pass only binds iteration
indices, resolves typed values/events, and emits the selected instructions.
"""

from ...ir.execution import ExecKernel
from ...ir.graph import KernelTensor
from ...ir.schedule import Scope, OpKind, LoopVar, Relation, InitialValue
from ...serialization.json_codec import digest
from .expansion import InstructionEmitter, TileContext, cta


def expand(native):
    native.verify()
    emitter = InstructionEmitter(native)
    tensor_ids = {tensor.id: index for index, tensor in enumerate(native.tensors)}
    tensors = tuple(
        KernelTensor(
            index,
            tensor.id,
            tensor.shape,
            (tensor.shape[1], 1),
            tensor.dtype,
            tensor.tier,
            preloaded=tensor.read_only or tensor.id in native.domain.partial_outputs,
            read_only=tensor.read_only,
        )
        for index, tensor in enumerate(native.tensors)
    )
    initializer = {state.producer: state.initializer for state in native.domain.states}
    prologue = {
        pattern.source.role
        for pattern in native.ops
        if pattern.source.scope == Scope.PROLOGUE
    }
    ctas = []
    outputs = set()
    for cta_id, group in enumerate(native.domain.workgroups):
        group_bindings = dict(group.bindings)
        context = TileContext(
            cta_id,
            cta_id % native.domain.num_sms,
            cta_id,
            group_bindings.get(LoopVar.ROW, 0),
            batch=group_bindings.get(LoopVar.BATCH, 0),
            head=group_bindings.get(LoopVar.HEAD, 0),
            kv_head=group_bindings.get(LoopVar.KV_HEAD, 0),
        )
        ctas.append(cta(context, native))
        values = {}
        rounds = (
            len(group.iterations)
            if group.iterations
            else native.domain.pipeline.loop.count
        )
        if group.ranges:
            _, start, end = group.ranges[0]
            rounds = min(
                rounds,
                (end - start + native.profile.kv_tile - 1) // native.profile.kv_tile,
            )
        aliases = dict(group.aliases)
        ranges = {axis: (start, end) for axis, start, end in group.ranges}

        def resolve(value, iteration):
            if value.producer in prologue:
                key = (value.producer, -1)
            else:
                target = iteration + value.iteration_delta
                if target < 0:
                    if value.initial == InitialValue.NONE:
                        raise ValueError("uninitialized loop-carried SSA value")
                    key = (initializer[value.producer], -1)
                else:
                    key = (value.producer, target)
            if key not in values:
                raise ValueError(
                    f"native SSA operand does not dominate consumer: {key}"
                )
            return values[key]

        def emit_pattern(pattern, iteration):
            source = pattern.source
            bindings = context.bindings(max(0, iteration))
            bindings.update(group_bindings)
            if group.iterations and iteration >= 0:
                bindings.update(dict(group.iterations[iteration]))
            valid = {
                axis: min(
                    native.profile.kv_tile,
                    end - start - max(0, iteration) * native.profile.kv_tile,
                )
                for axis, (start, end) in ranges.items()
            }
            internals = []
            for step_index, step in enumerate(pattern.steps):
                dependencies = [resolve(value, iteration) for value in step.operands]
                for index in step.input_steps:
                    if index >= len(internals):
                        raise ValueError(
                            "native internal step does not dominate consumer"
                        )
                    dependencies.append(internals[index])
                groups = {}
                for dep in step.dependencies:
                    if dep.relation == Relation.BUFFER_REUSE:
                        earlier = iteration - native.domain.pipeline.depth
                        groups[dep.relation] = (
                            [values[(dep.source, earlier)]] if earlier >= 0 else []
                        )
                    elif dep.relation != Relation.EXPLICIT:
                        raise ValueError(
                            f"unsupported partitioned native event {dep.relation}"
                        )
                tensor = 0
                dtype = step.math.dtype if step.math else native.tensors[0].dtype
                if source.region is not None:
                    name = source.region.tensor
                    name = aliases.get(name, name)
                    tensor = tensor_ids[name]
                    spec = native.tensors[tensor]
                    row, col = source.region.offsets_at(bindings)
                    rows, cols = step.fragment_shape[:2]
                    rows = min(rows, spec.shape[0] - row)
                    cols = min(cols, spec.shape[1] - col)
                    if source.region.valid_axis:
                        rows = min(rows, ranges[source.region.valid_axis][1] - row)
                    region = (row, col, rows, cols)
                    dtype = spec.dtype
                    if min(rows, cols) <= 0:
                        raise ValueError("empty or invalid native memory region")
                    if source.kind == OpKind.STORE:
                        outputs.add(tensor)
                else:
                    shape = list(step.fragment_shape)
                    for index, axis in enumerate(step.fragment_axes):
                        if axis in valid:
                            shape[index] = valid[axis]
                    if len(shape) == 1:
                        shape.insert(0, 1)
                    region = (0, 0, *shape[:2])
                nid = emitter.emit(
                    source.role,
                    context,
                    native.profile.producer_warp
                    if source.kind in (OpKind.COPY, OpKind.STORE)
                    else 0,
                    f"cta{cta_id}_iter{iteration}_op{source.role}_step{step_index}",
                    tensor,
                    region,
                    groups,
                    dtype=dtype,
                    ld=native.tensors[tensor].shape[1]
                    if step.opcode.startswith("cp.")
                    else 0,
                    step=step_index,
                    value_dependencies=tuple(dependencies),
                )
                internals.append(nid)
            values[(source.role, iteration)] = internals[-1]

        for pattern in native.ops:
            if pattern.source.scope == Scope.PROLOGUE:
                emit_pattern(pattern, -1)
        for iteration in range(rounds):
            for pattern in native.ops:
                if pattern.source.scope == Scope.PIPELINE:
                    emit_pattern(pattern, iteration)
        for pattern in native.ops:
            if pattern.source.scope == Scope.EPILOGUE:
                emit_pattern(pattern, rounds - 1)
    result = ExecKernel(
        symbol=native.symbol,
        source_hash=digest(native),
        sync_semantics_version=native.sync_semantics_version,
        tensors=tensors,
        ctas=tuple(ctas),
        ops=emitter.finish(),
        outputs=tuple(sorted(outputs)),
        partial_outputs=tuple(
            tensor_ids[name] for name in native.domain.partial_outputs
        ),
    )
    result.verify()
    return result
