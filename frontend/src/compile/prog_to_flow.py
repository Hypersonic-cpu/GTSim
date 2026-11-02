"""Link ordered invocations into one structured FlowIR, using supplied schedules."""

from dataclasses import replace
from pathlib import Path
from parser import ast as A, parse_file
from .algo_to_flow import compile_algorithm
from .static import source_symbol


def compile_program(program_path, parameters, algorithm_paths, schedule_paths):
    program = parse_file(program_path)
    (stream,) = program.streams
    algorithms = {source_symbol(path): Path(path) for path in algorithm_paths}
    schedules = {source_symbol(path): Path(path) for path in schedule_paths}
    tensors = {tensor.id: tensor for tensor in program.tensors}
    invocations = {inv.id: inv for inv in program.launches}
    regions, events, produced = [], {}, {t.id for t in program.tensors if t.external}
    for command in program.commands:
        if command.kind == A.CommandKind.SYNCHRONIZE:
            if command.target != stream or any(
                event not in events for event in command.after
            ):
                raise ValueError(
                    "synchronization references an unavailable stream/event"
                )
            continue
        if command.kind != A.CommandKind.LAUNCH:
            raise ValueError(
                f"unsupported ordered-stream command: {command.kind.value}"
            )
        invocation = invocations[command.target]
        if not set(invocation.inputs) <= produced:
            raise ValueError(f"input has no preceding producer: {invocation.id}")
        values = dict(invocation.attributes)
        algorithm = parse_file(algorithms[invocation.callee])
        if len(algorithm.inputs) != len(invocation.inputs) or len(
            algorithm.outputs
        ) != len(invocation.outputs):
            raise ValueError("invocation tensor arity differs from algorithm signature")
        bindings = dict(
            zip(
                (*algorithm.inputs, *algorithm.outputs),
                (*invocation.inputs, *invocation.outputs),
            )
        )
        region = compile_algorithm(
            algorithms[invocation.callee],
            schedules[invocation.implementation],
            values,
            bindings,
            tensors,
        )
        if invocation.guard:
            region = replace(
                region,
                static=replace(
                    region.static,
                    arguments=region.static.arguments
                    + (A.StaticBinding("__program_guard", invocation.guard.condition),),
                ),
                static_ifs=region.static_ifs
                + (A.StaticIf(invocation.id, "%__program_guard", ()),),
            )
        regions.append(
            replace(
                region,
                symbol=invocation.id,
                mapping_symbol=invocation.id + "_mapping",
                after=(regions[-1].symbol,) if regions else (),
            )
        )
        events.update((event, invocation.id) for event in invocation.events)
        produced.update(invocation.outputs)
    if (
        not set(program.returns) <= produced
        or not set(program.return_after) <= events.keys()
    ):
        raise ValueError("return references unavailable tensors/events")
    return A.FlowIR(
        program.symbol,
        replace(
            program.static,
            arguments=tuple(
                A.StaticBinding(name, repr(value)) for name, value in parameters.items()
            ),
        ),
        program.tensors,
        tuple(regions),
        stream,
        program.returns,
    )
