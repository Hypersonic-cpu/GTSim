"""Deterministic inspection syntax; JSON remains the authoritative encoding."""

from ..ir.program import Program
from ..ir.algorithm import Algorithm
from ..ir.flow import FlowKernel
from ..ir.arch import ArchKernel
from ..ir.execution import ExecProgram, ExecKernel


def print_ir(value):
    value.verify()
    if isinstance(value, Program):
        lines = [
            f"prog.program @{value.symbol} {{",
            f"  prog.stream @{value.streams[0]} ordered",
        ]
        lines += [
            f"  prog.tensor @{t.id} shape{t.shape} dtype<{t.dtype}> tier<{t.tier}>"
            for t in value.tensors
        ]
        lines += [
            f"  prog.launch @{l.id} callee<{l.algorithm.family}> bindings{l.bindings}"
            for l in value.launches
        ]
        lines += [f"  prog.{c.kind} @{c.target}" for c in value.commands]
    elif isinstance(value, Algorithm):
        lines = [
            f"algo.kernel @{value.symbol} family<{value.family}> dimensions{value.dimensions} {{"
        ]
        lines += [f"  algo.stage {stage}" for stage in value.stages]
        lines += [f'  algo.contract "{formula}"' for formula in value.computation]
    elif isinstance(value, (FlowKernel, ArchKernel)):
        prefix = "flow" if isinstance(value, FlowKernel) else "arch"
        lines = [f"{prefix}.kernel @{value.symbol} schedule<{value.schedule_id}> {{"]
        lines += [
            f"  {prefix}.loop {loop.variable.value} count({loop.count}) predicate<{loop.predicate.value}>"
            for loop in value.domain.loops
        ]
        pipeline = value.domain.pipeline
        lines += [
            f"  {prefix}.pipeline k count({pipeline.loop.count}) depth({pipeline.depth}) stage<{pipeline.stage}>"
        ]
        if isinstance(value, FlowKernel):
            lines += [
                f"  {op.kind.value} @{op.id} role<{op.role.value}> executor<{op.executor.value}> deps{op.dependencies} region<{op.region}>"
                for op in value.ops
            ]
        else:
            lines += [
                f"  arch.pattern @{pattern.source.id} role<{pattern.source.role.value}> steps{pattern.steps}"
                for pattern in value.ops
            ]
            lines += [f"  arch.async {value.async_protocol} resources{value.resources}"]
        lines += [
            f"  map.execute {executor.value} -> {engine}"
            for executor, engine in value.mapping.execute
        ]
    elif isinstance(value, ExecKernel):
        lines = [f"exec.kernel @{value.symbol} {{"]
        lines += [
            f"  %{op.id} = {op.op} cta({op.cta}) warp({op.warp}) tile{op.tile} after_issue{op.after_issue} after_complete{op.after_complete}"
            for op in value.ops
        ]
    elif isinstance(value, ExecProgram):
        lines = [f"exec.program @{value.symbol} stream<{value.streams[0]}> {{"]
        lines += [
            f"  exec.launch @{l.id} graph<{l.kernel_file}> bindings{l.bindings}"
            for l in value.launches
        ]
        lines += [f"  exec.{c.kind} @{c.target}" for c in value.commands]
    else:
        raise ValueError("unknown IR inspection type")
    return "\n".join(lines + ["}"]) + "\n"
