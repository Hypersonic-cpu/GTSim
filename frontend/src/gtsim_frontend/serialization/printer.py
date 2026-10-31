"""Deterministic inspection syntax; JSON remains the authoritative encoding."""
from ..ir.program import Program
from ..ir.algorithm import Algorithm
from ..ir.flow import FlowKernel
from ..ir.arch import ArchKernel
from ..ir.execution import ExecProgram,ExecKernel


def print_ir(value):
    value.verify()
    if isinstance(value,Program):
        lines=[f'prog.program @{value.symbol} {{',f'  prog.stream @{value.streams[0]} ordered']
        lines += [f'  prog.tensor @{t.id} shape{t.shape} dtype<{t.dtype}> tier<{t.tier}>' for t in value.tensors]
        lines += [f'  prog.launch @{l.id} callee<{l.algorithm.family}> bindings{l.bindings}' for l in value.launches]
        lines += [f'  prog.{c.kind} @{c.target}' for c in value.commands]
    elif isinstance(value,Algorithm):
        lines=[f'algo.kernel @{value.symbol} family<{value.family}> dimensions{value.dimensions} {{']
        lines += [f'  algo.stage {stage}' for stage in value.stages]
        lines += [f'  algo.contract "{formula}"' for formula in value.computation]
    elif isinstance(value,FlowKernel):
        lines=[f'flow.kernel @{value.symbol} schedule<{value.schedule_id}> {{']
        lines += [f'  flow.buffer @{b.id} slots({b.slots}) bytes_per_slot({b.bytes_per_slot})' for b in value.buffers]
        lines += [f'  flow.{o.capability} @{o.id} executor<{o.executor}> semantic<{o.semantic}> inputs{o.inputs} operands{o.operands} slot<{o.slot}>' for o in value.ops]
        lines += [f'  map.execute @{e} -> {engine}' for e,engine in value.mapping.execute]
        lines += [f'  map.cta {cta} -> sm{sm}' for cta,sm in value.mapping.ctas]
    elif isinstance(value,ArchKernel):
        prefix='exec' if isinstance(value,ExecKernel) else 'arch'
        lines=[f'{prefix}.kernel @{value.symbol} target<{value.target}> sync<{value.sync_semantics_version}> {{']
        lines += [f'  {prefix}.cta {c.id} sm({c.sm}) warps({c.warps}) shared({c.shared_bytes}) registers({c.register_bytes})' for c in value.ctas]
        lines += [f'  %{o.id} = {o.op} cta({o.cta}) warp({o.warp}) tensor({o.tensor}) tile{o.tile} after_issue{o.after_issue} after_complete{o.after_complete}' for o in value.ops]
    elif isinstance(value,ExecProgram):
        lines=[f'exec.program @{value.symbol} stream<{value.streams[0]}> {{']
        lines += [f'  exec.launch @{l.id} graph<{l.kernel_file}> bindings{l.bindings}' for l in value.launches]
        lines += [f'  exec.{c.kind} @{c.target}' for c in value.commands]
    else:raise ValueError('unknown IR inspection type')
    return '\n'.join(lines+['}'])+'\n'
