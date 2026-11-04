"""Text assembly for structured ArchIR."""

import re
from . import ast as A
from .semantic import items, scalar, split_balanced, statements, tensor
from .static import declarations, emit
from .writer import names, tensor as tensor_type


def read(pattern, text):
    match = re.fullmatch(pattern, text)
    if match is None:
        raise ValueError(f"invalid native assembly: {text!r}")
    return match.groups()


def head(node):
    if not isinstance(node, A.Operation):
        raise ValueError(f"expected native operation: {node}")
    return " ".join(s.strip() for s in node.header).removesuffix("{").rstrip()


def seq(text):
    return tuple(scalar(x) for x in split_balanced(text)) if text else ()


def csv(values):
    return ", ".join(map(str, values))


def binds(text):
    return tuple(
        (key.strip().removeprefix("%"), value.strip())
        for key, value in (x.split("=", 1) for x in split_balanced(text) if x)
    )


def binding_text(values):
    return csv(f"%{key} = {value}" for key, value in values)


def deps(text):
    return tuple(
        A.Dependency(src, A.Phase(phase), A.DependencyKind(kind), label)
        for src, phase, kind, label in (
            read(r"@(\w+)\.(\w+):(\w+):(.*)", x.strip())
            for x in text.split(",")
            if x.strip()
        )
    )


def after(values):
    return (
        "after["
        + csv(f"@{d.source}.{d.phase.value}:{d.kind.value}:{d.label}" for d in values)
        + "]"
    )


def loop_text(l):
    return f"arch.loop @{l.id} %{l.variable} = {l.start} to {l.stop} step {l.step}"


def loop(line):
    name, var, start, end, step = read(
        r"arch.loop @(\w+) %(\w+) = (.*?) to (.*?) step (.*)", line
    )
    return A.TileLoop(name, var, scalar(start), scalar(end), scalar(step))


def tensor_text(t):
    return (
        f"arch.tensor @{t.id} : {tensor_type(t, layout=True, symbolic=True)}"
        + (f" storage<{t.tier.value}>" if t.tier else "")
        + (" external" if t.external else "")
        + (" readonly" if t.read_only else "")
    )


def parse_tensor(line):
    name, ty = read(r"arch.tensor @(\w+) : (.*)", line)
    return tensor(name, ty)


def step_text(s):
    r = s.region
    loc = (
        f" region<@{r.tensor}[{csv(x.expression for x in r.offsets)}] shape[{csv(r.shape)}]>"
        if r
        else ""
    )
    ld = f" leading_dim<{s.leading_dim.expression}>" if s.leading_dim else ""
    args = binding_text((a.id, a.expression) for a in s.static_args)
    return [
        f"arch.op opcode<{s.opcode.value}> resource<{s.resource.value}> executor<{s.executor.expression}> "
        f"dtype<{s.dtype.value}>{loc}{ld} {after(s.dependencies)} inputs[{csv(s.input_steps)}] "
        f"work[{csv(s.work_shape)}] guard<{s.guard}> static.args[{args}] {{",
        *("  " + loop_text(l) for l in s.loops),
        "}",
    ]


def step(node):
    (
        op,
        resource,
        executor,
        dtype,
        name,
        offsets,
        shape,
        ld,
        edges,
        inputs,
        work,
        guard,
        args,
    ) = read(
        r"arch.op opcode<(.*?)> resource<(\w+)> executor<(.*?)> dtype<(\w+)>"
        r"(?: region<@(\w+)\[(.*?)\] shape\[(.*?)\]>)?(?: leading_dim<(.*?)>)?"
        r" after\[(.*?)\] inputs\[(.*?)\] work\[(.*?)\] guard<(.*?)> static.args\[(.*)\]",
        head(node),
    )
    region = (
        A.SymbolicRegion(
            name, tuple(map(A.IndexExpr, split_balanced(offsets))), seq(shape)
        )
        if name
        else None
    )
    return A.NativeStep(
        A.NativeOpcode(op),
        A.ExecutionResource(resource),
        A.IndexExpr(executor),
        tuple(loop(line) for line in statements(node.body)),
        region,
        A.DataType(dtype),
        A.IndexExpr(ld) if ld is not None else None,
        deps(edges),
        tuple(map(int, items(inputs))),
        seq(work),
        guard,
        tuple(A.StaticBinding(*pair) for pair in binds(args)),
    )


def region_text(r):
    p, res, reg, sync = r.domain.pipeline, r.resources, r.registers, r.async_protocol
    origin = f" origin<@{r.origin}>" if r.origin is not None else ""
    lines = [
        f"arch.kernel @{r.symbol} target<{r.target}>{origin} after[{names(r.after, '@')}] "
        f"cta<%{r.cta_variable}> sm<{r.sm.expression}> {{",
        *emit(r.static),
        *("  " + tensor_text(t) for t in r.tensors),
        f"  arch.resources warps({res.warps}) shared_bytes({res.shared_bytes}) register_bytes({res.register_bytes})",
        f"  arch.registers accumulator<{reg.accumulator_dtype.value}> fragments[{csv(reg.fragment_shape)}] "
        f"consumer_warps({reg.consumer_warps}) base_per_thread({reg.base_registers_per_thread}) per_cta_bytes({reg.per_cta_bytes})",
        f"  arch.async mbarrier_phases({sync.mbarrier_phases}) commit_groups({sync.commit_groups}) "
        f"wait_group({sync.wait_group}) reuse<{sync.reuse_phase.value}>",
        *(
            f"  arch.buffer %{l.buffer} : smem<{l.layout}> slots({l.slots}) bytes_per_slot({l.bytes_per_slot})"
            for l in r.layouts
        ),
        *("  " + loop_text(l) for l in r.domain.loops),
        f"  arch.pipeline @{p.id} depth({p.depth}) iteration<{p.stage.iteration.index.expression}> "
        f"modulus({p.stage.modulus}) generations[{csv(p.stage.generation_counts_per_wave)}] {{",
        "    " + loop_text(p.loop),
        "  }",
    ]
    for rule in r.dependency_rules:
        sources = csv(f"@{s}.{rule.phase.value}" for s in rule.sources)
        lines.append(
            f"  arch.dependency_rule @{rule.id} from[{sources}] at[{binding_text(rule.producer_bindings)}] "
            f"select<{rule.selection.value}> guard<{rule.static_if}>"
        )
    for guard in r.static_ifs:
        lines.append(
            f"  arch.guard @{guard.id} condition<{guard.condition}> scope[{names(guard.iterations, '%')}]"
        )
    for pat in r.ops:
        lines.append(
            f"  arch.pattern @{pat.id} scope<{pat.scope.value}> {after(pat.dependencies)} {{"
        )
        for s in pat.steps:
            lines.extend("    " + line for line in step_text(s))
        lines.append("  }")
    return [*lines, "}"]


def write(ir):
    stream = f" stream<@{ir.stream}>" if ir.stream is not None else ""
    lines = [
        f"arch.program @{ir.symbol}{stream} outputs[{names(ir.outputs, '@')}] {{",
        *emit(ir.static),
        *("  " + tensor_text(t) for t in ir.tensors),
    ]
    for r in ir.regions:
        lines.extend("  " + line for line in region_text(r))
    return "\n".join([*lines, "}"]) + "\n"


def parse_region(node):
    name, target, origin, predecessors, cta, sm = read(
        r"arch.kernel @(\w+) target<(.*?)>(?: origin<@(\w+)>)? after\[(.*?)\] cta<%(\w+)> sm<(.*)>",
        head(node),
    )
    node, static = declarations(node)
    tensors, layouts, loops, rules, guards, patterns = [], [], [], [], [], []
    required = {}

    def put(key, value):
        if key in required:
            raise ValueError(f"duplicate arch.{key}")
        required[key] = value

    for child in statements(node.body):
        if isinstance(child, A.Operation):
            if child.name == "arch.pipeline":
                pid, depth, iteration, modulus, gens = read(
                    r"arch.pipeline @(\w+) depth\((\d+)\) iteration<(.*?)> modulus\((\d+)\) generations\[(.*)\]",
                    head(child),
                )
                (body,) = statements(child.body)
                stage = A.PipelineStage(
                    A.IterationId(A.IndexExpr(iteration)), int(modulus), seq(gens)
                )
                put("pipeline", A.PipelineRegion(pid, loop(body), int(depth), stage))
            elif child.name == "arch.pattern":
                pid, scope, edges = read(
                    r"arch.pattern @(\w+) scope<(\w+)> after\[(.*)\]", head(child)
                )
                patterns.append(
                    A.NativePattern(
                        pid,
                        A.Scope(scope),
                        tuple(step(s) for s in statements(child.body)),
                        deps(edges),
                    )
                )
            else:
                raise ValueError(f"unsupported Arch kernel operation: {child.name}")
            continue
        if not isinstance(child, str):
            raise ValueError(f"unexpected Arch kernel member: {child}")
        if child.startswith("arch.tensor "):
            tensors.append(parse_tensor(child))
        elif child.startswith("arch.resources "):
            put(
                "resources",
                A.ResourceFootprint(
                    *map(
                        int,
                        read(
                            r"arch.resources warps\((\d+)\) shared_bytes\((\d+)\) register_bytes\((\d+)\)",
                            child,
                        ),
                    )
                ),
            )
        elif child.startswith("arch.registers "):
            dtype, shape, warps, base, size = read(
                r"arch.registers accumulator<(\w+)> fragments\[(.*?)\] consumer_warps\((\d+)\) base_per_thread\((\d+)\) per_cta_bytes\((\d+)\)",
                child,
            )
            put(
                "registers",
                A.RegisterLayout(
                    A.DataType(dtype),
                    tuple(map(int, items(shape))),
                    int(warps),
                    int(base),
                    int(size),
                ),
            )
        elif child.startswith("arch.async "):
            phases, groups, wait, reuse = read(
                r"arch.async mbarrier_phases\((\d+)\) commit_groups\((\d+)\) wait_group\((\d+)\) reuse<(\w+)>",
                child,
            )
            put(
                "async",
                A.AsyncProtocol(int(phases), int(groups), int(wait), A.Phase(reuse)),
            )
        elif child.startswith("arch.buffer "):
            buf, layout, slots, size = read(
                r"arch.buffer %(\w+) : smem<(.*?)> slots\((\d+)\) bytes_per_slot\((\d+)\)",
                child,
            )
            layouts.append(A.SharedLayout(buf, int(slots), layout, int(size)))
        elif child.startswith("arch.loop "):
            loops.append(loop(child))
        elif child.startswith("arch.dependency_rule "):
            rid, sources, at, select, guard = read(
                r"arch.dependency_rule @(\w+) from\[(.*?)\] at\[(.*?)\] select<(\w+)> guard<(.*)>",
                child,
            )
            events = [
                read(r"@(\w+)\.(\w+)", entry.strip()) for entry in sources.split(",")
            ]
            (phase,) = {phase for _, phase in events}
            rules.append(
                A.DependencyRule(
                    rid,
                    tuple(source for source, _ in events),
                    A.Phase(phase),
                    A.Selection(select),
                    guard,
                    binds(at),
                )
            )
        elif child.startswith("arch.guard "):
            gid, condition, scope = read(
                r"arch.guard @(.*?) condition<(.*)> scope\[(.*)\]", child
            )
            guards.append(A.StaticIf(gid, condition, items(scope)))
        else:
            raise ValueError(f"unsupported Arch kernel statement: {child}")
    return A.ArchRegion(
        name,
        static,
        target,
        A.ScheduleDomain(required["pipeline"], tuple(loops)),
        tuple(tensors),
        required["resources"],
        tuple(layouts),
        required["registers"],
        required["async"],
        tuple(patterns),
        tuple(rules),
        tuple(guards),
        cta,
        A.IndexExpr(sm),
        items(predecessors),
        origin,
    )


def parse(root, static):
    name, stream, outputs = read(
        r"arch.program @(\w+)(?: stream<@(\w+)>)? outputs\[(.*)\]", head(root)
    )
    tensors, regions = [], []
    for node in statements(root.body):
        if isinstance(node, A.Operation) and node.name == "arch.kernel":
            regions.append(parse_region(node))
        elif isinstance(node, str) and node.startswith("arch.tensor "):
            tensors.append(parse_tensor(node))
        else:
            raise ValueError(f"unsupported Arch program member: {node}")
    return A.ArchIR(
        name, static, tuple(tensors), tuple(regions), stream, items(outputs)
    )
