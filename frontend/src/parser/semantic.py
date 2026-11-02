"""Convert parsed assembly into independent, typed IR data structures."""

import re
from dataclasses import replace
from . import ast as A


def match(pattern, text):
    result = re.search(pattern, text, re.S)
    if result is None:
        raise ValueError(f"expected {pattern!r} in {text!r}")
    return result


def scalar(text):
    text = text.strip()
    return int(text) if re.fullmatch(r"-?\d+", text) else text


def items(text):
    return tuple(part.strip().lstrip("@%!") for part in text.split(",") if part.strip())


def tensor(name, text, **options):
    shape, dtype, layout = match(
        r"tensor<(.+)x(fp16|fp32|bf16|int32)(?:,\s*(\w+))?>", text
    ).groups()
    storage = re.search(r"storage<(\w+)>", text)
    return A.Tensor(
        name,
        tuple(scalar(d.lstrip("%")) for d in shape.split("x")),
        A.DataType(dtype),
        A.MemorySpace(storage[1]) if storage else None,
        "readonly" in text,
        A.Layout(layout) if layout else None,
        "external" in text,
        **options,
    )


def statements(nodes):
    result = []
    for node in nodes:
        if isinstance(node, (A.Operation, A.StaticBranch)):
            result.append(node)
        elif node.opcode or not result or not isinstance(result[-1], str):
            if node.text.strip():
                result.append(node.text.strip())
        elif node.text.strip():
            result[-1] += " " + node.text.strip()
    return result


def assignments(text):
    return tuple(
        (name, value.strip())
        for name, value in re.findall(r"%(\w+)\s*=\s*(.*?)(?=\s+%\w+\s*=|$)", text)
    )


def prog(root, static):
    header = " ".join(root.header)
    symbol = match(r"prog.program\s+@(\w+)", header)[1]
    tensors, streams, launches, commands = [], [], [], []
    returns, after = (), ()
    for node in statements(root.body):
        if isinstance(node, A.Operation):
            stream = match(r"prog.stream\s+@(\w+)\s+ordered", " ".join(node.header))[1]
            streams.append(stream)
            for line in statements(node.body):
                if not isinstance(line, str):
                    raise ValueError("unexpected nested program region")
                if "prog.launch" in line:
                    lhs, identity, callee, implementation, ins, outs, attrs = match(
                        r"(.*?)=\s*prog.launch\s+@(\w+)\s+callee<@(\w+)>\s+implementation<@(\w+)>\s+inputs\[(.*?)\]\s+outputs\[(.*?)\]\s+attrs\{(.*?)\}",
                        line,
                    ).groups()
                    outputs, inputs = items(outs), items(ins)
                    events = tuple(re.findall(r"!(\w+)", lhs))
                    attributes = tuple(
                        (k.strip(), v.strip())
                        for k, v in (
                            part.split("=", 1)
                            for part in attrs.split(",")
                            if part.strip()
                        )
                    )
                    launches.append(
                        A.Invocation(
                            identity,
                            inputs=inputs,
                            outputs=outputs,
                            events=events,
                            callee=callee,
                            implementation=implementation,
                            attributes=attributes,
                        )
                    )
                    commands.append(A.Command(A.CommandKind.LAUNCH, identity, ()))
                elif line.startswith("prog.synchronize"):
                    target, deps = match(
                        r"prog.synchronize\s+@(\w+)\s+after\[(.*?)\]", line
                    ).groups()
                    commands.append(
                        A.Command(A.CommandKind.SYNCHRONIZE, target, items(deps))
                    )
                else:
                    raise ValueError(f"unsupported program statement: {line}")
        elif node.startswith("prog.tensor"):
            tensors.append(tensor(match(r"prog.tensor\s+@(\w+)", node)[1], node))
        elif node.startswith("prog.return"):
            names, deps = match(r"prog.return\s+(.*?)\s+after\[(.*?)\]", node).groups()
            returns = items(names)
            after = items(deps)
        else:
            raise ValueError(f"unsupported program statement: {node}")
    return A.ProgIR(
        symbol,
        static=static,
        tensors=tuple(tensors),
        streams=tuple(streams),
        launches=tuple(launches),
        commands=tuple(commands),
        returns=returns,
        return_after=after,
    )


def algo(root, static):
    header = " ".join(root.header)
    symbol, ins, outs = match(
        r"algo.kernel\s+@(\w+)\((.*?)\)\s*->\s*\((.*?)\)", header
    ).groups()
    tensors = tuple(
        tensor(name, ty)
        for name, ty in re.findall(r"%(\w+):\s*(tensor<[^>]+>)", ins + "," + outs)
    )
    domains = []
    accumulators = []
    computations = []
    operations = []
    stages = []
    families = ()
    requirements = ()
    for line in statements(root.body):
        if not isinstance(line, str):
            raise ValueError("unexpected algorithm region")
        if line.startswith("algo.domain"):
            domains.extend(re.findall(r"%(\w+)\s+in\s+\[([^,]+),\s*([^\)]+)\)", line))
        elif line.startswith("algo.accumulator"):
            name, ty, init = match(
                r"algo.accumulator\s+@(\w+)\s*:\s*(tensor<[^>]+>)\s+init<(\w+)>", line
            ).groups()
            tensors += (tensor(name, ty),)
            accumulators.append(
                A.Constant(
                    name,
                    0 if init == "zero" else A.ConstantValue(init),
                    tensors[-1].dtype,
                )
            )
        elif line.startswith("algo.compute"):
            name, result, meaning = match(
                r"algo.compute\s+@(\w+)\s+result<(.*?)>\s+semantics<(.*)>$", line
            ).groups()
            computations.append(A.Computation(name, result, meaning))
        elif line.startswith("algo.op "):
            operations.append(math_operation(line))
        elif line.startswith("algo.stage"):
            name = match(r"algo.stage\s+@(\w+)", line)[1]
            values = []
            for label in ("reads", "after", "produces", "writes"):
                m = re.search(label + r"\[(.*?)\]", line)
                values.append(items(m[1]) if m else ())
            stages.append(A.AlgorithmStage(name, *values))
        elif line.startswith("algo.traits"):
            families = tuple(
                A.AlgorithmTrait(value) for value in items(match(r"<(.*)>", line)[1])
            )
        elif line.startswith("algo.require"):
            requirements = items(match(r"<(.*)>", line)[1])
        else:
            raise ValueError(f"unsupported algorithm statement: {line}")
    return A.AlgoIR(
        symbol,
        static=static,
        inputs=tuple(re.findall(r"%(\w+):", ins)),
        outputs=tuple(re.findall(r"%(\w+):", outs)),
        tensors=tensors,
        domains=tuple(domains),
        accumulators=tuple(accumulators),
        computations=tuple(computations),
        stages=tuple(stages),
        operations=tuple(operations),
        traits=families,
        requirements=requirements,
    )


def value(text):
    name, previous, initial = match(
        r"%(\w+)(\.prev)?(?:<(\w+)>)?$", text.strip()
    ).groups()
    return A.ValueId(
        name,
        result=0,
        iteration_delta=-1 if previous else 0,
        initial=(0 if initial == "zero" else A.ConstantValue(initial))
        if initial
        else None,
    )


def region(text):
    name, offsets, shape = match(r"@(\w+)\[(.*?)\]<([\dx]+)>", text).groups()
    return A.SymbolicRegion(
        name,
        tuple(A.IndexExpr(expression=x.strip()) for x in split_balanced(offsets)),
        tuple(int(n) for n in shape.split("x")),
    )


def operation(line, stage, views):
    opcode, identity = match(r"flow\.(\w+)\s+@(\w+)", line).groups()
    kind = {
        "acquire": A.OpKind.ACQUIRE,
        "release": A.OpKind.RELEASE,
        "load_async": A.OpKind.COPY,
        "matmul": A.OpKind.MATRIX,
        "store_async": A.OpKind.STORE,
    }.get(opcode)
    if kind is None:
        raise ValueError(f"unsupported Flow operation: {opcode}")
    deps = []
    depmatch = re.search(r"after\[(.*?)\]", line)
    for dep in depmatch[1].split(",") if depmatch else ():
        src, phase, relation = match(r"@(\w+)\.(\w+):(\w+)", dep).groups()
        deps.append(
            A.Dependency(src, A.Phase(phase), dependency_kind(relation), relation)
        )
    reads = re.search(r"reads\[(.*?)\]", line)
    operands = tuple(value(v) for v in reads[1].split(",")) if reads else ()
    reg = region(line) if re.search(r"@\w+\[", line) else None
    executor = A.ExecutorRole.MATRIX if opcode == "matmul" else A.ExecutorRole.PRODUCER
    buffer_match = re.search(r"(?:slots\[|to\s+)%([\w]+)\.s", line)
    slot = None
    if buffer_match:
        name = buffer_match[1]
        base = next((view.buffer for view in views if view.id == name), name)
        slot = A.SlotUse(base, stage)
    cap = re.search(r"requires<(.*?)>", line)
    event = re.search(r"!(\w+)\s*=", line)
    dest = re.search(r"to\s+(%[\w.]+\([^)]*\))", line)
    cast = re.search(r"cast<(\w+)>", line)
    results = ()
    if opcode in ("load_async", "matmul"):
        ty = tensor(identity, match(r":\s*(tensor<[^>]+>)", line)[1])
        results = (A.SSAValue(A.ValueId(identity, 0, 0, None), ty.shape, ty.dtype),)
    return A.FlowOp(
        identity,
        kind,
        A.Scope.EPILOGUE if opcode == "store_async" else A.Scope.PIPELINE,
        executor,
        operands,
        tuple(deps),
        reg,
        slot,
        results,
        math=None,
        capabilities=items(cap[1]) if cap else (),
        destination=dest[1] if dest else None,
        event=event[1] if event else None,
        cast=A.DataType(cast[1]) if cast else None,
        implements=(
            re.search(r"implements<@(\w+)>", line)[1] if "implements<" in line else None
        ),
    )


def flow_statements(nodes, condition="true"):
    """Preserve branch guards while decoding Flow dependency-rule leaves."""
    alternatives = []
    for node in statements(nodes):
        if isinstance(node, A.StaticBranch):
            if node.iterations:
                if node.kind != A.StaticBranchKind.IF or condition != "true":
                    raise ValueError(
                        "nested scoped static branches are not supported yet"
                    )
                yield node, "true"
                yield from flow_statements(node.body)
                continue
            if node.kind == A.StaticBranchKind.IF:
                alternatives = []
            prior = " or ".join(f"({value})" for value in alternatives)
            selected = node.condition or "true"
            if prior:
                selected = f"not ({prior}) and ({selected})"
            effective = (
                selected if condition == "true" else f"({condition}) and ({selected})"
            )
            yield from flow_statements(node.body, effective)
            if node.condition is not None:
                alternatives.append(node.condition)
        else:
            if condition != "true" and (
                not isinstance(node, str) or not node.startswith("flow.dependency_rule")
            ):
                raise ValueError(
                    "Flow static branches currently support dependency rules only"
                )
            yield node, condition


def flow_region(root, mapping, static):
    symbol, schedule, target = match(
        r"flow.(?:kernel|region)\s+@(\w+)\s+schedule<(\w+)>\s+target<(\w+)>",
        " ".join(root.header),
    ).groups()
    tensors = []
    executors = []
    buffers = []
    views = []
    loops = []
    ops = []
    config = {}
    pipeline = None
    rules = []
    static_ifs = []
    mathematics = {}
    for node, static_condition in flow_statements(root.body):
        if isinstance(node, A.StaticBranch):
            static_ifs.append(
                A.StaticIf(node.condition.lstrip("%"), node.condition, node.iterations)
            )
            continue
        if isinstance(node, A.Operation):
            text = " ".join(node.header)
            name, var, end, depth, gens = match(
                r"flow.pipeline\s+@(\w+)\s+%(\w+)\s*=\s*0\s+to\s+(.*?)\s+step\s+1\s+depth\((\d+)\)\s+generation_counts_per_wave\[(.*?)\]",
                text,
            ).groups()
            stage = A.PipelineStage(
                A.IterationId(A.IndexExpr(expression="%" + var)),
                int(depth),
                tuple(scalar(v) for v in split_balanced(gens)),
            )
            pipeline = A.PipelineRegion(
                name,
                A.TileLoop(name, var, 0, scalar(end.strip()), 1),
                int(depth),
                stage,
            )
            ops.extend(operation(line, stage, views) for line in statements(node.body))
            continue
        line = node
        if line.startswith("flow.math"):
            op = math_operation(line)
            mathematics[op.id] = op
        elif line.startswith("flow.dependency_rule"):
            name, sources, bindings, selection = match(
                r"flow.dependency_rule\s+@(\w+)\s+from\[(.*?)\]\s+at\[(.*?)\]\s+select<(\w+)>",
                line,
            ).groups()
            events = [
                match(r"@(\w+)\.(\w+)", part).groups() for part in sources.split(",")
            ]
            phases = {phase for _, phase in events}
            if len(phases) != 1:
                raise ValueError("dependency_rule sources must share an event phase")
            rules.append(
                A.DependencyRule(
                    name,
                    tuple(source for source, _ in events),
                    A.Phase(next(iter(phases))),
                    A.Selection(selection),
                    static_condition,
                    tuple(
                        (key.strip().lstrip("%"), value.strip())
                        for key, value in (
                            entry.split("=", 1) for entry in split_balanced(bindings)
                        )
                    ),
                )
            )
        elif line.startswith("flow.tensor"):
            tensors.append(tensor(match(r"flow.tensor\s+@(\w+)", line)[1], line))
        elif line.startswith("flow.schedule_config"):
            config = dict(re.findall(r"(\w+)\(([^)]+)\)", line))
        elif line.startswith("flow.loop"):
            loop_id, var, end = match(
                r"flow.loop\s+@(\w+)\s+%(\w+)\s*=\s*0\s+to\s+(.*?)\s+step\s+1", line
            ).groups()
            loops.append(A.TileLoop(loop_id, var, 0, scalar(end), 1))
        elif line.startswith("flow.executor"):
            name, role, organization = match(
                r"flow.executor\s+%(\w+)\s+role<(\w+)>\s+organization<(\w+)>", line
            ).groups()
            executors.append(
                A.Executor(name, A.ExecutorRole(role), A.Organization(organization))
            )
        elif line.startswith("flow.buffer_view"):
            name, base, offset, shape, dtype = match(
                r"flow.buffer_view\s+%(\w+)\s+of\s+%(\w+)\s+offset<(\d+)>\s*:\s*buffer<([\dx]+)x(\w+)>",
                line,
            ).groups()
            views.append(
                A.BufferView(
                    name,
                    base,
                    int(offset),
                    tuple(map(int, shape.split("x"))),
                    A.DataType(dtype),
                )
            )
        elif line.startswith("flow.buffer"):
            name, slots, access, size = match(
                r"flow.buffer\s+%(\w+)\s+slots\((\d+)\)\s+access<(\w+)>\s*:\s*bytes_per_slot<(\d+)>",
                line,
            ).groups()
            buffers.append(A.Buffer(name, int(slots), A.AccessMode(access), int(size)))
        elif "flow.store_async" in line:
            ops.append(operation(line, pipeline.stage, views))
        else:
            raise ValueError(f"unsupported Flow statement: {line}")
    if pipeline is None:
        raise ValueError("missing flow.pipeline")
    expected_config = {
        "tile",
        "stages",
        "warps",
        "consumer_warps",
        "producer_warp",
        "warpgroup",
        "matrix_tile",
    }
    if set(config) != expected_config:
        raise ValueError("schedule_config: missing or unknown fields")
    profile = A.ScheduleConfig(
        tuple(map(int, config["tile"].split("x"))),
        int(config["stages"]),
        int(config["warps"]),
        int(config["consumer_warps"]),
        int(config["producer_warp"]),
        int(config["warpgroup"]),
        tuple(map(int, config["matrix_tile"].split("x"))),
    )
    execute = []
    store = []
    routes = []
    mapping_symbol = None
    if mapping:
        mh = " ".join(mapping.header)
        mapping_symbol, flowref, mtarget, placement = match(
            r"mapping.program\s+@(\w+)\s+flow<@(\w+)>\s+target<(\w+)>\s+placement<(\w+)>",
            mh,
        ).groups()
        if flowref != symbol:
            raise ValueError("mapping refers to a different Flow kernel")
        executor_roles = {value.id: value.role for value in executors}
        for line in statements(mapping.body):
            if line.startswith("map.execute"):
                name, cap, unit = match(
                    r"map.execute\s+@(\w+)\s+capability<([^>]+)>\s*->\s*@(\w+)", line
                ).groups()
                execute.append((executor_roles[name], unit))
            elif line.startswith("map.store"):
                name, storage = match(
                    r"map.store\s+@(\w+)\s*->\s*@(\w+)", line
                ).groups()
                store.append((name, A.MemorySpace(storage)))
            elif line.startswith("map.route"):
                _, _, src, dst, _ = match(
                    r"map.route\s+@(\w+)\s+capability<([^>]+)>\s+@(\w+)\s*->\s*@(\w+)\s+via<@(\w+)>",
                    line,
                ).groups()
                routes.append((A.MemorySpace(src), A.MemorySpace(dst), "bulk_copy"))
            else:
                raise ValueError(f"unsupported mapping statement: {line}")
        mapobj = A.Mapping(
            mtarget, A.Placement(placement), tuple(execute), tuple(store), tuple(routes)
        )
    else:
        raise ValueError("flow.kernel requires an explicit mapping.program")
    if mtarget != target:
        raise ValueError("mapping target differs from flow.kernel target")
    domain = A.ScheduleDomain(pipeline, tuple(loops))
    return A.FlowRegion(
        symbol,
        static=static,
        schedule_id=schedule,
        config=profile,
        domain=domain,
        tensors=tuple(tensors),
        mapping=mapobj,
        ops=tuple(replace(op, math=mathematics.get(op.implements)) for op in ops),
        executors=tuple(executors),
        buffers=tuple(buffers),
        buffer_views=tuple(views),
        mapping_symbol=mapping_symbol,
        dependency_rules=tuple(rules),
        static_ifs=tuple(static_ifs),
        after=items(re.search(r"after\[(.*?)\]", " ".join(root.header))[1])
        if "after[" in " ".join(root.header)
        else (),
        origin=re.search(r"origin<@(\w+)>", " ".join(root.header))[1]
        if "origin<" in " ".join(root.header)
        else None,
    )


def split_balanced(text):
    parts = []
    start = 0
    depth = 0
    for i, char in enumerate(text):
        depth += (char == "(") - (char == ")")
        if char == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return tuple(parts)


def dependency_kind(label):
    if label == "buffer_reuse":
        return A.DependencyKind.BUFFER_REUSE
    if label in ("previous_fragment", "final_round"):
        return A.DependencyKind.LOOP_CARRIED
    if label in ("previous_tile_store", "previous_chunk"):
        return A.DependencyKind.OUTPUT_ORDER
    if label == "previous_load":
        return A.DependencyKind.LOOP_CARRIED
    return A.DependencyKind.DATA


def math_operation(line):
    identity, category, cost, dtype, inputs, output, axes = match(
        r"(?:algo.op|flow.math)\s+@(\w+)\s+category<(\w+)>\s+cost<(\w+)>\s+dtype<(\w+)>\s+inputs\[(.*?)\]\s+output\[%(\w+)\]\s+reduction\[(.*?)\]",
        line,
    ).groups()
    category = A.OperationClass(category)
    subtypes = {
        A.OperationClass.ARITHMETIC: A.ArithmeticCost,
        A.OperationClass.SPECIAL_FUNCTION: A.SpecialFunction,
        A.OperationClass.REDUCTION: A.ReductionKind,
    }
    if category in subtypes:
        cost = subtypes[category](cost)
    elif cost == "none":
        cost = None
    else:
        raise ValueError(f"{category.value} does not accept cost subtype {cost}")
    return A.MathOperation(
        identity, category, cost, A.DataType(dtype), items(inputs), output, items(axes)
    )


def flow_program(root, static):
    from .static import declarations

    header = " ".join(root.header)
    symbol = match(r"flow.program\s+@(\w+)", header)[1]
    stream = re.search(r"stream<@(\w+)>", header)
    outputs = re.search(r"outputs\[(.*?)\]", header)
    tensors, regions, mappings = [], [], {}
    for node in statements(root.body):
        if isinstance(node, A.Operation) and node.name == "mapping.program":
            mappings[match(r"flow<@(\w+)>", " ".join(node.header))[1]] = node
    for node in statements(root.body):
        if isinstance(node, A.Operation):
            if node.name == "mapping.program":
                continue
            if node.name != "flow.region":
                raise ValueError(f"unsupported Flow region: {node.name}")
            body, local = declarations(node)
            name = match(r"flow.region\s+@(\w+)", " ".join(node.header))[1]
            regions.append(flow_region(body, mappings[name], local))
        elif node.startswith("flow.tensor"):
            tensors.append(tensor(match(r"flow.tensor\s+@(\w+)", node)[1], node))
        else:
            raise ValueError(f"unsupported Flow program statement: {node}")
    return A.FlowIR(
        symbol,
        static,
        tuple(tensors),
        tuple(regions),
        stream[1] if stream else None,
        items(outputs[1]) if outputs else (),
    )
