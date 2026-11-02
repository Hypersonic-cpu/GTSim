"""IR-04 → IR-05: resolve static scopes, expand loops and assemble a concrete DAG."""

from dataclasses import replace
from parser import ast as A
from .static import EMPTY, Resolver, bound_resolver
from .expand_native import NativeExpander


def strides(shape):
    stride, result = 1, []
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


def lower_arch(arch, parameters=None):
    root_parent = Resolver(EMPTY, parameters or {})
    root = bound_resolver(arch.static, root_parent).materialize_constants()
    tensors = []
    for value in arch.tensors:
        shape = tuple(root.value(v) for v in value.shape)
        tensors.append(
            A.KernelTensor(
                len(tensors),
                value.id,
                shape,
                strides(shape),
                value.dtype,
                value.tier or A.MemorySpace.HBM,
                value.external,
                value.read_only,
            )
        )
    by_name = {t.name: t for t in tensors}
    nodes, ctas, regions, tails = [], [], [], {}
    target = None
    for region in arch.regions:
        arguments = {
            arg.id: root.value(arg.expression) for arg in region.static.arguments
        }
        values = {**root.values, **arguments}
        provisional = Resolver(
            replace(region.static, parameters=(), constraints=()), values
        )
        predecessors = tuple(n for name in region.after for n in tails[name])
        if any(
            not provisional.value(guard.condition)
            for guard in region.static_ifs
            if not guard.iterations
        ):
            tails[region.symbol] = predecessors
            continue
        resolver = Resolver(region.static, values).materialize_constants()
        for tensor in region.tensors:
            actual = by_name[tensor.id]
            if (
                tuple(resolver.value(d) for d in tensor.shape) != actual.shape
                or tensor.dtype != actual.dtype
            ):
                raise ValueError(f"tensor binding mismatch: {tensor.id}")
        begin, cta_begin = len(nodes), len(ctas)
        NativeExpander(region, resolver, by_name, nodes, ctas).expand()
        local_ids = tuple(range(begin, len(nodes)))
        for identity in local_ids:
            node = nodes[identity]
            if not node.after_issue and not node.after_complete:
                nodes[identity] = replace(node, after_complete=predecessors)
        consumed = {
            dependency for n in nodes[begin:] for dependency in n.after_complete
        }
        tails[region.symbol] = tuple(n for n in local_ids if n not in consumed)
        outputs = tuple(
            dict.fromkeys(
                n.tensor
                for n in nodes[begin:]
                if n.opcode == A.NativeOpcode.TMA_STORE.value
            )
        )
        regions.append(
            A.ExecRegion(
                region.symbol,
                local_ids,
                tuple(range(cta_begin, len(ctas))),
                region.after,
                outputs,
            )
        )
        if target is not None and target != region.target:
            raise ValueError("one executable stream must use a single target")
        target = region.target
    result = A.ExecIR(
        arch.symbol,
        EMPTY,
        target,
        tuple(tensors),
        tuple(ctas),
        tuple(nodes),
        (),
        tuple(by_name[name].id for name in arch.outputs),
        tuple(regions),
        arch.stream,
    )
    for node in result.ops:
        if any(
            dep >= node.id or dep < 0
            for dep in (*node.after_issue, *node.after_complete)
        ):
            raise ValueError("execution DAG contains a forward/cyclic dependency")
    return result


def lower_region(arch, region):
    """Expand one region; the caller retains the enclosing stream's ordering."""
    return lower_arch(replace(arch, regions=(replace(region, after=()),), outputs=()))
