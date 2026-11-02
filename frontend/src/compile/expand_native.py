"""IR-05-only structured instruction iteration and explicit event resolution."""

from dataclasses import dataclass
from parser import ast as A
from .static import Resolver


def iterations(loops, resolver, values):
    if not loops:
        yield values
        return
    loop, *rest = loops
    current = Resolver(resolver.context, {**resolver.values, **values})
    for value in range(
        current.value(loop.start), current.value(loop.stop), current.value(loop.step)
    ):
        yield from iterations(rest, resolver, {**values, loop.variable: value})


@dataclass
class Event:
    pattern: str
    step: int | None
    cta: int
    values: dict
    executor: int | None
    nodes: tuple[int, ...]


class NativeExpander:
    def __init__(self, region, resolver, tensors, nodes, ctas):
        self.region, self.resolver, self.tensors = region, resolver, tensors
        self.nodes, self.ctas = nodes, ctas
        outer = tuple(loop.variable for loop in region.domain.loops)
        self.index_fields = {
            tuple(sorted(outer)),
            tuple(sorted((*outer, region.domain.pipeline.loop.variable))),
        }
        for rule in region.dependency_rules:
            keys = tuple(sorted(key for key, _ in rule.producer_bindings))
            self.index_fields.add(keys)
            self.index_fields.add(tuple(sorted((*keys, "executor"))))
        self.index = {}
        self.patterns = {op.id: op for op in region.ops}
        self.warp_bases = {}
        self.next_warp = sum(c.warps for c in ctas)
        for pattern in region.ops:
            for dependency in (
                *pattern.dependencies,
                *(d for step in pattern.steps for d in step.dependencies),
            ):
                if dependency.source not in self.patterns:
                    raise ValueError(
                        f"unknown native event producer: {dependency.source}"
                    )

    def resolver_at(self, values, cta, pattern):
        executor = values.get("executor")
        wave = self.region.domain.loops[1].variable

        def has_previous_fragment(warp):
            return bool(
                self.lookup(pattern.id, {wave: values[wave], "executor": warp}, cta)
            )

        return Resolver(
            self.resolver.context,
            {**self.resolver.values, **values, "scope": pattern.scope.value},
            {"has_previous_fragment": has_previous_fragment},
        )

    def producers(self, dependency, values, cta, pattern):
        resolver = self.resolver_at(values, cta, pattern)
        rules = [
            r
            for r in self.region.dependency_rules
            if r.id == dependency.label and dependency.source in r.sources
        ]
        matches = []
        if rules:
            for rule in rules:
                if not resolver.value(rule.static_if):
                    continue
                bindings = {
                    key: resolver.value(expr) for key, expr in rule.producer_bindings
                }
                if (
                    rule.selection == A.Selection.LAST_PER_EXECUTOR
                    and "executor" in values
                ):
                    bindings["executor"] = values["executor"]
                candidates = self.lookup(dependency.source, bindings, cta)
                if rule.selection == A.Selection.LAST_PER_EXECUTOR:
                    latest = {}
                    for event in candidates:
                        latest[event.executor] = event
                    candidates = list(latest.values())
                matches.extend(n for e in candidates for n in e.nodes)
        else:
            bindings = {
                loop.variable: values[loop.variable]
                for loop in self.region.domain.loops
            }
            if (
                pattern.scope
                == self.patterns[dependency.source].scope
                == A.Scope.PIPELINE
            ):
                variable = self.region.domain.pipeline.loop.variable
                bindings[variable] = values[variable]
            candidates = self.lookup(dependency.source, bindings, cta)
            if not candidates:
                raise ValueError(
                    f"native dependency has no earlier producer: {dependency.source}"
                )
            matches.extend(n for event in candidates for n in event.nodes)
        return matches

    def lookup(self, source, bindings, cta, step=None):
        pattern = self.patterns[source]
        final_step = len(pattern.steps) - 1 if pattern.steps else None
        selected = final_step if step is None else step
        fields = tuple(sorted(bindings))
        key = (cta, source, selected, fields, tuple(bindings[k] for k in fields))
        return self.index.get(key, ())

    def register(self, event):
        values = {**event.values, "executor": event.executor}
        for fields in self.index_fields:
            if all(field in values for field in fields):
                key = (
                    event.cta,
                    event.pattern,
                    event.step,
                    fields,
                    tuple(values[k] for k in fields),
                )
                self.index.setdefault(key, []).append(event)

    def dependencies(self, dependencies, values, cta, pattern):
        issue, complete = [], []
        for dependency in dependencies:
            target = issue if dependency.phase == A.Phase.ISSUE else complete
            target.extend(self.producers(dependency, values, cta, pattern))
        return issue, complete

    def pattern(self, pattern, values, cta):
        if not pattern.steps:
            _, complete = self.dependencies(pattern.dependencies, values, cta, pattern)
            self.register(Event(pattern.id, None, cta, values, None, tuple(complete)))
            return
        for step_id, step in enumerate(pattern.steps):
            for point in iterations(step.loops, self.resolver, values):
                resolver = self.resolver_at(point, cta, pattern)
                executor = resolver.value(step.executor.expression)
                point = {**point, "executor": executor}
                issue, complete = self.dependencies(
                    step.dependencies, point, cta, pattern
                )
                outer = {
                    loop.variable: point[loop.variable]
                    for loop in self.region.domain.loops
                }
                if pattern.scope == A.Scope.PIPELINE:
                    variable = self.region.domain.pipeline.loop.variable
                    outer[variable] = point[variable]
                for previous in step.input_steps:
                    events = self.lookup(pattern.id, outer, cta, previous)
                    if not events:
                        raise ValueError(
                            "native input step is missing or forward-referenced"
                        )
                    complete.extend(n for e in events for n in e.nodes)
                tensor = self.tensors[step.region.tensor]
                offsets = tuple(
                    resolver.value(v.expression) for v in step.region.offsets
                )
                shape = tuple(resolver.value(v) for v in step.region.shape)
                if len(shape) != 2 or any(
                    off < 0 or off + length > limit
                    for off, length, limit in zip(offsets, shape, tensor.shape)
                ):
                    raise ValueError(
                        f"native memory/tensor region is out of bounds: {step.region.tensor}"
                    )
                identity = len(self.nodes)
                suffix = "_".join(
                    f"{k}{v}" for k, v in point.items() if k != "executor"
                )
                suffix += "_w" + str(self.warp_bases[cta] + executor)
                self.nodes.append(
                    A.ArchInstruction(
                        identity,
                        f"{self.region.symbol}_{pattern.id}_{step_id}_{suffix}",
                        step.opcode.value,
                        step.resource,
                        cta,
                        self.warp_bases[cta] + executor,
                        tensor.id,
                        (*offsets, *shape),
                        step.dtype,
                        resolver.value(step.leading_dim.expression)
                        if step.leading_dim
                        else None,
                        tuple(dict.fromkeys(issue)),
                        tuple(dict.fromkeys(complete)),
                    )
                )
                self.register(
                    Event(pattern.id, step_id, cta, point, executor, (identity,))
                )

    def expand(self):
        cta_ids = {}
        pipeline = self.region.domain.pipeline.loop
        for point in iterations(self.region.domain.loops, self.resolver, {}):
            resolver = Resolver(
                self.resolver.context, {**self.resolver.values, **point}
            )
            if any(
                guard.iterations and not resolver.value(guard.condition)
                for guard in self.region.static_ifs
            ):
                continue
            local_cta = point[self.region.cta_variable]
            if local_cta not in cta_ids:
                identity = len(self.ctas)
                cta_ids[local_cta] = identity
                self.warp_bases[identity] = self.next_warp
                self.next_warp += self.region.resources.warps
                resources = self.region.resources
                self.ctas.append(
                    A.CTA(
                        identity,
                        resolver.value(self.region.sm.expression),
                        resources.warps,
                        resources.shared_bytes,
                        resources.register_bytes,
                    )
                )
            cta = cta_ids[local_cta]
            for pattern in self.region.ops:
                if pattern.scope == A.Scope.PROLOGUE:
                    self.pattern(pattern, point, cta)
            for iteration in iterations((pipeline,), self.resolver, point):
                for pattern in self.region.ops:
                    if pattern.scope == A.Scope.PIPELINE:
                        self.pattern(pattern, iteration, cta)
            for pattern in self.region.ops:
                if pattern.scope == A.Scope.EPILOGUE:
                    self.pattern(pattern, point, cta)
