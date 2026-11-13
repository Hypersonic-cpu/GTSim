"""Capacity-aware partitioning and prefetch planning on structured FlowIR."""

import ast
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import groupby

from parser import ast as A
from ..static import EMPTY, Resolver, bound_resolver, iterations, parse_expression


@dataclass(frozen=True)
class Residency:
    stacks: int
    dies_per_stack: int
    planes_per_die: int
    planes_per_group: int
    page_bytes: int
    plane_bytes: int
    stack_bytes: int
    line_bytes: int = 128
    align_tensors_to_command_group: bool = True
    eligible_tensors: tuple[str, ...] | None = None
    buffer_count: int = 1
    streaming: bool = False

    def __post_init__(self):
        if (
            min(
                self.stacks,
                self.dies_per_stack,
                self.planes_per_die,
                self.planes_per_group,
                self.page_bytes,
                self.line_bytes,
                self.buffer_count,
            )
            <= 0
            or self.planes_per_die % self.planes_per_group
            or self.page_bytes % self.line_bytes
            or min(self.plane_bytes, self.stack_bytes) < 0
            or not (self.plane_bytes or self.stack_bytes)
        ):
            raise ValueError("invalid residency geometry/capacity")

    @property
    def group_bytes(self):
        return self.page_bytes * self.planes_per_group

    def location(self, page):
        group, lane = divmod(page, self.planes_per_group)
        stack = group % self.stacks
        die_group = group // self.stacks
        die = die_group % self.dies_per_stack
        group_in_die = (
            die_group
            // self.dies_per_stack
            % (self.planes_per_die // self.planes_per_group)
        )
        return (
            stack,
            die * self.planes_per_die + group_in_die * self.planes_per_group + lane,
        )


class PageBudget:
    def __init__(self, config, bases, slot=None):
        self.config, self.bases, self.slot = config, bases, slot
        self.pages = set()
        self.planes, self.spill = Counter(), Counter()

    def add(self, pages):
        added = pages - self.pages
        c = self.config
        counts = Counter(
            c.location(self.bases[tensor] // c.page_bytes + page)
            for tensor, page in added
        )
        plane_pages = c.plane_bytes // c.page_bytes
        stack_limit = c.stack_bytes // c.page_bytes
        if self.slot is not None:
            stack_limit //= c.buffer_count
        spill = Counter()
        for key, count in counts.items():
            plane_limit = plane_pages
            if self.slot is not None:
                whole, extra = divmod(plane_pages, c.buffer_count)
                # Whole physical pages are distributed evenly between slots.
                plane_limit = whole + ((self.slot - key[1]) % c.buffer_count < extra)
            old = self.planes[key]
            spill[key[0]] += max(0, old + count - plane_limit) - max(
                0, old - plane_limit
            )
        if any(
            self.spill[stack] + count > stack_limit for stack, count in spill.items()
        ):
            return False
        self.pages.update(added)
        self.planes.update(counts)
        self.spill.update(spill)
        return True


@dataclass(frozen=True)
class Span:
    tensor: str
    offset: int
    bytes: int


@dataclass(frozen=True)
class BufferSlot:
    slot: int
    count: int
    epoch: int


@dataclass(frozen=True)
class Reservation:
    id: str
    span: Span
    enqueue_at: int
    consumers: tuple[str, ...]
    buffer: BufferSlot | None = None


@dataclass(frozen=True)
class PrefetchPlan:
    flow: A.FlowIR
    queue: tuple[Reservation, ...]
    streaming: Residency | None = None
    bases: dict[str, int] | None = None

    def decorate(self, manifest):
        """Lower the side schedule at the normal Program export boundary."""
        if [r.symbol for r in self.flow.regions] != [
            x["id"] for x in manifest["launches"]
        ]:
            raise ValueError("runtime launch order differs from the prefetch plan")
        ids = {t["name"]: t["id"] for t in manifest["tensors"]}
        users, releases, enqueue = (
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )
        for item in self.queue:
            enqueue[item.enqueue_at].append(item.id)
            releases[item.consumers[-1]].append(item.id)
            for launch in item.consumers:
                users[launch].append(item.id)
        manifest["prefetches"] = [
            dict(
                id=p.id,
                tensor=ids[p.span.tensor],
                offset=p.span.offset,
                bytes=p.span.bytes,
                **({"buffer": asdict(p.buffer)} if p.buffer else {}),
            )
            for p in self.queue
        ]
        for launch in manifest["launches"]:
            if launch["id"] in users:
                launch["read_aliases"] = users[launch["id"]]

        # Keep the source ProgramIR command stream when one is available.  In
        # particular, ``prog_to_flow`` consumes explicit synchronizations and
        # events while constructing FlowIR, so the runtime sidecar is the only
        # place where those commands can survive.  Prefetch operations are
        # inserted immediately around the corresponding launch without
        # reordering or dropping the original host commands.
        source_commands = manifest.get("commands")
        if source_commands:
            launch_index = {
                launch["id"]: i for i, launch in enumerate(manifest["launches"])
            }
            commands = []
            seen_launches = set()
            for command in source_commands:
                if command.get("kind") != "LaunchKernel":
                    commands.append(command)
                    continue
                name = command.get("target")
                if name not in launch_index or name in seen_launches:
                    raise ValueError("runtime launch order differs from the prefetch plan")
                index = launch_index[name]
                commands.extend(
                    dict(record="Command", kind="QueuePrefetch", target=p)
                    for p in enqueue[index]
                )
                commands.append(command)
                commands.extend(
                    dict(record="Command", kind="Release", target=p)
                    for p in releases[name]
                )
                seen_launches.add(name)
            if seen_launches != set(launch_index):
                raise ValueError("runtime command stream omits a planned launch")
        else:
            commands = []
            for i, launch in enumerate(manifest["launches"]):
                name = launch["id"]
                commands.extend(
                    dict(record="Command", kind="QueuePrefetch", target=p)
                    for p in enqueue[i]
                )
                commands.append(dict(record="Command", kind="LaunchKernel", target=name))
                commands.extend(
                    dict(record="Command", kind="Release", target=p) for p in releases[name]
                )
            commands.append(dict(record="Command", kind="StreamSynchronize", target=""))
        manifest["commands"] = commands
        return manifest


def scope(region, root):
    arguments = {a.id: root.value(a.expression) for a in region.static.arguments}
    values = {**root.values, **arguments}
    probe = Resolver(replace(region.static, parameters=(), constraints=()), values)
    if any(not probe.value(g.condition) for g in region.static_ifs if not g.iterations):
        return None
    return Resolver(region.static, values).materialize_constants()


def access_tiles(region, resolver, operations):
    """Enumerate memory tiles only. No instruction selection or DAG expansion."""
    pipeline = region.domain.pipeline.loop
    for outer in iterations(region.domain.loops, resolver, {}):
        point = resolver.at(outer)
        if any(not point.value(g.condition) for g in region.static_ifs if g.iterations):
            continue
        point = point.materialize_constants()
        for op in operations:
            points = (
                (outer,)
                if op.scope != A.Scope.PIPELINE
                else iterations((pipeline,), point, outer)
            )
            for inner in points:
                for lane in iterations(op.iterations, point, inner):
                    at = point.at({**lane, "scope": op.scope.value})
                    if not at.value(op.guard):
                        continue
                    r, c = (at.value(v.expression) for v in op.region.offsets)
                    h, w = (at.value(v) for v in op.region.shape)
                    yield (
                        outer[region.domain.loops[0].variable],
                        op.region.tensor,
                        r,
                        c,
                        h,
                        w,
                    )


def eligible_loads(region, eligible):
    return tuple(
        op
        for op in region.ops
        if op.kind == A.OpKind.COPY
        and op.region is not None
        and op.region.tensor in eligible
    )


def trace_dependent(region, loads, root):
    # List parameters can encode decisions from a supplied trace. Only gate
    # loads whose addresses or iteration domain transitively use such inputs.
    if not loads:
        return False
    list_types = (A.DataType.INDEX_LIST, A.DataType.FLOAT_LIST)
    trace = {
        p.id
        for static in (root, region.static)
        for p in static.parameters
        if p.dtype in list_types
    }
    definitions = {}
    for static in (root, region.static):
        for binding in (*static.arguments, *static.bindings, *static.predicates):
            definitions.setdefault(binding.id, []).append(binding.expression)
    loops = list(region.domain.loops)
    if any(op.scope == A.Scope.PIPELINE for op in loads):
        loops.append(region.domain.pipeline.loop)
    expressions = [g.condition for g in region.static_ifs]
    for op in loads:
        loops.extend(op.iterations)
        expressions.extend(
            (op.guard, *op.region.shape, *(v.expression for v in op.region.offsets))
        )
    expressions.extend(v for loop in loops for v in (loop.start, loop.stop, loop.step))
    seen = set()
    while expressions:
        expr = expressions.pop()
        if not isinstance(expr, str):
            continue
        for node in ast.walk(parse_expression(expr)):
            if not isinstance(node, ast.Name) or node.id in seen:
                continue
            if node.id in trace:
                return True
            seen.add(node.id)
            expressions.extend(definitions.get(node.id, ()))
    return False


def split_region(region, resolver, shapes, widths, loads, bases, config, layouts):
    if not loads:
        return [(region, frozenset())]
    footprint = defaultdict(set)
    for cta, tensor, r, c, h, w in access_tiles(region, resolver, loads):
        rows, cols = shapes[tensor]
        if not (0 <= r < r + h <= rows and 0 <= c < c + w <= cols):
            raise ValueError(f"out-of-bounds prefetch input in {region.symbol}")
        size = widths[tensor]
        if layout := layouts.get(tensor):
            if layout.tile_k * layout.tile_n * size != config.page_bytes:
                raise ValueError(f"memory tile must occupy exactly one page: {tensor}")
            footprint[cta].update(
                (tensor, page) for page in layout.pages(cols, r, c, h, w)
            )
            continue
        for row in range(r, r + h):
            begin = (row * cols + c) * size
            end = begin + w * size
            footprint[cta].update(
                (tensor, p)
                for p in range(
                    begin // config.page_bytes, (end - 1) // config.page_bytes + 1
                )
            )
    pages = set().union(*footprint.values())
    if PageBudget(config, bases).add(pages):
        return [(region, frozenset(pages))]

    loop, *outer = region.domain.loops
    start, stop, step = (resolver.value(v) for v in (loop.start, loop.stop, loop.step))
    chunks = []
    first, budget = start, PageBudget(config, bases)
    for cta in range(start, stop, step):
        if budget.add(footprint[cta]):
            continue
        if not budget.pages:
            raise ValueError(
                f"{region.symbol}: a full-reduction CTA exceeds SRAM capacity"
            )
        chunks.append((first, cta, frozenset(budget.pages)))
        first, budget = cta, PageBudget(config, bases)
        if not budget.add(footprint[cta]):
            raise ValueError(
                f"{region.symbol}: a full-reduction CTA exceeds SRAM capacity"
            )
    chunks.append((first, stop, frozenset(budget.pages)))

    # CTA-local reductions can only be split if stores across slices are disjoint.
    stores = tuple(
        op for op in region.ops if op.kind == A.OpKind.STORE and op.region is not None
    )
    if not stores or any(op.scope == A.Scope.PIPELINE for op in stores):
        raise ValueError(f"{region.symbol}: no independent epilogue output tiles")
    destinations = defaultdict(list)
    owner = {
        cta: i for i, (lo, hi, _) in enumerate(chunks) for cta in range(lo, hi, step)
    }
    for cta, tensor, r, c, h, w in access_tiles(region, resolver, stores):
        rows, cols = shapes[tensor]
        if not (0 <= r < r + h <= rows and 0 <= c < c + w <= cols):
            raise ValueError(f"out-of-bounds output in {region.symbol}")
        for row in range(r, r + h):
            destinations[tensor, row].append((c, c + w, owner[cta]))
    # Reading a destination could carry state across otherwise disjoint CTA stores.
    outputs = {key[0] for key in destinations}
    if not outputs or any(
        op.kind == A.OpKind.COPY and op.region and op.region.tensor in outputs
        for op in region.ops
    ):
        raise ValueError(f"{region.symbol}: output aliases a scheduled input")
    for intervals in destinations.values():
        end, last_owner = 0, None
        for lo, hi, current in sorted(intervals):
            if lo < end and current != last_owner:
                raise ValueError(
                    f"{region.symbol}: CTA slices have overlapping outputs"
                )
            if hi > end:
                end, last_owner = hi, current
    return [
        (
            replace(
                region,
                symbol=f"{region.symbol}__tile{i}",
                domain=replace(
                    region.domain, loops=(replace(loop, start=lo, stop=hi), *outer)
                ),
            ),
            footprint,
        )
        for i, (lo, hi, footprint) in enumerate(chunks)
    ]


def spans(pages, sizes, config):
    result = []
    for name, group in groupby(sorted(pages), key=lambda x: x[0]):
        begin = end = None
        for _, page in group:
            offset = page * config.page_bytes
            if end is not None and offset != end:
                result.append(Span(name, begin, end - begin))
                begin = None
            if begin is None:
                begin = offset
            end = min(offset + config.page_bytes, sizes[name])
        if begin is not None:
            result.append(Span(name, begin, end - begin))
    return tuple(result)


def optimize(flow, config, layouts=None):
    layouts = layouts or {}
    root = bound_resolver(flow.static, Resolver(EMPTY, {})).materialize_constants()
    shapes = {t.id: tuple(root.value(d) for d in t.shape) for t in flow.tensors}
    widths = {
        t.id: 2 if t.dtype in (A.DataType.FP16, A.DataType.BF16) else 4
        for t in flow.tensors
    }
    if any(len(dims) != 2 or min(dims) <= 0 for dims in shapes.values()):
        raise ValueError("prefetch planning requires positive two-dimensional tensors")
    selected = None if config.eligible_tensors is None else set(config.eligible_tensors)
    sizes = {name: dims[0] * dims[1] * widths[name] for name, dims in shapes.items()}
    eligible, bases, cursor = set(), {}, 0
    for tensor in flow.tensors:
        if tensor.tier != A.MemorySpace.HBF:
            continue
        bases[tensor.id] = cursor
        size = sizes[tensor.id]
        alignment = (
            config.group_bytes
            if config.align_tensors_to_command_group
            else config.page_bytes
        )
        cursor += ((size + alignment - 1) // alignment) * alignment
        if (
            tensor.external
            and tensor.read_only
            and (selected is None or tensor.id in selected)
        ):
            if size % config.line_bytes:
                raise ValueError(f"unaligned prefetch tensor: {tensor.id}")
            eligible.add(tensor.id)
    if selected is not None and selected - eligible:
        raise ValueError(f"invalid prefetch eligibility: {sorted(selected - eligible)}")
    # Partition both per-plane and per-stack SRAM. No extra storage is added.
    budget = replace(
        config,
        plane_bytes=(config.plane_bytes // config.buffer_count // config.page_bytes)
        * config.page_bytes,
        stack_bytes=(config.stack_bytes // config.buffer_count // config.page_bytes)
        * config.page_bytes,
        buffer_count=1,
    )
    regions, uses, gates, tails = [], [], [], {}
    for region in flow.regions:
        if any(name not in tails for name in region.after):
            raise ValueError(f"non-topological dependency in {region.symbol}")
        after = tuple(dict.fromkeys(t for name in region.after for t in tails[name]))
        resolver = scope(region, root)
        if resolver is None:
            tails[region.symbol] = after
            continue
        loads = eligible_loads(region, eligible)
        gate = len(regions) if trace_dependent(region, loads, flow.static) else 0
        parts = (
            [(region, frozenset())]
            if config.streaming
            else split_region(
                region, resolver, shapes, widths, loads, bases, budget, layouts
            )
        )
        for part, pages in parts:
            regions.append(replace(part, after=after))
            uses.append(spans(pages, sizes, config))
            gates.append(gate)
            after = (part.symbol,)
        tails[region.symbol] = after
    if len({r.symbol for r in regions}) != len(regions):
        raise ValueError("duplicate transformed launch ID")
    queue, boundary, i = [], 0, 0
    reuse_after = [0] * config.buffer_count
    epoch = 0
    while i < len(regions):
        end = i + 1
        while end < len(regions) and uses[end] == uses[i] and gates[end] == gates[i]:
            end += 1
        if uses[i]:
            boundary = max(boundary, gates[i])
            consumers = tuple(r.symbol for r in regions[i:end])
            buffer = None
            enqueue_at = boundary
            if config.buffer_count > 1:
                slot = epoch % config.buffer_count
                buffer = BufferSlot(slot, config.buffer_count, epoch)
                enqueue_at = max(boundary, reuse_after[slot])
                reuse_after[slot] = end
                epoch += 1
            queue.extend(
                Reservation(
                    f"prefetch_{len(queue)}", span, enqueue_at, consumers, buffer
                )
                for span in uses[i]
            )
        i = end
    return PrefetchPlan(
        replace(flow, regions=tuple(regions)),
        tuple(queue),
        config if config.streaming else None,
        bases if config.streaming else None,
    )
