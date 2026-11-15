"""FlashAccel's compact N-slab partition pass.

The pass changes only region metadata.  Tensor declarations remain the
program's original declarations, so a partition is a view into one physical
weight allocation rather than another allocation in the memory registry.
"""

from dataclasses import dataclass, replace
import json
from pathlib import Path

from parser import ast as A
from ..static import EMPTY, Resolver, bound_resolver


@dataclass(frozen=True)
class TensorPartitionView:
    parent: str
    row: int
    col: int
    rows: int
    cols: int
    parent_row_stride: int


@dataclass(frozen=True)
class PartitionPolicy:
    partition_policy: str = "n_slab_fit_double_buffer"
    buffer_count: int = 2
    bm: int = 16
    bn: int = 64
    bk: int = 16
    stages: int = 2
    split: int = 1

    @classmethod
    def from_json(cls, path):
        values = json.loads(Path(path).read_text())
        tiling = values.get("gemm_tiling", {})
        result = cls(
            values.get("partition_policy", cls.partition_policy),
            int(values.get("buffer_count", cls.buffer_count)),
            int(tiling.get("bm", cls.bm)),
            int(tiling.get("bn", cls.bn)),
            int(tiling.get("bk", cls.bk)),
            int(tiling.get("stages", cls.stages)),
            int(tiling.get("split", cls.split)),
        )
        if result.partition_policy != "n_slab_fit_double_buffer":
            raise ValueError("unsupported FlashAccel partition policy")
        if result.buffer_count != 2 or min(
            result.bm, result.bn, result.bk, result.stages, result.split
        ) <= 0:
            raise ValueError("FlashAccel policy requires positive double-buffer tiling")
        if (result.bm, result.bn, result.bk, result.stages, result.split) != (
            16,
            64,
            16,
            2,
            1,
        ):
            raise ValueError(
                "FlashAccel uses the validated small-M 16x64x16, stages=2 schedule"
            )
        return result


@dataclass(frozen=True)
class PhysicalLayout:
    stacks: int
    dies_per_stack: int
    planes_per_die: int
    planes_per_command_group: int
    page_bytes: int
    sram_bytes_per_plane: int
    base_die_sram_bytes_per_stack: int
    hbf_base: int = 0
    address_mapping_policy: str = "megapage_round_robin"
    align_tensors_to_command_group: bool = True

    @classmethod
    def from_memory_yaml(cls, path):
        values = _yaml_scalars(Path(path))
        prefix = "hbf.controller."
        return cls(
            int(values[prefix + "stacks"]),
            int(values[prefix + "dies_per_stack"]),
            int(values[prefix + "planes_per_die"]),
            int(values[prefix + "planes_per_command_group"]),
            int(values.get(prefix + "page_bytes", 4096)),
            int(values.get(prefix + "sram_bytes_per_plane", 8192)),
            int(values.get(prefix + "base_die_sram_bytes_per_stack", 0)),
            int(values.get("address_layout.hbf_base", 0)),
            values.get("hbf.address_mapping_policy", "megapage_round_robin"),
            bool(values.get("hbf.align_tensors_to_command_group", True)),
        )

    @property
    def group_bytes(self):
        return self.page_bytes * self.planes_per_command_group

    @property
    def plane_bytes(self):
        return self.sram_bytes_per_plane

    @property
    def stack_bytes(self):
        return self.base_die_sram_bytes_per_stack

    def location(self, page):
        """Return the same (stack, plane) mapping used by grouped HBFine."""
        if page < 0:
            raise ValueError("negative physical page")
        lanes = self.planes_per_command_group
        if self.address_mapping_policy != "megapage_round_robin":
            raise ValueError("only megapage_round_robin is implemented by HBF backends")
        if lanes <= 0 or self.planes_per_die % lanes:
            raise ValueError("invalid grouped-plane geometry")
        group, lane = divmod(page, lanes)
        group %= self.stacks * self.dies_per_stack * (
            self.planes_per_die // lanes
        )
        stack = group % self.stacks
        die_group = group // self.stacks
        die = die_group % self.dies_per_stack
        group_in_die = die_group // self.dies_per_stack
        return stack, die * self.planes_per_die + group_in_die * lanes + lane

    def bases(self, tensors, values):
        """Mirror HBFMegapageAllocator's tensor-order placement cursor."""
        cursor = 0
        result = {}
        for tensor in tensors:
            if tensor.tier != A.MemorySpace.HBF:
                continue
            shape = tuple(values.value(d) for d in tensor.shape)
            size = 1
            for extent in shape:
                size *= extent
            size *= 2 if tensor.dtype in (A.DataType.BF16, A.DataType.FP16) else 4
            alignment = self.group_bytes if self.align_tensors_to_command_group else self.page_bytes
            cursor = (cursor + alignment - 1) // alignment * alignment
            result[tensor.id] = cursor
            cursor += (size + alignment - 1) // alignment * alignment
        return result

    def pages_for_rows(self, rows, cols, element_bytes, row, col, height, width):
        pages = set()
        for current_row in range(row, row + height):
            begin = (current_row * cols + col) * element_bytes
            end = begin + width * element_bytes
            pages.update(
                range(begin // self.page_bytes, (end - 1) // self.page_bytes + 1)
            )
        return pages


@dataclass(frozen=True)
class PartitionResult:
    flow: A.FlowIR
    origin_launches: dict[str, tuple[str, ...]]
    views: tuple[TensorPartitionView, ...]


def _yaml_scalars(path):
    """Read scalar paths from the small YAML subset used by memory profiles."""
    values, stack = {}, []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip())
        key, value = (item.strip() for item in line.split(":", 1))
        if key.startswith("-"):
            continue
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path_key = ".".join([item[1] for item in stack] + [key])
        if not value:
            stack.append((indent, key))
            continue
        if value.lower() in ("true", "false"):
            parsed = value.lower() == "true"
        else:
            try:
                parsed = int(value, 0)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value.strip("'\"")
        values[path_key] = parsed
    return values


def memory_topology(path):
    return _yaml_scalars(Path(path)).get("topology", "FullHBM")


def _resolver(flow):
    root = bound_resolver(flow.static, Resolver(EMPTY, {})).materialize_constants()
    return root


def _region_resolver(region, root):
    arguments = {arg.id: root.value(arg.expression) for arg in region.static.arguments}
    return Resolver(
        replace(region.static, parameters=(), constraints=()),
        {**root.values, **arguments},
    ).materialize_constants()


def _argument(resolver, region, name):
    binding = next((item for item in region.static.arguments if item.id == name), None)
    if binding is None:
        return None
    return resolver.value(binding.expression)


def _element_bytes(tensor):
    return 2 if tensor.dtype in (A.DataType.BF16, A.DataType.FP16) else 4


def _weight_rows(region, resolver, tensor, revealed_ids=None):
    k = _argument(resolver, region, "K")
    if k is None:
        return [(0, resolver.value(tensor.shape[0]))]
    # Expert IDs are unknown until the router kernel completes.  The
    # conservative pre-reveal footprint is the complete parent tensor.  A
    # trace-derived list may be supplied only by the post-router child pass.
    if revealed_ids is None:
        return [(0, resolver.value(tensor.shape[0]))]
    return [(int(expert) * k, k) for expert in sorted(set(revealed_ids))]


def _gemm_weight(region, global_tensors):
    refs = {op.region.tensor for op in region.ops if op.region is not None}
    candidates = [
        global_tensors[name]
        for name in refs
        if name in global_tensors
        and global_tensors[name].tier == A.MemorySpace.HBF
        and global_tensors[name].external
        and global_tensors[name].read_only
        and len(global_tensors[name].shape) == 2
    ]
    if len(candidates) != 1 or not any(
        op.kind == A.OpKind.MATRIX for op in region.ops
    ):
        return None
    return candidates[0]


def _shift(region, names, offset, parent_row_stride):
    if region.tensor not in names:
        return region
    offsets = list(region.offsets)
    if len(offsets) < 2:
        return region
    if offset:
        offsets[1] = A.IndexExpr(f"({offsets[1].expression})+{offset}")
    # The child shape is a logical view.  Keep the physical row stride of the
    # parent tensor for both TMA loads and stores.
    return replace(
        region,
        offsets=tuple(offsets),
        parent_row_stride=region.parent_row_stride or A.IndexExpr(str(parent_row_stride)),
    )


def _partition_region(
    region, tensor, resolver, layout, policy, bases, global_tensors,
    revealed_ids=None,
):
    n = _argument(resolver, region, "N")
    if n is None:
        return (region,), (), False
    shape = tuple(resolver.value(d) for d in tensor.shape)
    rows, parent_cols = shape
    if n > parent_cols:
        raise ValueError(f"{region.symbol}: GEMM N exceeds parent weight width")
    k = _argument(resolver, region, "K") or rows
    row_ranges = _weight_rows(region, resolver, tensor, revealed_ids)
    element_bytes = _element_bytes(tensor)
    base_page = bases[tensor.id] // layout.page_bytes
    def footprint(col, width):
        # All Expert rows which can be live in this N-slab share one SRAM
        # reservation.  Capacity is therefore checked against their union,
        # not one expert at a time.
        return {
            (tensor.id, base_page + page)
            for row, height in row_ranges
            for page in layout.pages_for_rows(
                rows, parent_cols, element_bytes, row, col, height, width
            )
        }

    def fits(col, width):
        pages = footprint(col, width)
        plane_pages = layout.sram_bytes_per_plane // layout.page_bytes
        stack_limit = layout.base_die_sram_bytes_per_stack // policy.buffer_count
        # ReadSramAllocator assigns whole pages to the plane first, using the
        # selected double-buffer slot's round-robin remainder, then spills to
        # the base die.  Check every slot: a reservation is legal only if the
        # same N-slab can be admitted in either slot selected by the epoch.
        for slot in range(policy.buffer_count):
            plane_used, stack_used = {}, {}
            for _, page in pages:
                stack, plane = layout.location(page)
                physical_plane = (
                    stack * layout.dies_per_stack * layout.planes_per_die + plane
                )
                extra = int(
                    (slot + policy.buffer_count - physical_plane % policy.buffer_count)
                    % policy.buffer_count
                    < plane_pages % policy.buffer_count
                )
                plane_limit = plane_pages // policy.buffer_count + extra
                key = (stack, physical_plane)
                plane_used[key] = plane_used.get(key, 0) + 1
                if plane_used[key] > plane_limit:
                    stack_used[stack] = stack_used.get(stack, 0) + 1
            if any(
                value > stack_limit // layout.page_bytes
                for value in stack_used.values()
            ):
                return False
        return True

    if n <= policy.bn:
        if not fits(0, n):
            raise ValueError(f"{region.symbol}: one GEMM BN does not fit SRAM slot")
        return (region,), (), False

    widths = []
    offset = 0
    while offset < n:
        remaining = n - offset
        width = min(policy.bn, remaining)
        if not fits(offset, width):
            raise ValueError(f"{region.symbol}: one GEMM BN does not fit SRAM slot")
        # Keep the paper's greedy largest-fitting N slab, but test the actual
        # physical column offset and the full Expert page union for every
        # candidate.  The final tail is allowed only when its own footprint
        # fits; no K-split or tiling search is introduced.
        candidate = width
        for probe in range(policy.bn, remaining + 1, policy.bn):
            if fits(offset, probe):
                candidate = probe
        width = candidate
        widths.append((offset, width))
        offset += width

    weight_names = {tensor.id}
    output_names = {
        op.region.tensor
        for op in region.ops
        if op.region is not None
        and op.kind in (A.OpKind.MATRIX, A.OpKind.STORE)
        and op.region.tensor not in weight_names
    }
    row_strides = {
        name: resolver.value(spec.shape[1])
        for name, spec in global_tensors.items()
        if name in weight_names | output_names
    }
    children, views = [], []
    for index, (col, width) in enumerate(widths):
        arguments = tuple(
            replace(item, expression=str(width)) if item.id == "N" else item
            for item in region.static.arguments
        )
        constraints = tuple(
            item for item in region.static.constraints if "N divisible_by" not in item
        )
        ops = tuple(
            replace(
                op,
                region=(
                    _shift(
                        op.region,
                        weight_names | output_names,
                        col,
                        row_strides.get(op.region.tensor, parent_cols),
                    )
                    if op.region is not None
                    else None
                ),
            )
            for op in region.ops
        )
        child = replace(
            region,
            symbol=f"{region.symbol}__part{index}",
            static=replace(region.static, arguments=arguments, constraints=constraints),
            ops=ops,
        )
        children.append(child)
        for row, height in row_ranges:
            views.append(
                TensorPartitionView(
                    tensor.id, row, col, height, width, parent_cols
                )
            )
    return tuple(children), tuple(views), True


def _split_selected_experts(region, resolver):
    """Expose one B=1 direct expert at a time for real SRAM reuse.

    The grouped/direct Qwen B=1 schedule carries one assignment per routed
    expert.  Keeping all expert rows in one reservation would make the
    double-buffer capacity check intentionally fail; splitting the existing
    legal assignment list gives each child a real QueuePrefetch/Release
    lifetime.  No routing prediction is introduced: the list is still only a
    trace used when the router result is revealed by the runtime barrier.
    """
    if not region.symbol.startswith("expert_gemm"):
        return ((region, None),)
    if not _argument(resolver, region, "__program_guard"):
        return ((region, None),)
    ids = _argument(resolver, region, "IDS")
    amap = _argument(resolver, region, "A_MAP")
    cmap = _argument(resolver, region, "C_MAP")
    if not ids or amap is None or cmap is None:
        return ((region, None),)
    groups = []
    for index, expert in enumerate(ids):
        group = next((item for item in groups if item[0] == expert), None)
        if group is None:
            group = [expert, []]
            groups.append(group)
        group[1].append(index)
    if len(groups) <= 1:
        return ((region, tuple(ids)),)
    children = []
    for expert, indices in groups:
        values = {
            "M": len(indices),
            "IDS": [ids[index] for index in indices],
            "A_MAP": [amap[index] for index in indices],
            "C_MAP": [cmap[index] for index in indices],
        }
        arguments = tuple(
            replace(item, expression=repr(values[item.id]))
            if item.id in values
            else item
            for item in region.static.arguments
        )
        children.append(
            (
                replace(
                    region,
                    symbol=f"{region.symbol}__expert{expert}",
                    static=replace(region.static, arguments=arguments),
                ),
                tuple(ids[index] for index in indices),
            )
        )
    return tuple(children)


def partition_flow(flow, policy: PartitionPolicy, layout: PhysicalLayout):
    if policy.partition_policy != "n_slab_fit_double_buffer":
        raise ValueError("unsupported FlashAccel partition policy")
    root = _resolver(flow)
    global_tensors = {tensor.id: tensor for tensor in flow.tensors}
    bases = layout.bases(flow.tensors, root)
    origin_launches = {}
    views = []
    transformed = []
    for region in flow.regions:
        origin_launches[region.symbol] = (region.symbol,)
        weight = _gemm_weight(region, global_tensors)
        if weight is None:
            transformed.append(region)
            continue
        resolver = _region_resolver(region, root)
        if _argument(resolver, region, "__program_guard") is False:
            transformed.append(region)
            continue
        expert_regions = _split_selected_experts(region, resolver)
        children, child_views, changed = [], [], False
        for expert_region, revealed_ids in expert_regions:
            expert_resolver = _region_resolver(expert_region, root)
            parts, expert_views, expert_changed = _partition_region(
                expert_region,
                weight,
                expert_resolver,
                layout,
                policy,
                bases,
                global_tensors,
                revealed_ids,
            )
            children.extend(parts)
            child_views.extend(expert_views)
            changed |= expert_changed or expert_region is not region
        if not changed:
            transformed.append(region)
            continue
        origin_launches[region.symbol] = tuple(child.symbol for child in children)
        views.extend(child_views)
        transformed.extend(children)

    # Rebuild dependency tails after expansion.  Every original dependency is
    # represented by the final child of that launch; children are serialized
    # in their original order.
    expanded = []
    tails = {}
    for region in flow.regions:
        children = tuple(
            item for item in transformed
            if item.symbol in origin_launches.get(region.symbol, (region.symbol,))
        )
        if not children:
            continue
        after = tuple(
            tails.get(name, origin_launches.get(name, (name,))[-1])
            for name in region.after
        )
        for child in children:
            child_after = after if child is children[0] else (expanded[-1].symbol,)
            expanded.append(replace(child, after=child_after))
        tails[region.symbol] = children[-1].symbol
    return PartitionResult(
        replace(flow, regions=tuple(expanded)), origin_launches, tuple(views)
    )
