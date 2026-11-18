"""Specialize logical weight rectangles into disjoint, shared NAND spans."""

from collections import Counter
from dataclasses import dataclass
from itertools import groupby

from .q_projection import Infeasible


@dataclass(frozen=True)
class PrefetchRegions:
    distance: int
    tensor: int
    shape: tuple[int, int]
    tile: tuple[int, int]
    page_bytes: int
    regions: tuple[tuple[int, ...], ...]
    bounds: tuple[tuple[int, int, int, int], ...]
    spans: tuple[tuple[int, int], ...]
    page_span: tuple[int, ...]
    region_ids: dict[tuple[int, int], int]
    summary: dict

    def consumer(self, node):
        r, c, h, w = node.tile
        ph, pw = self.tile
        regions = sorted(self.region_ids[y, x]
                         for y in range(r // ph, (r + h - 1) // ph + 1)
                         for x in range(c // pw, (c + w - 1) // pw + 1))
        stride = self.shape[1] * 2
        spans = set()
        for row in range(r, r + h):
            first = (row * stride + c * 2) // self.page_bytes
            last = (row * stride + (c + w) * 2 - 1) // self.page_bytes
            spans.update(self.page_span[first:last + 1])
        return dict(node=node.id, regions=regions, spans=sorted(spans))

    def emit(self, writer, consumers):
        writer.send("LookaheadBegin", distance=self.distance)
        for offset, size in self.spans:
            writer.send("LookaheadSpan", tensor=self.tensor, offset=offset, bytes=size)
        for spans in self.regions:
            writer.send("LookaheadRegion", spans=list(spans))
        for begin in range(0, len(consumers), 64):
            writer.send("LookaheadConsumers", consumers=consumers[begin:begin + 64])


def build_regions(schedule, weight, layout, base_bytes, queue_capacity):
    rows, cols = weight.shape
    ph, pw = schedule.pf_bk, schedule.pf_bn
    first_use = {}
    split_width = rows // schedule.split_k
    for rank, (split, n) in enumerate(schedule.tasks):
        r, c = split * split_width, n * schedule.bn
        # A task can touch multiple rectangles in either axis.
        for y in range(r // ph, (r + split_width - 1) // ph + 1):
            for x in range(c // pw, (c + schedule.bn - 1) // pw + 1):
                first_use.setdefault((y, x), rank)
    page_bytes = layout.page_bytes
    if base_bytes % page_bytes or rows * cols * 2 % page_bytes:
        raise Infeasible("prefetch requires a page-aligned weight allocation")
    region_pages = {}
    for y, x in first_use:
        r, c = y * ph, x * pw
        pages = layout.pages_for_rows(rows, cols, 2, r, c, min(ph, rows - r),
                                      min(pw, cols - c))
        region_pages[y, x] = sorted(pages)
    pages_per_group = layout.planes_per_command_group
    groups = layout.stacks * layout.dies_per_stack * (
        layout.planes_per_die // pages_per_group)

    def physical_order(page):
        group, lane = divmod(base_bytes // page_bytes + page, pages_per_group)
        return group // groups, lane, group % groups

    def region_order(coord):
        return first_use[coord], min(map(physical_order, region_pages[coord])), coord

    order = sorted(first_use, key=region_order)
    region_ids = {coord: i for i, coord in enumerate(order)}
    bounds = tuple((y * ph, x * pw, min(ph, rows - y * ph), min(pw, cols - x * pw))
                   for y, x in order)
    total_pages = rows * cols * 2 // page_bytes
    memberships = [[] for _ in range(total_pages)]
    for region_id, coord in enumerate(order):
        pages = region_pages[coord]
        _check_capacity(layout, base_bytes, pages, f"prefetch region {coord}")
        for page in pages:
            memberships[page].append(region_id)
    if any(not owners for owners in memberships):
        raise Infeasible("prefetch rectangles fail to cover the complete weight tensor")
    # Sharing is physical: overlapping rectangles use one reservation and one
    # fill per span, with actual loads determining its final release.
    spans, page_span = [], []
    region_spans = [[] for _ in order]
    for owners, page_run in groupby(enumerate(memberships), key=lambda item: item[1]):
        pages = [page for page, _ in page_run]
        span_id = len(spans)
        spans.append((pages[0] * page_bytes, len(pages) * page_bytes))
        page_span.extend([span_id] * len(pages))
        for owner in owners:
            region_spans[owner].append(span_id)
    if len(spans) > queue_capacity:
        raise Infeasible(f"{len(spans)} physical spans exceed the bounded descriptor queue")
    footprints = [len(region_pages[coord]) for coord in order]
    all_planes = Counter(layout.location(base_bytes // page_bytes + page)
                         for page in range(total_pages))
    spill = Counter()
    for (stack, _), count in all_planes.items():
        spill[stack] += max(0, count * page_bytes - layout.plane_bytes)
    summary = dict(region_count=len(order), physical_span_count=len(spans),
                   page_bytes=page_bytes, unique_pages=total_pages,
                   shared_pages=sum(len(owners) > 1 for owners in memberships),
                   summed_region_pages=sum(footprints), max_region_pages=max(footprints),
                   min_region_pages=min(footprints),
                   peak_plane_footprint_bytes=max(all_planes.values()) * page_bytes,
                   all_regions_fit_at_once=all(bytes_ <= layout.stack_bytes for bytes_ in spill.values()))
    return PrefetchRegions(schedule.ahead, weight.id, weight.shape, (ph, pw), page_bytes,
                           tuple(map(tuple, region_spans)), bounds, tuple(spans),
                           tuple(page_span), region_ids, summary)


def _check_capacity(layout, base_bytes, pages, region):
    private_pages = layout.plane_bytes // layout.page_bytes
    spill_budget = layout.stack_bytes // layout.page_bytes
    by_plane = Counter(layout.location(base_bytes // layout.page_bytes + p) for p in pages)
    spill = Counter()
    for (stack, _), count in by_plane.items():
        spill[stack] += max(0, count - private_pages)
    if any(count > spill_budget for count in spill.values()):
        raise Infeasible(f"{region} exceeds mapped per-plane and base-die SRAM capacity")
