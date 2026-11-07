"""IR-05 memory windows driven by native copy iterations and actual consumers."""

from dataclasses import replace
from parser import ast as A
from .prefetch import PageBudget, spans


def lower_panels(kernel, config, bases):
    tensors = {t.id: t for t in kernel.tensors}
    sizes = {
        t.name: t.shape[0]
        * t.shape[1]
        * (2 if t.dtype in (A.DataType.BF16, A.DataType.FP16) else 4)
        for t in kernel.tensors
    }
    by_name = {t.name: t.id for t in kernel.tensors}
    # One CTA per SM is a conservative, always-admissible scheduling wave.
    waves = {}
    seen = set()
    wave = 0
    for cta in kernel.ctas:
        if cta.sm in seen:
            wave += 1
            seen = set()
        seen.add(cta.sm)
        waves[cta.id] = wave
    loads = [
        n
        for n in kernel.ops
        if n.opcode == A.NativeOpcode.TMA_LOAD.value
        and n.tensor is not None
        and tensors[n.tensor].tier == A.MemorySpace.HBF
        and tensors[n.tensor].preloaded
        and tensors[n.tensor].read_only
    ]
    loads.sort(
        key=lambda n: (
            waves[n.cta],
            -1 if n.pipeline_iteration is None else n.pipeline_iteration,
            n.cta,
            n.id,
        )
    )
    windows = []
    consumers = []
    budget = PageBudget(config, bases, len(windows) % config.buffer_count)
    active_wave = None

    def finish():
        if not consumers:
            return
        epoch = len(windows)
        ranges = tuple(
            A.PrefetchRange(by_name[s.tensor], s.offset, s.bytes)
            for s in spans(budget.pages, sizes, config)
        )
        windows.append(
            A.PrefetchWindow(
                A.PrefetchBuffer(
                    epoch % config.buffer_count, config.buffer_count, epoch
                ),
                tuple(consumers),
                ranges,
            )
        )

    for node in loads:
        tensor = tensors[node.tensor]
        r, c, h, w = node.tile
        size = 2 if tensor.dtype in (A.DataType.BF16, A.DataType.FP16) else 4
        pages = set()
        if tensor.tile_major:
            pages.update(
                (tensor.name, p)
                for p in tensor.tile_major.pages(tensor.shape[1], r, c, h, w)
            )
        else:
            for row in range(r, r + h):
                begin = (row * tensor.shape[1] + c) * size
                end = begin + w * size
                pages.update(
                    (tensor.name, p)
                    for p in range(
                        begin // config.page_bytes, (end - 1) // config.page_bytes + 1
                    )
                )
        if waves[node.cta] != active_wave:
            finish()
            consumers = []
            budget = PageBudget(config, bases, len(windows) % config.buffer_count)
            active_wave = waves[node.cta]
        if not budget.add(pages):
            finish()
            consumers = []
            budget = PageBudget(config, bases, len(windows) % config.buffer_count)
            if not budget.add(pages):
                raise ValueError("one native copy exceeds a prefetch buffer")
        consumers.append(node.id)
    finish()
    return replace(kernel, prefetch_windows=tuple(windows))
