"""Kernel-at-a-time frontend producer and the Unix framed transport.

The producer keeps the bound ArchIR and a compact launch plan.  Native
instructions are emitted directly from ``NativeExpander.expand_to`` when the
GTSim process asks for one launch; no program-wide ExecIR is built.
"""

from dataclasses import dataclass, replace
import json
import socket
import struct
from typing import Any

from parser import ast as A
from .expand_native import NativeExpander
from .static import EMPTY, Resolver, bound_resolver
from .opt.prefetch import PageBudget, spans
from .export_runtime import PIPELINES


PROTOCOL_VERSION = 1


class SocketFramedTransport:
    """Length-prefixed JSON transport matching runtime/stream_protocol.h."""

    def __init__(self, sock, max_frame=1 << 20):
        self.sock = sock
        self.max_frame = max_frame

    def send(self, message_type, **fields):
        payload = json.dumps(
            {"type": message_type, **fields}, separators=(",", ":")
        ).encode()
        if not payload or len(payload) > self.max_frame:
            raise ValueError("stream message exceeds frame limit")
        self.sock.sendall(struct.pack("<I", len(payload)) + payload)

    def receive(self):
        header = self._read_exact(4)
        (length,) = struct.unpack("<I", header)
        if not length or length > self.max_frame:
            raise ValueError("invalid stream frame length")
        message = json.loads(self._read_exact(length))
        if not isinstance(message, dict) or not message.get("type"):
            raise ValueError("stream message has no type")
        return message

    def _read_exact(self, count):
        result = bytearray()
        while len(result) < count:
            part = self.sock.recv(count - len(result))
            if not part:
                raise EOFError("GTSim stream closed")
            result.extend(part)
        return bytes(result)


@dataclass(frozen=True)
class LaunchPlan:
    launch_id: str
    stream_id: str
    arch_region_index: int
    tensor_bindings: tuple[tuple[int, int], ...]
    wait_tokens: tuple[str, ...]
    signal_token: str
    read_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProgramPlan:
    stream_id: str
    tensors: tuple[dict, ...]
    commands: tuple[dict, ...]
    launches: tuple[LaunchPlan, ...]
    prefetches: tuple[dict, ...] = ()
    symbol: str = "program"
    launch_overhead_cycles: int = 0


def make_program_plan(arch, manifest, regions):
    """Build a compact plan from the existing runtime manifest.

    ``manifest`` contains only metadata; ``regions`` are ArchIR regions and do
    not contain expanded instructions.  A completion token is automatically
    chained on the single ordered stream, while explicit command records are
    preserved verbatim.
    """
    if len(manifest.get("streams", ())) != 1:
        raise ValueError("streaming execution supports exactly one stream")
    stream = manifest["streams"][0]
    region_indices = {region.symbol: i for i, region in enumerate(regions)}
    command_launches = tuple(
        command.get("target")
        for command in manifest["commands"]
        if command.get("kind") == "LaunchKernel"
    )
    manifest_launches = tuple(launch["id"] for launch in manifest["launches"])
    if command_launches != manifest_launches:
        raise ValueError(
            "single-stream command order must match the launch manifest"
        )
    launches = []
    previous_token = None
    for launch in manifest["launches"]:
        launch_id = launch["id"]
        if launch_id not in region_indices:
            raise ValueError(f"launch has no active ArchIR region: {launch_id}")
        wait = (previous_token,) if previous_token else ()
        signal = launch_id + ".done"
        launches.append(
            LaunchPlan(
                launch_id,
                stream,
                region_indices[launch_id],
                tuple(tuple(pair) for pair in launch.get("bindings", ())),
                wait,
                signal,
                tuple(launch.get("read_aliases", ())),
            )
        )
        previous_token = signal
    return ProgramPlan(
        stream,
        tuple(manifest["tensors"]),
        tuple(manifest["commands"]),
        tuple(launches),
        tuple(manifest.get("prefetches", ())),
        manifest.get("symbol", arch.symbol),
        manifest.get("launch_overhead_cycles", 0),
    )


class ExecutionIRProducer:
    def __init__(self, arch_ir, plan: ProgramPlan, options=None):
        self.arch_ir = arch_ir
        self.plan = plan
        self.options = options or {}
        self._active_launch = None

    def emit_program_manifest(self, writer):
        writer.send(
            "ProgramBegin",
            protocol_version=PROTOCOL_VERSION,
            stream_id=self.plan.stream_id,
            symbol=self.plan.symbol,
            launch_overhead_cycles=self.plan.launch_overhead_cycles,
        )
        for tensor in self.plan.tensors:
            writer.send("TensorDef", **{k: v for k, v in tensor.items() if k != "record"})
        for launch in self.plan.launches:
            writer.send(
                "LaunchDef",
                id=launch.launch_id,
                stream_id=launch.stream_id,
                tensor_bindings=[list(pair) for pair in launch.tensor_bindings],
                wait_tokens=list(launch.wait_tokens),
                signal_token=launch.signal_token,
                read_aliases=list(launch.read_aliases),
            )
        for prefetch in self.plan.prefetches:
            writer.send(
                "PrefetchDef",
                **{k: v for k, v in prefetch.items() if k != "record"},
            )
        for command in self.plan.commands:
            writer.send(
                "CommandDef",
                kind=command["kind"],
                target=command.get("target", ""),
                after=command.get("after", []),
            )
        writer.send("ProgramReady", protocol_version=PROTOCOL_VERSION)

    def emit_kernel(self, launch_id: str, writer) -> None:
        if self._active_launch is not None:
            raise RuntimeError("concurrent kernel expansion is forbidden")
        launch = next((x for x in self.plan.launches if x.launch_id == launch_id), None)
        if launch is None:
            raise KeyError(launch_id)
        self._active_launch = launch_id
        try:
            region = self.arch_ir.regions[launch.arch_region_index]
            root = bound_resolver(self.arch_ir.static, Resolver(EMPTY, {})).materialize_constants()
            arguments = {
                arg.id: root.value(arg.expression) for arg in region.static.arguments
            }
            values = {**root.values, **arguments}
            resolver = Resolver(
                replace(region.static, parameters=(), constraints=()), values
            ).materialize_constants()
            tensors = []
            for value in self.arch_ir.tensors:
                shape = tuple(root.value(d) for d in value.shape)
                tensors.append(
                    A.KernelTensor(
                        len(tensors),
                        value.id,
                        shape,
                        _strides(shape),
                        value.dtype,
                        value.tier or A.MemorySpace.HBM,
                        value.external,
                        value.read_only,
                        self.options.get("layouts", {}).get(value.id),
                    )
                )
            by_name = {tensor.name: tensor for tensor in tensors}
            writer.send(
                "KernelBegin",
                launch_id=launch.launch_id,
                stream_id=launch.stream_id,
                wait_tokens=list(launch.wait_tokens),
                signal_token=launch.signal_token,
                tensor_bindings=[list(pair) for pair in launch.tensor_bindings],
                read_aliases=list(launch.read_aliases),
            )
            ctas = []
            loads = []
            store_areas = {}
            # A kernel's output contract is the set of tensors it actually
            # stores.  Program-level returns may omit intermediates (e.g. H in
            # a two-GEMM chain), so using ArchIR.outputs here would reject a
            # valid intermediate kernel.
            output_ids = set()

            def emit_cta(cta):
                ctas.append(cta)
                writer.send(
                    "CTA",
                    id=cta.id,
                    sm=cta.sm,
                    warps=cta.warps,
                    shared_bytes=cta.shared_bytes,
                    register_bytes=cta.register_bytes,
                )

            def emit_instruction(node):
                writer.send(
                    "Instruction",
                    id=node.id,
                    name=node.name,
                    op=node.opcode,
                    pipeline=PIPELINES[node.resource],
                    cta=node.cta,
                    warp=node.warp,
                    tensor=node.tensor if node.tensor is not None else 0,
                    dtype=node.dtype.value,
                    leading_dim=node.leading_dim or 0,
                    tile=list(node.tile),
                    after_issue=list(node.after_issue),
                    after_complete=list(node.after_complete),
                )
                if node.opcode == A.NativeOpcode.TMA_LOAD.value and node.tensor is not None:
                    loads.append((node.cta, node.pipeline_iteration, node.id, node.tensor, node.tile))
                if node.opcode == A.NativeOpcode.TMA_STORE.value and node.tensor is not None:
                    store_areas[node.tensor] = store_areas.get(node.tensor, 0) + node.tile[2] * node.tile[3]
                    output_ids.add(node.tensor)

            expander = NativeExpander(region, resolver, by_name)
            expander.expand_to(emit_cta, emit_instruction)

            for window in self._prefetch_windows(loads, ctas, tensors):
                writer.send(
                    "PrefetchWindow",
                    buffer={
                        "slot": window.buffer.slot,
                        "count": window.buffer.count,
                        "epoch": window.buffer.epoch,
                    },
                    consumers=list(window.consumers),
                    ranges=[
                        {"tensor": r.tensor, "offset": r.offset, "bytes": r.bytes}
                        for r in window.ranges
                    ],
                )
            partial = [
                tensor_id
                for tensor_id in output_ids
                if store_areas.get(tensor_id, 0)
                < tensors[tensor_id].shape[0] * tensors[tensor_id].shape[1]
            ]
            writer.send(
                "KernelEnd",
                op_count=expander.next_node_id,
                cta_count=len(ctas),
                outputs=sorted(output_ids),
                partial_outputs=partial,
            )
        finally:
            self._active_launch = None

    def _prefetch_windows(self, loads, ctas, tensors):
        config = self.options.get("prefetch_config")
        bases = self.options.get("prefetch_bases")
        if config is None or bases is None:
            return ()
        tensors_by_id = {tensor.id: tensor for tensor in tensors}
        tensor_ids_by_name = {tensor.name: tensor.id for tensor in tensors}
        sizes = {
            tensor.name: tensor.shape[0]
            * tensor.shape[1]
            * (2 if tensor.dtype in (A.DataType.BF16, A.DataType.FP16) else 4)
            for tensor in tensors
        }
        waves, seen, wave = {}, set(), 0
        for cta in ctas:
            if cta.sm in seen:
                wave += 1
                seen = set()
            seen.add(cta.sm)
            waves[cta.id] = wave
        eligible = [
            item
            for item in loads
            if tensors_by_id[item[3]].tier == A.MemorySpace.HBF
            and tensors_by_id[item[3]].preloaded
            and tensors_by_id[item[3]].read_only
        ]
        eligible.sort(key=lambda item: (waves[item[0]], -1 if item[1] is None else item[1], item[0], item[2]))
        windows, consumers = [], []
        budget = PageBudget(config, bases, len(windows) % config.buffer_count)
        active_wave = None

        def finish():
            if not consumers:
                return
            windows.append(
                A.PrefetchWindow(
                    A.PrefetchBuffer(len(windows) % config.buffer_count, config.buffer_count, len(windows)),
                    tuple(consumers),
                    tuple(
                        A.PrefetchRange(tensor_ids_by_name[span.tensor], span.offset, span.bytes)
                        for span in spans(budget.pages, sizes, config)
                    ),
                )
            )

        for _, _, node_id, tensor_id, tile in eligible:
            tensor = tensors_by_id[tensor_id]
            r, c, h, w = tile
            width = 2 if tensor.dtype in (A.DataType.BF16, A.DataType.FP16) else 4
            pages = set()
            if tensor.tile_major:
                pages.update((tensor.name, p) for p in tensor.tile_major.pages(tensor.shape[1], r, c, h, w))
            else:
                for row in range(r, r + h):
                    begin = (row * tensor.shape[1] + c) * width
                    end = begin + w * width
                    pages.update(
                        (tensor.name, p)
                        for p in range(begin // config.page_bytes, (end - 1) // config.page_bytes + 1)
                    )
            current_wave = waves[_]
            if current_wave != active_wave:
                finish()
                consumers = []
                budget = PageBudget(config, bases, len(windows) % config.buffer_count)
                active_wave = current_wave
            if not budget.add(pages):
                finish()
                consumers = []
                budget = PageBudget(config, bases, len(windows) % config.buffer_count)
                if not budget.add(pages):
                    raise ValueError("one native copy exceeds a prefetch buffer")
            consumers.append(node_id)
        finish()
        return tuple(windows)


def _strides(shape):
    stride, result = 1, []
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))
