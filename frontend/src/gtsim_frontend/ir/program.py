from dataclasses import dataclass
from .common import Header, Tensor, require
from .algorithm import Algorithm


@dataclass(frozen=True)
class Invocation:
    id: str
    algorithm: Algorithm
    bindings: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Command:
    kind: str
    target: str = ""


@dataclass(frozen=True)
class Program(Header):
    streams: tuple[str, ...] = ("compute",)
    tensors: tuple[Tensor, ...] = ()
    launches: tuple[Invocation, ...] = ()
    commands: tuple[Command, ...] = ()

    def verify(self):
        super().verify()
        require(
            len(self.streams) == 1 and bool(self.streams[0]),
            self.symbol,
            "exactly one stream required",
        )
        tensors = {t.id: t for t in self.tensors}
        launches = {k.id: k for k in self.launches}
        require(
            len(tensors) == len(self.tensors), self.symbol, "duplicate tensor identity"
        )
        require(
            len(launches) == len(self.launches),
            self.symbol,
            "duplicate invocation identity",
        )
        for t in self.tensors:
            t.verify()
        for k in self.launches:
            k.algorithm.verify()
            bindings = dict(k.bindings)
            if k.algorithm.schedule_id in (
                "decode_simt",
                "decode_combine",
                "grouped_small_m_mma",
                "simt_vector",
            ):
                from ..kernels.hopper.attention_decode import physical_tensors

                if k.algorithm.schedule_id == "decode_combine":
                    from ..kernels.hopper.attention_combine import physical_tensors
                if k.algorithm.schedule_id == "grouped_small_m_mma":
                    from ..kernels.hopper.gemm_groups import physical_tensors
                if k.algorithm.family == "vector":
                    from ..kernels.vector import signature

                    specs = signature(k.algorithm.primitive)
                else:
                    specs = physical_tensors(k.algorithm)
                require(
                    len(bindings) == len(k.bindings)
                    and set(bindings) == {t.id for t in specs},
                    k.id,
                    "invalid scheduled kernel bindings",
                )
                for spec in specs:
                    require(
                        bindings[spec.id] in tensors, k.id, "unbound scheduled tensor"
                    )
                    actual = tensors[bindings[spec.id]]
                    require(
                        (actual.shape, actual.dtype, actual.tier)
                        == (spec.shape, spec.dtype, spec.tier),
                        k.id,
                        "scheduled tensor shape/dtype mismatch",
                    )
                    require(
                        spec.read_only or not actual.read_only,
                        k.id,
                        "scheduled output is read-only",
                    )
                inputs = {bindings[t.id] for t in specs if t.read_only}
                outputs = [bindings[t.id] for t in specs if not t.read_only]
                require(
                    len(outputs) == len(set(outputs)) and not (set(outputs) & inputs),
                    k.id,
                    "scheduled output alias",
                )
                continue
            expected = (
                ("A", "B", "C")
                if k.algorithm.family == "gemm"
                else ("Q", "K", "V", "O")
            )
            require(
                len(bindings) == len(k.bindings) and set(bindings) == set(expected),
                k.id,
                "invalid local bindings",
            )
            require(
                all(t in tensors for t in bindings.values()),
                k.id,
                "unbound program tensor",
            )
            a = k.algorithm
            if a.family == "gemm":
                m, n, r = a.dimensions
                shapes = ((m, r), (r, n), (m, n))
            else:
                b, hq, hkv, q, kv = a.dimensions
                shapes = (
                    (b, hq, q, 128),
                    (b, hkv, 128, kv),
                    (b, hkv, 128, kv),
                    (b, hq, q, 128),
                )
            for local, shape in zip(expected, shapes):
                t = tensors[bindings[local]]
                require(
                    t.shape == shape and t.dtype == a.dtype,
                    k.id,
                    f"{local}: incompatible shape or dtype",
                )
            output = bindings[expected[-1]]
            require(
                not tensors[output].read_only
                and output not in [bindings[n] for n in expected[:-1]],
                k.id,
                "invalid output alias or read-only allocation",
            )
        events, submitted = set(), set()
        for i, c in enumerate(self.commands):
            where = f"{self.symbol}.commands[{i}]"
            if c.kind == "LaunchKernel":
                require(
                    c.target in launches and c.target not in submitted,
                    where,
                    "unknown or duplicate launch",
                )
                submitted.add(c.target)
            elif c.kind == "RecordEvent":
                require(
                    bool(c.target) and c.target not in events,
                    where,
                    "empty or duplicate event",
                )
                events.add(c.target)
            elif c.kind == "WaitEvent":
                require(
                    c.target in events, where, "wait-before-record or dangling event"
                )
            else:
                require(
                    c.kind == "StreamSynchronize" and not c.target,
                    where,
                    "unsupported command",
                )
        require(submitted == set(launches), self.symbol, "unsubmitted invocation")
