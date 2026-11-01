import re
from dataclasses import replace, dataclass
from ..ir.graph import KernelTensor
from ..ir.common import require
from ..ir.execution import ExecProgram, ExecutionLaunch
from ..serialization.json_codec import digest


@dataclass(frozen=True)
class KernelInterface:
    """Verified per-launch metadata retained after its eager DAG is released."""

    tensors: tuple[KernelTensor, ...]
    outputs: tuple[int, ...]
    content_hash: str

    def verify(self):
        ids = {t.id for t in self.tensors}
        require(
            len(ids) == len(self.tensors) and set(self.outputs) <= ids,
            "kernel interface",
            "invalid tensor/output metadata",
        )


def compose(program, kernels, kernel_files=None, launch_overhead_cycles=0):
    program.verify()
    tensor_ids = {}
    tensors = []
    launches = []
    produced = set()
    for invocation in program.launches:
        kernel = kernels[invocation.id]
        kernel.verify()
        bindings = dict(invocation.bindings)
        resolved = []
        for t in kernel.tensors:
            if invocation.algorithm.family in (
                "gemm",
                "vector",
            ) or invocation.algorithm.schedule_id in ("decode_simt", "decode_combine"):
                key = bindings[t.name]
            else:
                match = re.fullmatch(r"([QKVO])_b(\d+)_h(\d+)", t.name)
                if not match:
                    raise ValueError("invalid FA3 tensor identity")
                key = f"{bindings[match[1]]}.b{match[2]}.h{match[3]}"
            if key not in tensor_ids:
                tensor_ids[key] = len(tensors)
                tensors.append(replace(t, id=tensor_ids[key], name=key))
            else:
                old = tensors[tensor_ids[key]]
                if (old.shape, old.strides, old.dtype, old.tier) != (
                    t.shape,
                    t.strides,
                    t.dtype,
                    t.tier,
                ):
                    raise ValueError("incompatible persistent allocation alias")
                # A tensor produced by one kernel may be read by another.
                tensors[old.id] = replace(
                    old,
                    read_only=old.read_only and t.read_only,
                    preloaded=old.preloaded or (t.preloaded and key not in produced),
                )
            resolved.append((t.id, tensor_ids[key]))
        for output in kernel.outputs:
            global_id = dict(resolved)[output]
            produced.add(tensors[global_id].name)
        path = kernel_files[invocation.id] if kernel_files else f"{invocation.id}.gtsir"
        launches.append(ExecutionLaunch(invocation.id, path, tuple(resolved)))
    result = ExecProgram(
        symbol=program.symbol,
        source_hash=digest(program),
        streams=program.streams,
        tensors=tuple(tensors),
        launches=tuple(launches),
        commands=program.commands,
        launch_overhead_cycles=launch_overhead_cycles,
    )
    result.verify()
    return result
