import re
from dataclasses import replace
from ..ir.execution import ExecProgram, ExecutionLaunch
from ..serialization.json_codec import digest


def compose(program, kernels, kernel_files=None, launch_overhead_cycles=0):
    program.verify()
    tensor_ids = {}
    tensors = []
    launches = []
    for invocation in program.launches:
        kernel = kernels[invocation.id]
        kernel.verify()
        bindings = dict(invocation.bindings)
        resolved = []
        for t in kernel.tensors:
            if invocation.algorithm.family == "gemm":
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
                    preloaded=old.preloaded or t.preloaded,
                )
            resolved.append((t.id, tensor_ids[key]))
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
