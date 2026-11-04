"""Adapter from canonical ExecIR to the existing C++ ProgramExecutor wire format.

Legacy schema discriminators exist only at this boundary, never in compiler ASTs.
"""

from pathlib import Path
import json
from parser import ast as A

PIPELINES = {
    A.ExecutionResource.LOAD: "LD",
    A.ExecutionResource.STORE: "ST",
    A.ExecutionResource.MATRIX: "Tensor",
    A.ExecutionResource.ARITHMETIC: "SIMD",
    A.ExecutionResource.SPECIAL_FUNCTION: "SFU",
    A.ExecutionResource.SHUFFLE: "SHFL",
}


def tensor_record(tensor):
    return dict(
        record="KernelTensor",
        id=tensor.id,
        name=tensor.name,
        shape=tensor.shape,
        strides=tensor.strides,
        dtype=tensor.dtype.value,
        tier=tensor.tier.value,
        preloaded=tensor.preloaded,
        read_only=tensor.read_only,
    )


def export_runtime(execution, directory, launch_overhead_cycles=0, prefetch=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = [tensor_record(t) for t in execution.tensors]
    bases, next_warp = {}, 0
    for cta in execution.ctas:
        bases[cta.id] = next_warp
        next_warp += cta.warps
    launches, commands = [], []
    for region in execution.regions:
        ids = {node: i for i, node in enumerate(region.nodes)}
        cta_ids = {old: i for i, old in enumerate(region.ctas)}
        ctas = [execution.ctas[i] for i in region.ctas]
        local_bases, offset = {}, 0
        for cta in ctas:
            local_bases[cta.id] = offset
            offset += cta.warps
        ops = []
        for identity in region.nodes:
            node = execution.ops[identity]
            ops.append(
                dict(
                    record="ArchInstruction",
                    id=ids[node.id],
                    name=node.name,
                    op=node.opcode,
                    pipeline=PIPELINES[node.resource],
                    cta=cta_ids[node.cta],
                    warp=node.warp - bases[node.cta] + local_bases[node.cta],
                    tensor=node.tensor if node.tensor is not None else 0,
                    dtype=node.dtype.value,
                    leading_dim=node.leading_dim or 0,
                    tile=node.tile,
                    after_issue=[ids[d] for d in node.after_issue if d in ids],
                    after_complete=[ids[d] for d in node.after_complete if d in ids],
                )
            )
        kernel = dict(
            record="ExecKernel",
            format="gtsim.exec.kernel/1",
            schema_version=1,
            symbol=region.id,
            source_hash="",
            target=execution.target,
            sync_semantics_version="",
            tensors=tensors,
            ctas=[
                dict(
                    record="CTA",
                    id=cta_ids[c.id],
                    sm=c.sm,
                    warps=c.warps,
                    shared_bytes=c.shared_bytes,
                    register_bytes=c.register_bytes,
                )
                for c in ctas
            ],
            ops=ops,
            partial_outputs=[
                tensor.id
                for tensor in execution.tensors
                if tensor.id in region.outputs
                and sum(
                    node.tile[2] * node.tile[3]
                    for node in (execution.ops[i] for i in region.nodes)
                    if node.opcode == A.NativeOpcode.TMA_STORE.value
                    and node.tensor == tensor.id
                )
                < tensor.shape[0] * tensor.shape[1]
            ],
            outputs=region.outputs,
        )
        filename = region.id + ".gtsir"
        with (directory / filename).open("w") as output:
            json.dump(kernel, output, separators=(",", ":"))

        launches.append(
            dict(
                record="ExecutionLaunch",
                id=region.id,
                kernel_file=filename,
                bindings=[[t.id, t.id] for t in execution.tensors],
            )
        )
        commands.append(dict(record="Command", kind="LaunchKernel", target=region.id))
    commands.append(dict(record="Command", kind="StreamSynchronize", target=""))
    program = dict(
        record="ExecProgram",
        format="gtsim.exec.program/1",
        schema_version=1,
        symbol=execution.symbol,
        source_hash="",
        launch_overhead_cycles=launch_overhead_cycles,
        streams=[execution.stream or "compute"],
        tensors=tensors,
        launches=launches,
        commands=commands,
    )
    if prefetch is not None:
        prefetch.decorate(program)
    path = directory / "program.gtsir"
    path.write_text(json.dumps(program, indent=2) + "\n")
    return path
