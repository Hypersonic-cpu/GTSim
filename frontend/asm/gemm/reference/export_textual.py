"""Export the legacy Reference GEMM generator in design-v0.2 text form."""

from pathlib import Path
import sys

FRONTEND_SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(FRONTEND_SRC))

from gtsim_frontend.passes.instantiate import instantiate
from gtsim_frontend.programs.gemm_single import build


def _enum(value):
    return value.value if hasattr(value, "value") else str(value)


def _shape(shape, dtype):
    return f"{'x'.join(map(str, shape))}x{dtype}"


def _index(expression):
    terms = []
    if expression.constant:
        terms.append(str(expression.constant))
    for variable, coefficient in expression.terms:
        name = variable.value if hasattr(variable, "value") else variable
        terms.append(f"%{name}" if coefficient == 1 else f"%{name}*{coefficient}")
    return " + ".join(terms) if terms else "0"


def _value(value):
    name = _enum(value.producer)
    suffix = ".prev" if value.iteration_delta < 0 else ""
    if value.result:
        suffix += f".{value.result}"
    if _enum(value.initial) != "none":
        suffix += f"<{_enum(value.initial)}>"
    return f"%{name}{suffix}"


def _dependencies(operation):
    return ", ".join(
        f"@{_enum(dep.source)}.{_enum(dep.phase)}:{_enum(dep.relation)}"
        for dep in operation.dependencies
    )


def _region(region):
    offsets = ", ".join(_index(offset) for offset in region.offsets)
    shape = "x".join(map(str, region.shape))
    return f"@{region.tensor}[{offsets}]<{shape}>"


def _operation(operation, profile):
    role = _enum(operation.role)
    name = operation.id
    executor = _enum(operation.executor)
    scope = _enum(operation.scope)
    deps = _dependencies(operation)
    after = f" after[{deps}]" if deps else ""
    slot = ""
    if operation.slot:
        buffer = _enum(operation.slot.buffer).lower()
        depth = operation.slot.stage.modulus
        slot = f" slots[%{buffer}_stage.s(%k mod {depth})]"

    if role == "acquire":
        return f"flow.acquire @{name} scope<{scope}>{slot}{after}"
    if role in ("load_A", "load_B"):
        region = _region(operation.region)
        result = operation.results[0]
        shape = _shape(result.shape, result.dtype)
        view = "a" if role == "load_A" else "b"
        return (
            f"%{name}, !{name}_ready = flow.load_async @{name} "
            f"{region} to %{view}_stage.s(%k mod {profile.stages}) "
            f"by %{executor}{after} requires<bulk_copy> : tensor<{shape}>"
        )
    if role == "matmul":
        values = ", ".join(_value(value) for value in operation.operands)
        result = operation.results[0]
        shape = _shape(result.shape, result.dtype)
        return (
            f"%{name}, !{name}_done = flow.matmul @{name} reads[{values}] "
            f"on_tile<{_region(operation.region)}> by %{executor}{after} "
            f"requires<matrix.mma> : tensor<{shape}>"
        )
    if role == "release":
        return f"flow.release @{name}{slot}{after}"
    if role == "store":
        values = ", ".join(_value(value) for value in operation.operands)
        return (
            f"!{name}_done = flow.store_async @{name} reads[{values}] "
            f"cast<fp16> to {_region(operation.region)} by %{executor}{after} "
            f"requires<bulk_copy>"
        )
    raise ValueError(f"unsupported Reference GEMM Flow operation: {role}")


def _algorithm_text(algorithm):
    dtype = algorithm.dtype
    return f"""algo.kernel @gemm_reference(%A: tensor<MxKx{dtype}>, %B: tensor<KxNx{dtype}>)
    -> (%C: tensor<MxNx{dtype}>) {{
  algo.domain %m in [0, M), %n in [0, N), %k in [0, K)
  algo.accumulator @acc : tensor<MxNxfp32> init<zero>
  algo.compute @matmul result<%acc[%m,%n]>
      semantics<sum_k(cast_f32(%A[%m,%k]) * cast_f32(%B[%k,%n]))>
  algo.compute @cast_output result<%C[%m,%n]>
      semantics<cast_fp16(%acc[%m,%n])>
  algo.stage @operand_acquisition reads[%A, %B]
  algo.stage @matrix_accumulate after[@operand_acquisition] produces[%acc]
  algo.stage @epilogue after[@matrix_accumulate] writes[%C]
  algo.family <tiled_matrix, producer_consumer_pipeline>
  algo.require <same_result, reduction_complete, output_complete>
}}
"""


def _flow_text(flow):
    profile = flow.profile
    bm, bn, bk, stages = profile.bm, profile.bn, profile.bk, profile.stages
    operations = [_operation(op, profile) for op in flow.ops]
    pipeline_ops = [
        line for line, op in zip(operations, flow.ops) if _enum(op.scope) == "pipeline"
    ]
    epilogue_ops = [
        line for line, op in zip(operations, flow.ops) if _enum(op.scope) == "epilogue"
    ]
    pipeline_body = "\n".join(f"    {line}" for line in pipeline_ops)
    epilogue_body = "\n".join(f"  {line}" for line in epilogue_ops)
    a_stage_bytes = bm * bk * profile.io_bytes
    b_stage_bytes = bk * bn * profile.io_bytes
    bytes_per_slot = a_stage_bytes + b_stage_bytes
    return f"""flow.kernel @gemm_reference_flow schedule<{flow.schedule_id}> target<sm90> {{
  flow.symbol %M range[{bm}, *]
  flow.symbol %N range[{bn}, *]
  flow.symbol %K range[{bk}, *]
  flow.symbol %NUM_SMS range[1, *]
  flow.require <%M divisible_by {bm}, %N divisible_by {bn}, %K divisible_by {bk}>
  flow.tensor @A : tensor<%Mx%Kx{flow.tensors[0].dtype}, row_major>
  flow.tensor @B : tensor<%Kx%Nx{flow.tensors[1].dtype}, row_major>
  flow.tensor @C : tensor<%Mx%Nx{flow.tensors[2].dtype}, row_major>
  flow.domain %tile_count = ceildiv(%M,{bm})*ceildiv(%N,{bn})
      %cta_count = min(%NUM_SMS, %tile_count)
      %wave_count = ceildiv(%tile_count, %cta_count)
  flow.schedule_config bm({bm}) bn({bn}) bk({bk}) stages({stages})
      warps({profile.warps}) consumer_warps({profile.consumer_warps})
      producer_warp({profile.producer_warp}) warpgroup({profile.warpgroup})
      mma_tile({profile.mma_m}x{profile.mma_n}x{profile.mma_k})
  flow.loop @cta %cta = 0 to %cta_count step 1
  flow.loop @wave %wave = 0 to %wave_count step 1
  flow.tile_map %tile = %cta + %wave*%cta_count
      %row = floordiv(%tile, ceildiv(%N,{bn}))
      %column = %tile mod ceildiv(%N,{bn})
  flow.executor %producer role<producer> organization<independent>
  flow.executor %matrix role<matrix> organization<cooperative>
  flow.buffer %ab_stage slots({stages}) access<read_write> : bytes_per_slot<{bytes_per_slot}>
  flow.buffer_view %a_stage of %ab_stage offset<0> : buffer<{bm}x{bk}x{flow.tensors[0].dtype}>
  flow.buffer_view %b_stage of %ab_stage offset<{a_stage_bytes}> : buffer<{bk}x{bn}x{flow.tensors[1].dtype}>
  flow.pipeline @k_loop %k = 0 to ceildiv(%K,{bk}) step 1 depth({stages})
      generation_counts_per_wave[ceildiv(ceildiv(%K,{bk}),{stages}), floordiv(ceildiv(%K,{bk}),{stages})] {{
{pipeline_body}
  }}
{epilogue_body}
}}

mapping.program @gemm_reference_mapping flow<@gemm_reference_flow>
    target<{flow.mapping.target}> placement<{flow.mapping.placement}> {{
  map.execute @producer capability<bulk_copy> -> @TMA
  map.execute @matrix capability<matrix.mma> -> @WGMMA
  map.store @ab_stage -> @SMEM
  map.store @output -> @SMEM
  map.route @input_route capability<bulk_copy> @HBM -> @SMEM via<@tma_route>
}}
"""


def _program_text(program):
    invocation = program.launches[0]
    inputs = [source for source, target in invocation.bindings if target != "C"]
    output = next(source for source, target in invocation.bindings if target == "C")
    tensors = {tensor.id: tensor for tensor in program.tensors}
    tensor_lines = []
    for name, shape in (
        ("A", "%Mx%K"),
        ("B", "%Kx%N"),
        ("C", "%Mx%N"),
    ):
        tensor = tensors[name]
        external = " external readonly" if tensor.read_only else ""
        tensor_lines.append(
            f"  prog.tensor @{name} : tensor<{shape}x{tensor.dtype}> "
            f"storage<{tensor.tier}>{external}"
        )
    return f"""prog.program @gemm_reference_example params(%M: index, %N: index, %K: index) {{
{chr(10).join(tensor_lines)}
  prog.stream @compute ordered {{
    %{output}, !e0 = prog.launch @{invocation.id}
        callee<@gemm_reference> implementation<@gemm_reference_flow>
        inputs[{", ".join("@" + name for name in inputs)}] outputs[@{output}]
        attrs{{M=%M, N=%N, K=%K}}
    prog.synchronize @compute after[!e0]
  }}
  prog.return @{output} after[!e0]
}}
"""


def main():
    output = Path(__file__).resolve().parent
    program = build(128, 256, 64, "fp16")
    algorithm = program.launches[0].algorithm
    flow = instantiate(algorithm, 132)
    (output / "gemm.algoir").write_text(_algorithm_text(algorithm))
    (output / "gemm.flowir").write_text(_flow_text(flow))
    (output / "example.progir").write_text(_program_text(program))


if __name__ == "__main__":
    main()
