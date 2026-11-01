"""Structured IR-03/IR-04 lowering for decode-capable Hopper kernels."""

from math import ceil
from ..ir.decode import (
    AssignmentWork,
    AttentionWork,
    CombineWork,
    DecodeAlgorithm,
    DecodeFlowKernel,
    DecodeNativeKernel,
    ExpertGemmWork,
    FlowLoop,
    FlowLoopAxis,
    FlowOperation,
    FlowOperator,
    FlowRole,
    GemmWork,
    KernelFamily,
    KVUpdateWork,
    NativeOpcode,
    NativeOperation,
    Pipeline,
    QKVTransformWork,
    RouterTopKWork,
    VectorWork,
    WeightedSumWork,
)
from ..serialization.json_codec import digest


def _loop(axis, extent, tile=1, predicate="tile_exists"):
    return FlowLoop(axis, max(1, extent), tile, predicate)


def _loops(spec):
    work = spec.work
    if isinstance(work, GemmWork):
        ctas = ceil(work.rows / work.tile_m) * ceil(work.columns / work.tile_n)
        return (
            _loop(FlowLoopAxis.CTA, ctas),
            _loop(FlowLoopAxis.GEMM_N, work.columns, work.tile_n),
            _loop(FlowLoopAxis.GEMM_K, work.reduction, work.tile_k),
        )
    if isinstance(work, VectorWork):
        ctas = ceil(work.rows / work.tile_rows) * ceil(work.columns / work.tile_columns)
        return (
            _loop(FlowLoopAxis.CTA, ctas),
            _loop(FlowLoopAxis.ROW, work.rows, work.tile_rows, "valid_rows"),
        )
    if isinstance(work, QKVTransformWork):
        return (
            _loop(FlowLoopAxis.CTA, work.batch * work.query_heads),
            _loop(FlowLoopAxis.ROW, work.head_dim, 32, "valid_rows"),
        )
    if isinstance(work, RouterTopKWork):
        return (
            _loop(FlowLoopAxis.CTA, work.batch),
            _loop(FlowLoopAxis.ROW, work.experts, 32, "valid_rows"),
        )
    if isinstance(work, KVUpdateWork):
        return (_loop(FlowLoopAxis.CTA, len(work.appends)),)
    if isinstance(work, AttentionWork):
        active_splits = sum(1 for _ in work.chunks)
        maximum_tiles = max(ceil((c.end - c.start) / work.kv_tile) for c in work.chunks)
        return (
            _loop(FlowLoopAxis.CTA, active_splits * work.query_heads),
            _loop(FlowLoopAxis.BATCH, len(work.requests)),
            _loop(FlowLoopAxis.HEAD, work.query_heads),
            _loop(FlowLoopAxis.SPLIT, work.split_count),
            _loop(FlowLoopAxis.KV_TILE, maximum_tiles, work.kv_tile, "valid_rows"),
        )
    if isinstance(work, AssignmentWork):
        ctas = max(1, ceil(len(work.assignments) / 16))
        return (
            _loop(FlowLoopAxis.CTA, ctas),
            _loop(FlowLoopAxis.EXPERT, max(1, len(work.groups))),
        )
    if isinstance(work, ExpertGemmWork):
        ctas = sum(
            ceil(len(group.assignments) / work.tile_m)
            * ceil(work.columns / work.tile_n)
            for group in work.groups
        )
        return (
            _loop(FlowLoopAxis.CTA, max(1, ctas)),
            _loop(FlowLoopAxis.EXPERT, len(work.groups)),
            _loop(FlowLoopAxis.EXPERT_M, max(1, work.rows), work.tile_m, "valid_rows"),
            _loop(FlowLoopAxis.GEMM_N, work.columns, work.tile_n),
            _loop(FlowLoopAxis.GEMM_K, work.reduction, work.tile_k),
        )
    if isinstance(work, CombineWork):
        return (
            _loop(FlowLoopAxis.CTA, work.requests * work.heads),
            _loop(FlowLoopAxis.SPLIT, work.splits),
        )
    if isinstance(work, WeightedSumWork):
        return (
            _loop(FlowLoopAxis.CTA, work.batch),
            _loop(FlowLoopAxis.ROW, work.hidden_size, 128),
        )
    raise TypeError(f"unsupported decode work: {type(work).__name__}")


def _body(spec):
    work = spec.work
    loops = tuple(loop.axis for loop in _loops(spec))
    ops = []

    def add(role, operator, tensor="", dependencies=(), valid_region=""):
        ops.append(
            FlowOperation(
                len(ops),
                role,
                operator,
                tensor,
                tuple(dependencies),
                loops,
                valid_region,
            )
        )
        return len(ops) - 1

    if isinstance(work, (GemmWork, ExpertGemmWork)):
        load_a = add(
            FlowRole.LOAD_A, FlowOperator.LOAD, work.input, valid_region="valid_m_rows"
        )
        load_b_tensor = (
            work.weight if isinstance(work, GemmWork) else "expert_weight_by_group"
        )
        weight_region = (
            "k16_n_tile"
            if isinstance(work, GemmWork)
            else f"active_expert_weight_{work.phase}_by_group_identity"
        )
        load_b = add(
            FlowRole.LOAD_B,
            FlowOperator.LOAD,
            load_b_tensor,
            valid_region=weight_region,
        )
        multiply = add(
            FlowRole.MATMUL,
            FlowOperator.MATMUL,
            work.output,
            (load_a, load_b),
            "m16_physical_valid_rows_masked",
        )
        add(
            FlowRole.STORE,
            FlowOperator.STORE,
            work.output,
            (multiply,),
            "logical_rows_only",
        )
    elif isinstance(work, VectorWork):
        load_ids = []
        for index, name in enumerate(work.inputs):
            role = (
                FlowRole.LOAD_A,
                FlowRole.LOAD_B,
                FlowRole.LOAD_Q,
                FlowRole.LOAD_K,
                FlowRole.LOAD_V,
            )[index % 5]
            load_ids.append(
                add(role, FlowOperator.LOAD, name, valid_region="logical_tile")
            )
        stage_ids = []
        for stage_index, stage in enumerate(work.stages):
            dependencies = tuple(stage_ids[index] for index in stage.dependencies)
            if not dependencies and stage_index:
                dependencies = (stage_ids[stage_index - 1],)
            dependencies += tuple(load_ids[index] for index in stage.input_indices)
            if stage.operation == FlowOperator.SILU:
                negative = add(
                    FlowRole.VECTOR,
                    FlowOperator.SUBTRACT,
                    work.outputs[0],
                    dependencies,
                    "silu_negate_input",
                )
                exponential = add(
                    FlowRole.VECTOR,
                    FlowOperator.EXPONENTIAL,
                    work.outputs[0],
                    (negative,),
                    "silu_exp_negative_input",
                )
                denominator = add(
                    FlowRole.VECTOR,
                    FlowOperator.ADD,
                    work.outputs[0],
                    (exponential,),
                    "silu_add_one",
                )
                stage_id = add(
                    FlowRole.VECTOR,
                    FlowOperator.NORMALIZE,
                    work.outputs[0],
                    (denominator, *dependencies),
                    "silu_x_over_one_plus_exp_minus_x",
                )
            else:
                stage_id = -1
                previous_stage = dependencies
                for _ in range(stage.repeats):
                    stage_id = add(
                        FlowRole.VECTOR,
                        stage.operation,
                        work.outputs[0],
                        previous_stage,
                        f"logical_elements_epsilon_{work.epsilon:g}",
                    )
                    previous_stage = (stage_id,)
            stage_ids.append(stage_id)
        previous = (stage_ids[-1],)
        for output in work.outputs:
            add(
                FlowRole.STORE,
                FlowOperator.STORE,
                output,
                previous,
                "logical_output_region",
            )
    elif isinstance(work, QKVTransformWork):
        qkv = add(
            FlowRole.LOAD_Q,
            FlowOperator.LOAD,
            work.qkv,
            valid_region="one_token_qkv_projection",
        )
        q_weight = add(
            FlowRole.LOAD_A,
            FlowOperator.LOAD,
            work.q_norm_weight,
            valid_region="32_query_head_norm_weights",
        )
        k_weight = add(
            FlowRole.LOAD_B,
            FlowOperator.LOAD,
            work.k_norm_weight,
            valid_region="4_kv_head_norm_weights",
        )
        q_norm = add(
            FlowRole.VECTOR,
            FlowOperator.NORMALIZE,
            work.query_output,
            (qkv, q_weight),
            "rmsnorm_epsilon_1e-6_fp32",
        )
        k_norm = add(
            FlowRole.VECTOR,
            FlowOperator.NORMALIZE,
            work.key_output,
            (qkv, k_weight),
            "rmsnorm_epsilon_1e-6_fp32",
        )
        q_rope = add(
            FlowRole.VECTOR,
            FlowOperator.ROTARY_EMBEDDING,
            work.query_output,
            (q_norm,),
            "qwen_rope_position_per_request",
        )
        k_rope = add(
            FlowRole.VECTOR,
            FlowOperator.ROTARY_EMBEDDING,
            work.key_output,
            (k_norm,),
            "qwen_rope_position_per_request",
        )
        value = add(
            FlowRole.VECTOR,
            FlowOperator.IDENTITY,
            work.value_output,
            (qkv,),
            "v_projection_unmodified",
        )
        add(
            FlowRole.STORE,
            FlowOperator.STORE,
            work.query_output,
            (q_rope,),
            "logical_query_heads_only",
        )
        add(
            FlowRole.STORE,
            FlowOperator.STORE,
            work.key_output,
            (k_rope,),
            "logical_kv_heads_only",
        )
        add(
            FlowRole.STORE,
            FlowOperator.STORE,
            work.value_output,
            (value,),
            "logical_kv_heads_only",
        )
    elif isinstance(work, RouterTopKWork):
        logits = add(
            FlowRole.LOAD_A,
            FlowOperator.LOAD,
            work.logits,
            valid_region="128_router_logits_per_request",
        )
        maximum = add(
            FlowRole.VECTOR,
            FlowOperator.MAX_REDUCTION,
            work.logits,
            (logits,),
            "softmax_over_128_experts",
        )
        exponentials = add(
            FlowRole.VECTOR,
            FlowOperator.EXPONENTIAL,
            work.logits,
            (maximum, logits),
            "stable_softmax_exp",
        )
        total = add(
            FlowRole.VECTOR,
            FlowOperator.SUM_REDUCTION,
            work.logits,
            (exponentials,),
            "softmax_normalizer",
        )
        topk = add(
            FlowRole.VECTOR,
            FlowOperator.TOPK,
            work.expert_ids,
            (logits, total),
            "top_8_unique_experts",
        )
        renormalized = add(
            FlowRole.VECTOR,
            FlowOperator.NORMALIZE,
            work.probabilities,
            (topk, exponentials, total),
            "renormalize_selected_probabilities",
        )
        add(
            FlowRole.STORE,
            FlowOperator.STORE,
            work.expert_ids,
            (topk,),
            "b_by_8_int32_route_ids",
        )
        add(
            FlowRole.STORE,
            FlowOperator.STORE,
            work.probabilities,
            (renormalized,),
            "b_by_8_bf16_route_weights",
        )
    elif isinstance(work, KVUpdateWork):
        for append in work.appends:
            load_key = add(
                FlowRole.LOAD_K,
                FlowOperator.LOAD,
                append.new_key,
                valid_region=f"request_{append.request}_head_{append.head}_one_new_key",
            )
            load_value = add(
                FlowRole.LOAD_V,
                FlowOperator.LOAD,
                append.new_value,
                valid_region=f"request_{append.request}_head_{append.head}_one_new_value",
            )
            add(
                FlowRole.STORE,
                FlowOperator.STORE,
                append.key_cache,
                (load_key,),
                f"append_at_context_{append.context_length}",
            )
            add(
                FlowRole.STORE,
                FlowOperator.STORE,
                append.value_cache,
                (load_value,),
                f"append_at_context_{append.context_length}",
            )
    elif isinstance(work, AttentionWork):
        load_q = add(
            FlowRole.LOAD_Q,
            FlowOperator.LOAD,
            work.query,
            valid_region="one_query_row_per_request_head",
        )
        load_k = add(
            FlowRole.LOAD_K,
            FlowOperator.LOAD,
            "request_kv",
            valid_region="request_context_plus_current_token",
        )
        load_v = add(
            FlowRole.LOAD_V,
            FlowOperator.LOAD,
            "request_vv",
            valid_region="request_context_plus_current_token",
        )
        initial_m = add(
            FlowRole.VECTOR,
            FlowOperator.FILL,
            "online_m",
            (),
            "negative_infinity_initial_max",
        )
        initial_l = add(
            FlowRole.VECTOR,
            FlowOperator.FILL,
            "online_l",
            (),
            "zero_initial_normalizer",
        )
        initial_o = add(
            FlowRole.VECTOR,
            FlowOperator.FILL,
            "online_o",
            (),
            "zero_initial_accumulator",
        )
        qk = add(
            FlowRole.QK,
            FlowOperator.MATMUL,
            "scores",
            (load_q, load_k),
            "GQA_head_to_kv_head",
        )
        scaled = add(
            FlowRole.VECTOR,
            FlowOperator.MULTIPLY,
            "scores",
            (qk,),
            "scale_by_inverse_sqrt_head_dim",
        )
        tile_max = add(
            FlowRole.SOFTMAX_MAX,
            FlowOperator.MAX_REDUCTION,
            "tile_max",
            (scaled,),
            "valid_kv_rows_only",
        )
        maximum = add(
            FlowRole.SOFTMAX_MAX,
            FlowOperator.MAX_REDUCTION,
            "online_m",
            (initial_m, tile_max),
            "stable_running_max",
        )
        old_delta = add(
            FlowRole.VECTOR,
            FlowOperator.SUBTRACT,
            "old_max_delta",
            (initial_m, maximum),
            "previous_max_minus_new_max",
        )
        alpha = add(
            FlowRole.SOFTMAX_EXP,
            FlowOperator.EXPONENTIAL,
            "online_alpha",
            (old_delta,),
            "online_softmax_rescale",
        )
        score_delta = add(
            FlowRole.VECTOR,
            FlowOperator.SUBTRACT,
            "score_delta",
            (scaled, maximum),
            "score_minus_new_max",
        )
        probabilities = add(
            FlowRole.SOFTMAX_EXP,
            FlowOperator.EXPONENTIAL,
            "online_p",
            (score_delta,),
            "valid_probability_rows",
        )
        tile_sum = add(
            FlowRole.SOFTMAX_SUM,
            FlowOperator.SUM_REDUCTION,
            "tile_sum",
            (probabilities,),
            "valid_probability_rows",
        )
        total = add(
            FlowRole.SOFTMAX_SUM,
            FlowOperator.FUSED_MULTIPLY_ADD,
            "online_l",
            (alpha, initial_l, tile_sum),
            "alpha_times_old_l_plus_tile_sum",
        )
        pv = add(
            FlowRole.PV,
            FlowOperator.MATMUL,
            "tile_pv",
            (load_v, probabilities),
            "bf16_v_online_pv_accum_fp32",
        )
        accumulator = add(
            FlowRole.VECTOR,
            FlowOperator.FUSED_MULTIPLY_ADD,
            "online_o",
            (alpha, initial_o, pv),
            "alpha_times_old_o_plus_tile_pv",
        )
        output_tensor = work.output if work.split_count == 1 else work.partial_output
        normalized = add(
            FlowRole.NORMALIZE,
            FlowOperator.NORMALIZE,
            output_tensor,
            (accumulator, total),
            "fp32_online_o_div_l",
        )
        add(
            FlowRole.STORE,
            FlowOperator.STORE,
            output_tensor,
            (normalized,),
            "one_output_per_valid_query_head",
        )
        if work.split_count > 1:
            lse = add(
                FlowRole.VECTOR,
                FlowOperator.LOGARITHM,
                work.partial_lse,
                (maximum, total),
                "partial_logsumexp_m_plus_log_l",
            )
            add(
                FlowRole.STORE,
                FlowOperator.STORE,
                work.partial_lse,
                (lse,),
                "one_partial_lse_per_query_head_and_split",
            )
    elif isinstance(work, AssignmentWork):
        loads = []
        for index, name in enumerate(
            (work.expert_ids, work.expert_probabilities, work.activation)
        ):
            role = (FlowRole.LOAD_A, FlowRole.LOAD_B, FlowRole.LOAD_Q)[index]
            loads.append(
                add(
                    role,
                    FlowOperator.LOAD,
                    name,
                    valid_region="only_b_times_topk_assignments"
                    if index < 2
                    else "one_activation_row_per_assignment",
                )
            )
        histogram = add(
            FlowRole.ASSIGN,
            FlowOperator.EXPERT_HISTOGRAM,
            work.sorted_assignments,
            tuple(loads),
            "128_expert_bins",
        )
        prefix = add(
            FlowRole.ASSIGN,
            FlowOperator.EXPERT_PREFIX_SUM,
            work.sorted_assignments,
            (histogram,),
            "variable_group_offsets",
        )
        gather = add(
            FlowRole.ASSIGN,
            FlowOperator.EXPERT_GATHER,
            work.sorted_activation,
            (prefix,),
            "valid_assignments_only",
        )
        add(
            FlowRole.STORE,
            FlowOperator.STORE,
            work.sorted_assignments,
            (gather,),
            "sorted_token_expert_slot_and_validity",
        )
        add(
            FlowRole.STORE,
            FlowOperator.STORE,
            work.sorted_probabilities,
            (gather,),
            "renormalized_selected_probabilities",
        )
        add(
            FlowRole.STORE,
            FlowOperator.STORE,
            work.sorted_activation,
            (gather,),
            "no_padded_global_rows",
        )
    elif isinstance(work, CombineWork):
        load_o = add(
            FlowRole.LOAD_A,
            FlowOperator.LOAD,
            work.partial_output,
            valid_region="all_split_partial_outputs",
        )
        load_lse = add(
            FlowRole.LOAD_B,
            FlowOperator.LOAD,
            work.partial_lse,
            valid_region="split_lse_values",
        )
        maximum = add(
            FlowRole.COMBINE,
            FlowOperator.MAX_REDUCTION,
            "split_lse_max",
            (load_lse,),
            "max_split_logsumexp",
        )
        exponentials = add(
            FlowRole.COMBINE,
            FlowOperator.EXPONENTIAL,
            "split_weights",
            (load_lse, maximum),
            "exp_split_lse_minus_global_max",
        )
        normalizer = add(
            FlowRole.COMBINE,
            FlowOperator.SUM_REDUCTION,
            "global_normalizer",
            (exponentials,),
            "sum_split_normalization_weights",
        )
        weighted = add(
            FlowRole.COMBINE,
            FlowOperator.WEIGHTED_SUM,
            "weighted_partials",
            (load_o, exponentials),
            "sum_exp_lse_weighted_partial_outputs",
        )
        combined = add(
            FlowRole.COMBINE,
            FlowOperator.NORMALIZE,
            work.output,
            (weighted, normalizer),
            "divide_by_global_lse_normalizer",
        )
        add(
            FlowRole.STORE,
            FlowOperator.STORE,
            work.output,
            (combined,),
            "one_query_output",
        )
    elif isinstance(work, WeightedSumWork):
        load_o = add(
            FlowRole.LOAD_A,
            FlowOperator.LOAD,
            work.expert_output,
            valid_region="only_selected_expert_outputs",
        )
        load_p = add(
            FlowRole.LOAD_B,
            FlowOperator.LOAD,
            work.expert_probabilities,
            valid_region="renormalized_topk_weights",
        )
        load_a = add(
            FlowRole.LOAD_Q,
            FlowOperator.LOAD,
            work.assignments,
            valid_region="token_slot_map",
        )
        result = add(
            FlowRole.VECTOR,
            FlowOperator.WEIGHTED_SUM,
            work.output,
            (load_o, load_p, load_a),
            "sum_e(probability_e_times_expert_output_e)",
        )
        add(
            FlowRole.STORE,
            FlowOperator.STORE,
            work.output,
            (result,),
            "logical_batch_hidden",
        )
    else:
        raise TypeError(f"unsupported decode work: {type(work).__name__}")
    return tuple(ops)


def make_flow(spec: DecodeAlgorithm) -> DecodeFlowKernel:
    spec.verify()
    result = DecodeFlowKernel(
        symbol=spec.symbol,
        source_hash=digest(spec),
        algorithm=spec,
        loops=_loops(spec),
        ops=_body(spec),
    )
    result.verify()
    return result


def _native_for(operator, role):
    if operator == FlowOperator.LOAD:
        return NativeOpcode.TMA_LOAD, Pipeline.LD
    if operator == FlowOperator.STORE:
        return NativeOpcode.TMA_STORE, Pipeline.ST
    if operator == FlowOperator.MATMUL:
        if role in (FlowRole.QK, FlowRole.PV):
            return NativeOpcode.SIMD_FMA, Pipeline.SIMD
        return NativeOpcode.MMA_SYNC, Pipeline.TENSOR
    if operator == FlowOperator.ADD:
        return NativeOpcode.SIMD_ADD, Pipeline.SIMD
    if operator == FlowOperator.SUBTRACT:
        return NativeOpcode.SIMD_SUB, Pipeline.SIMD
    if operator == FlowOperator.MULTIPLY:
        return NativeOpcode.SIMD_MUL, Pipeline.SIMD
    if operator == FlowOperator.FUSED_MULTIPLY_ADD:
        return NativeOpcode.SIMD_FMA, Pipeline.SIMD
    if operator == FlowOperator.MAX_REDUCTION:
        return (
            NativeOpcode.LOCAL_MAX
            if role == FlowRole.SOFTMAX_MAX
            else NativeOpcode.REDUCE_MAX
        ), Pipeline.SIMD
    if operator == FlowOperator.SUM_REDUCTION:
        return NativeOpcode.LOCAL_SUM, Pipeline.SIMD
    if operator == FlowOperator.EXPONENTIAL:
        return (
            NativeOpcode.EXP2 if role == FlowRole.SOFTMAX_EXP else NativeOpcode.SFU_EXP
        ), Pipeline.SFU
    if operator == FlowOperator.LOGARITHM:
        return NativeOpcode.SFU_LOG, Pipeline.SFU
    if operator == FlowOperator.RECIPROCAL_SQRT:
        return NativeOpcode.SFU_RSQRT, Pipeline.SFU
    if operator == FlowOperator.SIN:
        return NativeOpcode.SFU_SIN, Pipeline.SFU
    if operator == FlowOperator.COS:
        return NativeOpcode.SFU_COS, Pipeline.SFU
    if operator in (FlowOperator.NORMALIZE, FlowOperator.WEIGHTED_SUM):
        return (
            NativeOpcode.ACC_O_DIV
            if operator == FlowOperator.NORMALIZE
            else NativeOpcode.SIMD_FMA,
            Pipeline.SFU if operator == FlowOperator.NORMALIZE else Pipeline.SIMD,
        )
    if operator in (FlowOperator.ROTARY_EMBEDDING, FlowOperator.SILU):
        return (
            NativeOpcode.SIMD_FMA
            if operator == FlowOperator.ROTARY_EMBEDDING
            else NativeOpcode.SIMD_MUL,
            Pipeline.SIMD,
        )
    if operator == FlowOperator.TOPK:
        return NativeOpcode.SIMD_MAX, Pipeline.SIMD
    if operator == FlowOperator.EXPERT_HISTOGRAM:
        return NativeOpcode.SIMD_ADD, Pipeline.SIMD
    if operator == FlowOperator.EXPERT_PREFIX_SUM:
        return NativeOpcode.REDUCE_SUM, Pipeline.SIMD
    if operator == FlowOperator.EXPERT_GATHER:
        return NativeOpcode.SIMD_IDENTITY, Pipeline.SIMD
    if operator == FlowOperator.CAST:
        return NativeOpcode.SIMD_CAST, Pipeline.SIMD
    if operator == FlowOperator.IDENTITY:
        return NativeOpcode.SIMD_IDENTITY, Pipeline.SIMD
    if operator == FlowOperator.FILL:
        return NativeOpcode.SIMD_FILL, Pipeline.SIMD
    raise ValueError(f"no Hopper native pattern for {operator.value}")


def lower_native(flow: DecodeFlowKernel) -> DecodeNativeKernel:
    flow.verify()
    native_ops = []
    for op in flow.ops:
        opcode, pipeline = _native_for(op.operator, op.role)
        fragment = ()
        warps = 1
        if opcode == NativeOpcode.MMA_SYNC:
            fragment = (16, 8)
        elif opcode == NativeOpcode.WGMMA:
            fragment = (64, 64)
            warps = 4
        native_ops.append(
            NativeOperation(
                id=op.id,
                role=op.role,
                opcode=opcode,
                pipeline=pipeline,
                tensor=op.tensor,
                fragment_shape=fragment,
                collective_warps=warps,
                loop_axes=op.loops,
                dependencies=op.dependencies,
            )
        )
    work = flow.algorithm.work
    shared_bytes = 32768
    if isinstance(work, AttentionWork):
        shared_bytes += 2 * work.kv_tile * work.head_dim * 2
    result = DecodeNativeKernel(
        symbol=flow.symbol,
        source_hash=digest(flow),
        flow=flow,
        resource_warps=4,
        resource_shared_bytes=shared_bytes,
        resource_register_bytes=32768,
        ops=tuple(native_ops),
    )
    result.verify()
    return result
