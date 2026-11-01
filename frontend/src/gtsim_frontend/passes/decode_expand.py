"""Expand structured decode-native IR to executable IR-05 instructions."""

from math import ceil

from ..ir.execution import ExecKernel
from ..ir.graph import ArchInstruction, CTA, KernelTensor
from ..ir.decode import (
    AssignmentWork,
    AttentionWork,
    CombineWork,
    ExpertGemmWork,
    FlowOperator,
    GemmWork,
    KernelFamily,
    KVUpdateWork,
    NativeOpcode,
    QKVTransformWork,
    RouterTopKWork,
    VectorWork,
    WeightedSumWork,
)
from ..serialization.json_codec import digest


_PIPELINE = {
    NativeOpcode.TMA_LOAD: "LD",
    NativeOpcode.TMA_STORE: "ST",
    NativeOpcode.WGMMA: "Tensor",
    NativeOpcode.MMA_SYNC: "Tensor",
    NativeOpcode.SIMD_ADD: "SIMD",
    NativeOpcode.SIMD_SUB: "SIMD",
    NativeOpcode.SIMD_MUL: "SIMD",
    NativeOpcode.SIMD_FMA: "SIMD",
    NativeOpcode.SIMD_MAX: "SIMD",
    NativeOpcode.SIMD_MIN: "SIMD",
    NativeOpcode.SIMD_CAST: "SIMD",
    NativeOpcode.SIMD_FILL: "SIMD",
    NativeOpcode.SIMD_IDENTITY: "SIMD",
    NativeOpcode.REDUCE_MAX: "SIMD",
    NativeOpcode.MERGE_MAX: "SIMD",
    NativeOpcode.REDUCE_SUM: "SIMD",
    NativeOpcode.MERGE_SUM: "SIMD",
    NativeOpcode.LOCAL_MAX: "SIMD",
    NativeOpcode.LOCAL_SUM: "SIMD",
    NativeOpcode.EXP2: "SFU",
    NativeOpcode.SCORES_SCALE: "SFU",
    NativeOpcode.ACC_O_SCALE: "SIMD",
    NativeOpcode.ACC_S_CAST: "SIMD",
    NativeOpcode.ACC_O_DIV: "SFU",
    NativeOpcode.LOGSUM: "SIMD",
    NativeOpcode.SHFL: "SHFL",
    NativeOpcode.SFU_EXP: "SFU",
    NativeOpcode.SFU_LOG: "SFU",
    NativeOpcode.SFU_SQRT: "SFU",
    NativeOpcode.SFU_RSQRT: "SFU",
    NativeOpcode.SFU_DIVIDE: "SFU",
    NativeOpcode.SFU_SIN: "SFU",
    NativeOpcode.SFU_COS: "SFU",
}


class KernelExpander:
    def __init__(self, native, num_sms):
        native.verify()
        self.native = native
        self.spec = native.flow.algorithm
        self.num_sms = num_sms
        self.uses = {use.name: use for use in self.spec.tensors}
        self.tensor_ids = {name: index for index, name in enumerate(self.uses)}
        self.ctas = []
        self.ops = []
        self.output_ids = set()
        self.partial_ids = set()
        self._last_cta_op = {}

    def cta(self, cta_id, shared_bytes=None, register_bytes=None, warps=4):
        while len(self.ctas) <= cta_id:
            current = len(self.ctas)
            shared = shared_bytes or self.native.resource_shared_bytes
            registers = register_bytes or self.native.resource_register_bytes
            self.ctas.append(
                CTA(
                    current,
                    current % self.num_sms,
                    warps,
                    shared,
                    registers,
                )
            )

    def emit(
        self,
        cta,
        opcode,
        tensor="",
        region=(0, 0, 1, 1),
        dependencies=(),
        dtype="fp32",
        warp=0,
    ):
        self.cta(cta)
        if isinstance(opcode, str):
            opcode = NativeOpcode(opcode)
        identity = len(self.ops)
        tid = self.tensor_ids[tensor] if tensor in self.tensor_ids else 0
        instruction = ArchInstruction(
            id=identity,
            name=f"{self.spec.symbol}_{identity}",
            op=opcode.value,
            pipeline=_PIPELINE[opcode],
            cta=cta,
            warp=cta * 4 + warp,
            tensor=tid,
            tile=tuple(region),
            dtype=dtype,
            leading_dim=(
                self.uses[tensor].tensor.shape[1]
                if opcode in (NativeOpcode.TMA_LOAD, NativeOpcode.TMA_STORE)
                else 0
            ),
            after_complete=tuple(sorted(set(dependencies))),
        )
        self.ops.append(instruction)
        self._last_cta_op[cta] = identity
        return identity

    def load(self, cta, tensor, region, dependencies=()):
        value = self.uses[tensor].tensor
        row, col, rows, cols = region
        if rows <= 0 or cols <= 0:
            raise ValueError("empty decode memory load")
        if (
            row < 0
            or col < 0
            or row + rows > value.shape[0]
            or col + cols > value.shape[1]
        ):
            raise ValueError(f"decode load outside {tensor}: {region} of {value.shape}")
        return self.emit(
            cta,
            NativeOpcode.TMA_LOAD,
            tensor,
            region,
            dependencies,
            dtype=value.dtype,
        )

    def store(self, cta, tensor, region, dependencies=()):
        value = self.uses[tensor].tensor
        row, col, rows, cols = region
        if rows <= 0 or cols <= 0:
            raise ValueError("empty decode memory store")
        if (
            row < 0
            or col < 0
            or row + rows > value.shape[0]
            or col + cols > value.shape[1]
        ):
            raise ValueError(
                f"decode store outside {tensor}: {region} of {value.shape}"
            )
        self.output_ids.add(self.tensor_ids[tensor])
        return self.emit(
            cta,
            NativeOpcode.TMA_STORE,
            tensor,
            region,
            dependencies,
            dtype=value.dtype,
        )

    def compute(self, cta, opcode, tensor, region, dependencies=(), dtype="fp32"):
        return self.emit(cta, opcode, tensor, region, dependencies, dtype=dtype)

    def build(self):
        work = self.spec.work
        dispatch = (
            (GemmWork, self.expand_gemm),
            (VectorWork, self.expand_vector),
            (QKVTransformWork, self.expand_qkv),
            (RouterTopKWork, self.expand_router_topk),
            (KVUpdateWork, self.expand_kv_update),
            (AttentionWork, self.expand_attention),
            (AssignmentWork, self.expand_assignment),
            (ExpertGemmWork, self.expand_expert_gemm),
            (CombineWork, self.expand_combine),
            (WeightedSumWork, self.expand_weighted_sum),
        )
        for type_, method in dispatch:
            if isinstance(work, type_):
                method(work)
                break
        else:
            raise TypeError(f"unsupported decode work: {type(work).__name__}")

        tensors = tuple(
            KernelTensor(
                self.tensor_ids[use.name],
                use.name,
                use.tensor.shape,
                (use.tensor.shape[1], 1),
                use.tensor.dtype,
                use.tensor.tier,
                preloaded=use.access.value == "read" and use.tensor.read_only,
                read_only=use.tensor.read_only,
            )
            for use in self.spec.tensors
        )
        if not self.ops or not self.ctas or not self.output_ids:
            raise ValueError(
                f"{self.spec.symbol}: expansion produced no executable graph"
            )
        kernel = ExecKernel(
            symbol=self.spec.symbol,
            source_hash=digest(self.native),
            tensors=tensors,
            ctas=tuple(self.ctas),
            ops=tuple(self.ops),
            partial_outputs=tuple(sorted(self.partial_ids)),
            outputs=tuple(sorted(self.output_ids)),
        )
        kernel.verify()
        return kernel

    def expand_gemm(self, work: GemmWork):
        row_tiles = ceil(work.rows / work.tile_m)
        column_tiles = ceil(work.columns / work.tile_n)
        for row_tile in range(row_tiles):
            row_base = row_tile * work.tile_m
            valid_rows = min(work.tile_m, work.rows - row_base)
            for column_tile in range(column_tiles):
                cta = row_tile * column_tiles + column_tile
                col_base = column_tile * work.tile_n
                valid_cols = min(work.tile_n, work.columns - col_base)
                last_mma = None
                for k_base in range(0, work.reduction, work.tile_k):
                    valid_k = min(work.tile_k, work.reduction - k_base)
                    a = self.load(
                        cta, work.input, (row_base, k_base, valid_rows, valid_k)
                    )
                    b = self.load(
                        cta, work.weight, (k_base, col_base, valid_k, valid_cols)
                    )
                    for n_offset in range(0, valid_cols, 8):
                        last_mma = self.compute(
                            cta,
                            NativeOpcode.MMA_SYNC,
                            work.output,
                            (
                                row_base,
                                col_base + n_offset,
                                work.tile_m,
                                8,
                            ),
                            (a, b) + ((last_mma,) if last_mma is not None else ()),
                        )
                self.store(
                    cta,
                    work.output,
                    (row_base, col_base, valid_rows, valid_cols),
                    (last_mma,),
                )

    def expand_vector(self, work: VectorWork):
        for row in range(work.rows):
            cta = row
            loads = []
            for name in work.inputs:
                shape = self.uses[name].tensor.shape
                source_row = 0 if shape[0] == 1 else row
                if source_row >= shape[0]:
                    raise ValueError(f"vector input {name} does not cover row {row}")
                loads.append(self.load(cta, name, (source_row, 0, 1, shape[1])))
            stage_results = []
            for stage_index, stage in enumerate(work.stages):
                dependencies = tuple(
                    stage_results[index] for index in stage.dependencies
                )
                if not dependencies and stage_index:
                    dependencies = (stage_results[stage_index - 1],)
                dependencies += tuple(loads[index] for index in stage.input_indices)
                previous = dependencies
                if stage.operation == FlowOperator.SILU:
                    source = tuple(dependencies)
                    previous = (
                        self.compute(
                            cta,
                            NativeOpcode.SIMD_SUB,
                            work.outputs[0],
                            (0, 0, 1, stage.elements),
                            source,
                        ),
                    )
                    previous = (
                        self.compute(
                            cta,
                            NativeOpcode.SFU_EXP,
                            work.outputs[0],
                            (0, 0, 1, stage.elements),
                            previous,
                        ),
                    )
                    previous = (
                        self.compute(
                            cta,
                            NativeOpcode.SIMD_ADD,
                            work.outputs[0],
                            (0, 0, 1, stage.elements),
                            (*previous, *source),
                        ),
                    )
                    previous = (
                        self.compute(
                            cta,
                            NativeOpcode.ACC_O_DIV,
                            work.outputs[0],
                            (0, 0, 1, stage.elements),
                            (*previous, *source),
                        ),
                    )
                else:
                    for _ in range(stage.repeats):
                        previous = (
                            self.compute(
                                cta,
                                _vector_opcode(stage.operation),
                                work.outputs[0],
                                (0, 0, 1, stage.elements),
                                previous,
                            ),
                        )
                stage_results.append(previous[0])
            previous = (stage_results[-1],)
            output_width = (
                work.output_width or self.uses[work.outputs[0]].tensor.shape[1]
            )
            self.store(cta, work.outputs[0], (row, 0, 1, output_width), previous)

    def expand_qkv(self, work: QKVTransformWork):
        for request in range(work.batch):
            for q_head in range(work.query_heads):
                cta = request * work.query_heads + q_head
                qkv = self.load(
                    cta, work.qkv, (request, q_head * work.head_dim, 1, work.head_dim)
                )
                qw = self.load(cta, work.q_norm_weight, (0, 0, 1, work.head_dim))
                q = self._norm_rope(
                    cta,
                    work.query_output,
                    q_head + request * work.query_heads,
                    work.positions[request],
                    (qkv, qw),
                    work.head_dim,
                )
                self.store(
                    cta,
                    work.query_output,
                    (request * work.query_heads + q_head, 0, 1, work.head_dim),
                    (q,),
                )
                if q_head % (work.query_heads // work.kv_heads) != 0:
                    continue
                kv_head = q_head // (work.query_heads // work.kv_heads)
                k_source = self.load(
                    cta,
                    work.qkv,
                    (
                        request,
                        work.query_heads * work.head_dim + kv_head * work.head_dim,
                        1,
                        work.head_dim,
                    ),
                )
                kw = self.load(cta, work.k_norm_weight, (0, 0, 1, work.head_dim))
                k = self._norm_rope(
                    cta,
                    work.key_output,
                    request * work.kv_heads + kv_head,
                    work.positions[request],
                    (k_source, kw),
                    work.head_dim,
                )
                self.store(
                    cta,
                    work.key_output,
                    (request * work.kv_heads + kv_head, 0, 1, work.head_dim),
                    (k,),
                )
                v = self.load(
                    cta,
                    work.qkv,
                    (request, 4608 + kv_head * work.head_dim, 1, work.head_dim),
                )
                identity = self.compute(
                    cta,
                    NativeOpcode.SIMD_IDENTITY,
                    work.value_output,
                    (0, 0, 1, work.head_dim),
                    (v,),
                )
                self.store(
                    cta,
                    work.value_output,
                    (request * work.kv_heads + kv_head, 0, 1, work.head_dim),
                    (identity,),
                )

    def _norm_rope(self, cta, output, output_row, position, dependencies, head_dim):
        square = self.compute(
            cta, NativeOpcode.SIMD_MUL, output, (0, 0, 1, head_dim), dependencies
        )
        total = self.compute(
            cta, NativeOpcode.REDUCE_SUM, output, (0, 0, 1, head_dim), (square,)
        )
        inverse = self.compute(
            cta, NativeOpcode.SFU_RSQRT, output, (0, 0, 1, 1), (total,)
        )
        scaled = self.compute(
            cta,
            NativeOpcode.SIMD_MUL,
            output,
            (0, 0, 1, head_dim),
            (inverse, dependencies[0]),
        )
        normalized = self.compute(
            cta,
            NativeOpcode.SIMD_MUL,
            output,
            (0, 0, 1, head_dim),
            (scaled, dependencies[-1]),
        )
        sine = self.compute(
            cta, NativeOpcode.SFU_SIN, output, (0, 0, 1, head_dim // 2), (normalized,)
        )
        cosine = self.compute(
            cta, NativeOpcode.SFU_COS, output, (0, 0, 1, head_dim // 2), (normalized,)
        )
        return self.compute(
            cta,
            NativeOpcode.SIMD_FMA,
            output,
            (0, 0, 1, head_dim),
            (sine, cosine, normalized),
        )

    def expand_router_topk(self, work: RouterTopKWork):
        for row in range(work.batch):
            cta = row
            logits = self.load(cta, work.logits, (row, 0, 1, work.experts))
            maximum = self.compute(
                cta,
                NativeOpcode.REDUCE_MAX,
                work.logits,
                (0, 0, 1, work.experts),
                (logits,),
            )
            exponentials = self.compute(
                cta,
                NativeOpcode.SFU_EXP,
                work.logits,
                (0, 0, 1, work.experts),
                (logits, maximum),
            )
            total = self.compute(
                cta,
                NativeOpcode.REDUCE_SUM,
                work.logits,
                (0, 0, 1, work.experts),
                (exponentials,),
            )
            selected = self.compute(
                cta,
                NativeOpcode.SIMD_MAX,
                work.expert_ids,
                (0, 0, 1, work.experts),
                (logits, total),
            )
            probabilities = self.compute(
                cta,
                NativeOpcode.SFU_DIVIDE,
                work.probabilities,
                (0, 0, 1, work.top_k),
                (selected, exponentials, total),
            )
            self.store(cta, work.expert_ids, (row, 0, 1, work.top_k), (selected,))
            self.store(
                cta, work.probabilities, (row, 0, 1, work.top_k), (probabilities,)
            )

    def expand_kv_update(self, work: KVUpdateWork):
        self.partial_ids.update(
            self.tensor_ids[append.key_cache] for append in work.appends
        )
        self.partial_ids.update(
            self.tensor_ids[append.value_cache] for append in work.appends
        )
        for cta, append in enumerate(work.appends):
            row = append.request * 4 + append.head
            key = self.load(cta, append.new_key, (row, 0, 1, work.head_dim))
            value = self.load(cta, append.new_value, (row, 0, 1, work.head_dim))
            self.store(
                cta,
                append.key_cache,
                (append.context_length, 0, 1, work.head_dim),
                (key,),
            )
            self.store(
                cta,
                append.value_cache,
                (append.context_length, 0, 1, work.head_dim),
                (value,),
            )

    def expand_attention(self, work: AttentionWork):
        kv_heads_per_query = work.query_heads // work.kv_heads
        chunk_by_request = {request: [] for request in range(len(work.requests))}
        for chunk in work.chunks:
            chunk_by_request[chunk.request].append(chunk)
        cta_id = 0
        for request_id, request in enumerate(work.requests):
            for q_head in range(work.query_heads):
                kv_head = q_head // kv_heads_per_query
                query_row = request_id * work.query_heads + q_head
                for chunk in sorted(
                    chunk_by_request[request_id], key=lambda item: item.split
                ):
                    cta = cta_id
                    cta_id += 1
                    query = self.load(cta, work.query, (query_row, 0, 1, work.head_dim))
                    state_max = self.compute(
                        cta, NativeOpcode.SIMD_FILL, work.query, (0, 0, 1, 1), (query,)
                    )
                    state_sum = self.compute(
                        cta,
                        NativeOpcode.SIMD_FILL,
                        work.query,
                        (0, 0, 1, 1),
                        (state_max,),
                    )
                    state_output = self.compute(
                        cta,
                        NativeOpcode.SIMD_FILL,
                        work.query,
                        (0, 0, 1, work.head_dim),
                        (state_sum,),
                    )
                    for tile_start in range(chunk.start, chunk.end, work.kv_tile):
                        valid = min(work.kv_tile, chunk.end - tile_start)
                        key_name = request.key_caches[kv_head]
                        value_name = request.value_caches[kv_head]
                        key = self.load(
                            cta, key_name, (tile_start, 0, valid, work.head_dim)
                        )
                        value = self.load(
                            cta, value_name, (tile_start, 0, valid, work.head_dim)
                        )
                        scores = self.compute(
                            cta,
                            NativeOpcode.SIMD_FMA,
                            work.query,
                            (tile_start, 0, valid, work.head_dim),
                            (query, key),
                        )
                        scores = self.compute(
                            cta,
                            NativeOpcode.SCORES_SCALE,
                            work.query,
                            (0, 0, valid, 1),
                            (scores,),
                        )
                        tile_max = self.compute(
                            cta,
                            NativeOpcode.SIMD_MAX,
                            work.query,
                            (0, 0, valid, 1),
                            (scores, state_max),
                        )
                        alpha = self.compute(
                            cta,
                            NativeOpcode.SFU_EXP,
                            work.query,
                            (0, 0, 1, 1),
                            (state_max, tile_max),
                        )
                        probs = self.compute(
                            cta,
                            NativeOpcode.EXP2,
                            work.query,
                            (0, 0, valid, 1),
                            (scores, tile_max),
                        )
                        tile_sum = self.compute(
                            cta,
                            NativeOpcode.REDUCE_SUM,
                            work.query,
                            (0, 0, valid, 1),
                            (probs,),
                        )
                        state_sum = self.compute(
                            cta,
                            NativeOpcode.SIMD_FMA,
                            work.query,
                            (0, 0, 1, 1),
                            (alpha, state_sum, tile_sum),
                        )
                        pv = self.compute(
                            cta,
                            NativeOpcode.SIMD_FMA,
                            work.query,
                            (tile_start, 0, valid, work.head_dim),
                            (probs, value),
                        )
                        state_output = self.compute(
                            cta,
                            NativeOpcode.SIMD_FMA,
                            work.query,
                            (0, 0, 1, work.head_dim),
                            (alpha, state_output, pv),
                        )
                        state_max = tile_max
                    normalized = self.compute(
                        cta,
                        NativeOpcode.SFU_DIVIDE,
                        work.query,
                        (0, 0, 1, work.head_dim),
                        (state_output, state_sum),
                    )
                    lse = self.compute(
                        cta,
                        NativeOpcode.LOGSUM,
                        work.query,
                        (0, 0, 1, 1),
                        (state_max, state_sum),
                    )
                    if work.split_count > 1:
                        partial_row = (
                            request_id * work.query_heads + q_head
                        ) * work.split_count + chunk.split
                        self.store(
                            cta,
                            work.partial_output,
                            (partial_row, 0, 1, work.head_dim),
                            (normalized,),
                        )
                        self.store(
                            cta, work.partial_lse, (partial_row, 0, 1, 1), (lse,)
                        )
                    else:
                        self.store(
                            cta,
                            work.output,
                            (request_id, q_head * work.head_dim, 1, work.head_dim),
                            (normalized,),
                        )

    def expand_assignment(self, work: AssignmentWork):
        cta = 0
        route_ids = self.load(
            cta, work.expert_ids, (0, 0, self.uses[work.expert_ids].tensor.shape[0], 8)
        )
        route_probs = self.load(
            cta,
            work.expert_probabilities,
            (0, 0, self.uses[work.expert_probabilities].tensor.shape[0], 8),
        )
        activations = self.load(
            cta,
            work.activation,
            (0, 0, self.uses[work.activation].tensor.shape[0], work.hidden_size),
        )
        count = len(work.assignments)
        histogram = self.compute(
            cta,
            NativeOpcode.SIMD_ADD,
            work.sorted_assignments,
            (0, 0, count, 128),
            (route_ids,),
        )
        prefix = self.compute(
            cta,
            NativeOpcode.REDUCE_SUM,
            work.sorted_assignments,
            (0, 0, 128, 1),
            (histogram,),
        )
        gather = self.compute(
            cta,
            NativeOpcode.SIMD_IDENTITY,
            work.sorted_activation,
            (0, 0, count, work.hidden_size),
            (route_ids, activations, prefix),
        )
        metadata = self.compute(
            cta,
            NativeOpcode.SIMD_IDENTITY,
            work.sorted_assignments,
            (0, 0, count, 3),
            (gather, route_ids),
        )
        probabilities = self.compute(
            cta,
            NativeOpcode.SIMD_IDENTITY,
            work.sorted_probabilities,
            (0, 0, count, 1),
            (route_probs, gather),
        )
        self.store(
            cta, work.sorted_activation, (0, 0, count, work.hidden_size), (gather,)
        )
        self.store(cta, work.sorted_assignments, (0, 0, count, 3), (metadata,))
        self.store(cta, work.sorted_probabilities, (0, 0, count, 1), (probabilities,))

    def expand_expert_gemm(self, work: ExpertGemmWork):
        for group in work.groups:
            row_count = len(group.assignments)
            row_tiles = ceil(row_count / work.tile_m)
            column_tiles = ceil(work.columns / work.tile_n)
            for row_tile in range(row_tiles):
                local_row = row_tile * work.tile_m
                valid_rows = min(work.tile_m, row_count - local_row)
                if work.variant == "grouped":
                    input_row = group.row_start + local_row
                    output_row = group.row_start + local_row
                elif work.phase == "gemm1":
                    input_row = group.assignments[local_row].token
                    output_row = (
                        group.assignments[local_row].token * 8
                        + group.assignments[local_row].slot
                    )
                else:
                    input_row = (
                        group.assignments[local_row].token * 8
                        + group.assignments[local_row].slot
                    )
                    output_row = input_row
                for column_tile in range(column_tiles):
                    cta = len(self.ctas)
                    col_base = column_tile * work.tile_n
                    valid_cols = min(work.tile_n, work.columns - col_base)
                    last_mma = None
                    for k_base in range(0, work.reduction, work.tile_k):
                        valid_k = min(work.tile_k, work.reduction - k_base)
                        a = self.load(
                            cta, work.input, (input_row, k_base, valid_rows, valid_k)
                        )
                        weight_name = (
                            group.gate_up_weight
                            if work.phase == "gemm1"
                            else group.down_weight
                        )
                        weight = self.load(
                            cta, weight_name, (k_base, col_base, valid_k, valid_cols)
                        )
                        for n_offset in range(0, valid_cols, 8):
                            last_mma = self.compute(
                                cta,
                                NativeOpcode.MMA_SYNC,
                                work.output,
                                (
                                    output_row,
                                    col_base + n_offset,
                                    work.tile_m,
                                    8,
                                ),
                                (a, weight)
                                + ((last_mma,) if last_mma is not None else ()),
                            )
                    self.store(
                        cta,
                        work.output,
                        (output_row, col_base, valid_rows, valid_cols),
                        (last_mma,),
                    )

    def expand_combine(self, work: CombineWork):
        for request in range(work.requests):
            for head in range(work.heads):
                cta = request * work.heads + head
                partial_rows = []
                lse_rows = []
                for split in range(work.splits):
                    row = (request * work.heads + head) * work.splits + split
                    partial_rows.append(
                        self.load(cta, work.partial_output, (row, 0, 1, work.head_dim))
                    )
                    lse_rows.append(self.load(cta, work.partial_lse, (row, 0, 1, 1)))
                maximum = self.compute(
                    cta,
                    NativeOpcode.REDUCE_MAX,
                    work.partial_lse,
                    (0, 0, work.splits, 1),
                    tuple(lse_rows),
                )
                exponentials = self.compute(
                    cta,
                    NativeOpcode.SFU_EXP,
                    work.partial_lse,
                    (0, 0, work.splits, 1),
                    (maximum, *lse_rows),
                )
                normalizer = self.compute(
                    cta,
                    NativeOpcode.REDUCE_SUM,
                    work.partial_lse,
                    (0, 0, work.splits, 1),
                    (exponentials,),
                )
                weighted = self.compute(
                    cta,
                    NativeOpcode.SIMD_FMA,
                    work.output,
                    (0, 0, work.splits, work.head_dim),
                    (*partial_rows, exponentials),
                )
                output = self.compute(
                    cta,
                    NativeOpcode.SFU_DIVIDE,
                    work.output,
                    (0, 0, 1, work.head_dim),
                    (weighted, normalizer),
                )
                self.store(
                    cta,
                    work.output,
                    (request, head * work.head_dim, 1, work.head_dim),
                    (output,),
                )

    def expand_weighted_sum(self, work: WeightedSumWork):
        for token in range(work.batch):
            cta = token
            pieces = []
            probability_rows = []
            assignment_rows = []
            if work.direct_order:
                expert_ids = self.load(cta, work.assignments, (token, 0, 1, work.top_k))
                probabilities = self.load(
                    cta, work.expert_probabilities, (token, 0, 1, work.top_k)
                )
                for slot in range(work.top_k):
                    row = token * work.top_k + slot
                    pieces.append(
                        self.load(
                            cta, work.expert_output, (row, 0, 1, work.hidden_size)
                        )
                    )
            else:
                row_by_route = {
                    (assignment.token, assignment.slot): row
                    for row, assignment in enumerate(work.grouped_assignments)
                }
                rows = [row_by_route[(token, slot)] for slot in range(work.top_k)]
                for row in rows:
                    pieces.append(
                        self.load(
                            cta, work.expert_output, (row, 0, 1, work.hidden_size)
                        )
                    )
                    probability_rows.append(
                        self.load(cta, work.expert_probabilities, (row, 0, 1, 1))
                    )
                    assignment_rows.append(
                        self.load(cta, work.assignments, (row, 0, 1, 3))
                    )
                expert_ids = tuple(assignment_rows)
                probabilities = tuple(probability_rows)
            dependencies = (
                *pieces,
                *((probabilities,) if work.direct_order else probabilities),
                *((expert_ids,) if work.direct_order else expert_ids),
            )
            result = self.compute(
                cta,
                NativeOpcode.SIMD_FMA,
                work.output,
                (0, 0, work.top_k, work.hidden_size),
                dependencies,
            )
            self.store(cta, work.output, (token, 0, 1, work.hidden_size), (result,))


def _vector_opcode(operation):
    return {
        FlowOperator.ADD: NativeOpcode.SIMD_ADD,
        FlowOperator.SUBTRACT: NativeOpcode.SIMD_SUB,
        FlowOperator.MULTIPLY: NativeOpcode.SIMD_MUL,
        FlowOperator.FUSED_MULTIPLY_ADD: NativeOpcode.SIMD_FMA,
        FlowOperator.MAX_REDUCTION: NativeOpcode.REDUCE_MAX,
        FlowOperator.SUM_REDUCTION: NativeOpcode.REDUCE_SUM,
        FlowOperator.EXPONENTIAL: NativeOpcode.SFU_EXP,
        FlowOperator.LOGARITHM: NativeOpcode.SFU_LOG,
        FlowOperator.RECIPROCAL_SQRT: NativeOpcode.SFU_RSQRT,
        FlowOperator.SIN: NativeOpcode.SFU_SIN,
        FlowOperator.COS: NativeOpcode.SFU_COS,
        FlowOperator.NORMALIZE: NativeOpcode.ACC_O_DIV,
        FlowOperator.TOPK: NativeOpcode.SIMD_MAX,
        FlowOperator.SILU: NativeOpcode.SFU_DIVIDE,
        FlowOperator.WEIGHTED_SUM: NativeOpcode.SIMD_FMA,
        FlowOperator.CAST: NativeOpcode.SIMD_CAST,
        FlowOperator.IDENTITY: NativeOpcode.SIMD_IDENTITY,
        FlowOperator.FILL: NativeOpcode.SIMD_FILL,
    }[operation]


def expand(native, num_sms):
    return KernelExpander(native, num_sms).build()
