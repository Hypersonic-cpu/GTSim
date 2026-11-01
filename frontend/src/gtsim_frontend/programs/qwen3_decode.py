"""Declarative single-step Qwen3 MoE layer construction."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from ..ir.common import Tensor
from ..ir.decode import (
    AccessMode,
    AttentionChunk,
    AttentionRequest,
    AttentionWork,
    AssignmentWork,
    CombineWork,
    DecodeAlgorithm,
    DecodeInvocation,
    DecodeProgram,
    ExpertAssignment,
    ExpertGemmWork,
    ExpertGroup,
    GemmWork,
    KernelFamily,
    KVAppend,
    KVUpdateWork,
    QKVTransformWork,
    Qwen3RoutingToken,
    RouterTopKWork,
    TensorUse,
    VectorStage,
    VectorWork,
    WeightedSumWork,
    FlowOperator,
)


HIDDEN = 2048
Q_HEADS = 32
KV_HEADS = 4
HEAD_DIM = 128
TOP_K = 8
EXPERTS = 128
INTERMEDIATE = 768
DTYPE = "bf16"


class LayerBuilder:
    def __init__(
        self, batch_size, context_lengths, split_count, layer, routes, external_tensors
    ):
        self.batch = batch_size
        self.contexts = tuple(context_lengths)
        self.split_count = split_count
        self.layer = layer
        self.routes = tuple(routes)
        self.external_tensors = external_tensors
        self.tensors: dict[str, Tensor] = {}
        self.launches: list[DecodeInvocation] = []
        self.commands: list[tuple[str, str]] = []
        self.metadata = {}

    def tensor(self, name, shape, dtype=DTYPE, read_only=False):
        identity = f"l{self.layer}.{name}"
        if identity in self.tensors:
            old = self.tensors[identity]
            if (old.shape, old.dtype) != (tuple(shape), dtype):
                raise ValueError(
                    f"incompatible repeated tensor declaration: {identity}"
                )
            return identity
        value = Tensor(identity, tuple(shape), dtype, read_only=read_only)
        value.verify()
        self.tensors[identity] = value
        return identity

    def weight(self, name, shape, dtype=DTYPE):
        return self.tensor(f"weight.{name}", shape, dtype, read_only=True)

    def invoke(self, name, variant, family, work, specs):
        algorithm_tensors = tuple(
            TensorUse(
                local,
                Tensor(
                    local,
                    self.tensors.get(
                        global_name, self.external_tensors.get(global_name)
                    ).shape,
                    self.tensors.get(
                        global_name, self.external_tensors.get(global_name)
                    ).dtype,
                    self.tensors.get(
                        global_name, self.external_tensors.get(global_name)
                    ).tier,
                    read_only=(access == AccessMode.READ),
                ),
                access,
                broadcast_rows,
            )
            for local, global_name, access, broadcast_rows in specs
        )
        algorithm = DecodeAlgorithm(
            symbol=f"l{self.layer}_{name}",
            family=family,
            layer=self.layer,
            variant=variant,
            tensors=algorithm_tensors,
            work=work,
        )
        bindings = tuple((local, global_name) for local, global_name, _, _ in specs)
        identity = f"l{self.layer}_{name}"
        invocation = DecodeInvocation(
            identity, self.layer, name, variant, algorithm, bindings
        )
        algorithm.verify()
        self.launches.append(invocation)
        self.commands.append(("LaunchKernel", identity))
        return identity

    @staticmethod
    def read(name, global_name, broadcast=False):
        return name, global_name, AccessMode.READ, broadcast

    @staticmethod
    def write(name, global_name):
        return name, global_name, AccessMode.WRITE, False

    def build(self, input_name):
        l = self.layer
        b = self.batch
        contexts = self.contexts

        x = input_name
        norm_w = self.weight("input_norm", (1, HIDDEN))
        normalized = self.tensor("input_norm", (b, HIDDEN))
        norm_work = VectorWork(
            inputs=("x", "weight"),
            outputs=("normalized",),
            rows=b,
            columns=HIDDEN,
            tile_rows=1,
            tile_columns=HIDDEN,
            stages=(
                VectorStage(FlowOperator.MULTIPLY, HIDDEN, input_indices=(0, 0)),
                VectorStage(FlowOperator.SUM_REDUCTION, HIDDEN, dependencies=(0,)),
                VectorStage(FlowOperator.RECIPROCAL_SQRT, 1, dependencies=(1,)),
                VectorStage(
                    FlowOperator.MULTIPLY, HIDDEN, input_indices=(0,), dependencies=(2,)
                ),
                VectorStage(
                    FlowOperator.MULTIPLY, HIDDEN, input_indices=(1,), dependencies=(3,)
                ),
            ),
        )
        self.invoke(
            "input_rmsnorm",
            "rmsnorm_fp32",
            KernelFamily.VECTOR,
            norm_work,
            (
                self.read("x", x),
                self.read("weight", norm_w, True),
                self.write("normalized", normalized),
            ),
        )

        qkv_weight = self.weight("qkv", (HIDDEN, 5120))
        qkv = self.tensor("qkv_projection", (b, 5120))
        self.invoke(
            "qkv_projection",
            "skinny_m16_mma_sync",
            KernelFamily.SKINNY_GEMM,
            GemmWork("input", "weight", "output", b, 5120, HIDDEN, tile_n=64),
            (
                self.read("input", normalized),
                self.read("weight", qkv_weight),
                self.write("output", qkv),
            ),
        )

        q_norm = self.weight("q_norm", (1, HEAD_DIM))
        k_norm = self.weight("k_norm", (1, HEAD_DIM))
        q = self.tensor("query", (b * Q_HEADS, HEAD_DIM))
        k = self.tensor("key_new", (b * KV_HEADS, HEAD_DIM))
        v = self.tensor("value_new", (b * KV_HEADS, HEAD_DIM))
        qkv_work = QKVTransformWork(
            "qkv",
            "q_norm_weight",
            "k_norm_weight",
            "query",
            "key",
            "value",
            b,
            contexts,
        )
        self.invoke(
            "qkv_norm_rope",
            "qk_rmsnorm_then_rope",
            KernelFamily.VECTOR,
            qkv_work,
            (
                self.read("qkv", qkv),
                self.read("q_norm_weight", q_norm, True),
                self.read("k_norm_weight", k_norm, True),
                self.write("query", q),
                self.write("key", k),
                self.write("value", v),
            ),
        )

        key_caches = []
        value_caches = []
        appends = []
        kv_specs = [self.read("new_key", k), self.read("new_value", v)]
        kv_output_names = []
        for request, context in enumerate(contexts):
            request_keys, request_values = [], []
            for head in range(KV_HEADS):
                key_cache = self.tensor(
                    f"kv.key.r{request}.h{head}", (context + 1, HEAD_DIM)
                )
                value_cache = self.tensor(
                    f"kv.value.r{request}.h{head}", (context + 1, HEAD_DIM)
                )
                key_local = f"key_r{request}_h{head}"
                value_local = f"value_r{request}_h{head}"
                kv_specs.extend(
                    (
                        self.write(key_local, key_cache),
                        self.write(value_local, value_cache),
                    )
                )
                kv_output_names.extend((key_local, value_local))
                request_keys.append(key_local)
                request_values.append(value_local)
                appends.append(
                    KVAppend(
                        request,
                        head,
                        context,
                        "new_key",
                        "new_value",
                        key_local,
                        value_local,
                    )
                )
            key_caches.append(tuple(request_keys))
            value_caches.append(tuple(request_values))
        self.invoke(
            "kv_cache_update",
            "append_one_token_per_request_kv_head",
            KernelFamily.KV_UPDATE,
            KVUpdateWork(tuple(appends)),
            tuple(kv_specs),
        )

        split = self.split_count
        if split > 1:
            attention_variant = "split_kv"
            partial = self.tensor(
                "attention.partial", (b * Q_HEADS * split, HEAD_DIM), "fp32"
            )
            partial_lse = self.tensor(
                "attention.partial_lse", (b * Q_HEADS * split, 1), "fp32"
            )
            attention_output = self.tensor("attention.heads", (b, Q_HEADS * HEAD_DIM))
        else:
            attention_variant = "non_split"
            partial = ""
            partial_lse = ""
            attention_output = self.tensor("attention.heads", (b, Q_HEADS * HEAD_DIM))
        chunks = []
        for request, context in enumerate(contexts):
            effective = context + 1
            for split_id in range(split):
                start = effective * split_id // split
                end = effective * (split_id + 1) // split
                chunks.append(AttentionChunk(request, split_id, start, end))
        attention_requests = tuple(
            AttentionRequest(request, key_caches[request], value_caches[request])
            for request in range(b)
        )
        attention_specs = [self.read("query", q)]
        for request in range(b):
            for head in range(KV_HEADS):
                attention_specs.extend(
                    (
                        self.read(
                            key_caches[request][head],
                            self.tensors[f"l{l}.kv.key.r{request}.h{head}"].id,
                        ),
                        self.read(
                            value_caches[request][head],
                            self.tensors[f"l{l}.kv.value.r{request}.h{head}"].id,
                        ),
                    )
                )
        if split > 1:
            attention_specs.extend(
                (self.write("partial", partial), self.write("partial_lse", partial_lse))
            )
        else:
            attention_specs.append(self.write("output", attention_output))
        attention_work = AttentionWork(
            "query",
            "output" if split == 1 else "",
            attention_requests,
            contexts,
            split,
            tuple(chunks),
            "partial" if split > 1 else "",
            "partial_lse" if split > 1 else "",
            kv_tile=256,
        )
        self.invoke(
            "fa3_decode",
            attention_variant,
            KernelFamily.FA3_DECODE,
            attention_work,
            tuple(attention_specs),
        )
        if split > 1:
            combined = self.tensor("attention.combined", (b, Q_HEADS * HEAD_DIM))
            self.invoke(
                "fa3_combine",
                "split_combine",
                KernelFamily.FA3_COMBINE,
                CombineWork("partial", "partial_lse", "output", b, Q_HEADS, split),
                (
                    self.read("partial", partial),
                    self.read("partial_lse", partial_lse),
                    self.write("output", combined),
                ),
            )
            attention_output = combined

        o_weight = self.weight("attention_output", (Q_HEADS * HEAD_DIM, HIDDEN))
        attention_projection = self.tensor("attention_projection", (b, HIDDEN))
        self.invoke(
            "attention_output_projection",
            "skinny_m16_mma_sync",
            KernelFamily.SKINNY_GEMM,
            GemmWork(
                "input", "weight", "output", b, HIDDEN, Q_HEADS * HEAD_DIM, tile_n=64
            ),
            (
                self.read("input", attention_output),
                self.read("weight", o_weight),
                self.write("output", attention_projection),
            ),
        )

        post_norm_w = self.weight("post_attention_norm", (1, HIDDEN))
        post_norm = self.tensor("post_attention_norm", (b, HIDDEN))
        post_work = VectorWork(
            inputs=("residual", "attention", "weight"),
            outputs=("normalized",),
            rows=b,
            columns=HIDDEN,
            tile_rows=1,
            tile_columns=HIDDEN,
            stages=(
                VectorStage(FlowOperator.ADD, HIDDEN, input_indices=(0, 1)),
                VectorStage(FlowOperator.MULTIPLY, HIDDEN, dependencies=(0,)),
                VectorStage(FlowOperator.SUM_REDUCTION, HIDDEN, dependencies=(1,)),
                VectorStage(FlowOperator.RECIPROCAL_SQRT, 1, dependencies=(2,)),
                VectorStage(FlowOperator.MULTIPLY, HIDDEN, dependencies=(0, 3)),
                VectorStage(
                    FlowOperator.MULTIPLY, HIDDEN, input_indices=(2,), dependencies=(4,)
                ),
            ),
        )
        self.invoke(
            "post_attention_residual_rmsnorm",
            "residual_add_then_rmsnorm_fp32",
            KernelFamily.VECTOR,
            post_work,
            (
                self.read("residual", x),
                self.read("attention", attention_projection),
                self.read("weight", post_norm_w, True),
                self.write("normalized", post_norm),
            ),
        )

        router_weight = self.weight("router", (HIDDEN, EXPERTS))
        logits = self.tensor("router_logits", (b, EXPERTS))
        self.invoke(
            "router_projection",
            "skinny_m16_mma_sync_narrow",
            KernelFamily.SKINNY_GEMM,
            GemmWork("input", "weight", "output", b, EXPERTS, HIDDEN, tile_n=64),
            (
                self.read("input", post_norm),
                self.read("weight", router_weight),
                self.write("output", logits),
            ),
        )
        expert_ids = self.tensor("route.expert_ids", (b, TOP_K), "int32")
        expert_probs = self.tensor("route.probabilities", (b, TOP_K))
        self.invoke(
            "router_softmax_topk",
            "softmax_top8_renormalized",
            KernelFamily.VECTOR,
            RouterTopKWork("logits", "expert_ids", "probabilities", self.routes, b),
            (
                self.read("logits", logits),
                self.write("expert_ids", expert_ids),
                self.write("probabilities", expert_probs),
            ),
        )

        assignments = tuple(
            ExpertAssignment(token, slot, expert, probability)
            for token, route in enumerate(self.routes)
            for slot, (expert, probability) in enumerate(
                zip(route.expert_ids, route.probabilities)
            )
        )
        assignment_by_expert = {}
        for item in assignments:
            assignment_by_expert.setdefault(item.expert, []).append(item)
        expert_counts = {
            expert: len(rows) for expert, rows in sorted(assignment_by_expert.items())
        }
        grouped = b > 1
        sorted_activation = (
            self.tensor("moe.sorted_activation", (b * TOP_K, HIDDEN)) if grouped else ""
        )
        sorted_assignments = (
            self.tensor("moe.sorted_assignment", (b * TOP_K, 3), "int32")
            if grouped
            else ""
        )
        sorted_probabilities = (
            self.tensor("moe.sorted_probability", (b * TOP_K, 1)) if grouped else ""
        )
        groups = []
        row = 0
        for expert, rows in sorted(assignment_by_expert.items()):
            groups.append(
                ExpertGroup(
                    expert,
                    row if grouped else rows[0].token * TOP_K + rows[0].slot,
                    tuple(rows),
                    "",
                    "",
                )
            )
            row += len(rows)

        if grouped:
            self.invoke(
                "expert_assignment",
                "histogram_prefix_sum_expert_grouping",
                KernelFamily.EXPERT_ASSIGNMENT,
                AssignmentWork(
                    "activation",
                    "expert_ids",
                    "expert_probabilities",
                    "sorted_activation",
                    "sorted_assignments",
                    "sorted_probabilities",
                    assignments,
                    tuple(groups),
                ),
                (
                    self.read("activation", post_norm),
                    self.read("expert_ids", expert_ids),
                    self.read("expert_probabilities", expert_probs),
                    self.write("sorted_activation", sorted_activation),
                    self.write("sorted_assignments", sorted_assignments),
                    self.write("sorted_probabilities", sorted_probabilities),
                ),
            )
            expert_input = sorted_activation
            assignment_tensor = sorted_assignments
            probability_tensor = sorted_probabilities
            variant = "grouped"
            groups = [
                ExpertGroup(
                    group.expert,
                    group.row_start,
                    group.assignments,
                    f"w1_e{group.expert}",
                    f"w2_e{group.expert}",
                )
                for group in groups
            ]
            for group in groups:
                self.weight(
                    f"expert.{group.expert}.gate_up", (HIDDEN, 2 * INTERMEDIATE)
                )
                self.weight(f"expert.{group.expert}.down", (INTERMEDIATE, HIDDEN))
        else:
            expert_input = post_norm
            assignment_tensor = expert_ids
            probability_tensor = expert_probs
            variant = "direct"
            groups = [
                ExpertGroup(
                    group.expert,
                    group.row_start,
                    group.assignments,
                    f"w1_e{group.expert}",
                    f"w2_e{group.expert}",
                )
                for group in groups
            ]
            for group in groups:
                self.weight(
                    f"expert.{group.expert}.gate_up", (HIDDEN, 2 * INTERMEDIATE)
                )
                self.weight(f"expert.{group.expert}.down", (INTERMEDIATE, HIDDEN))

        expert_gate_up = self.tensor("moe.gate_up", (b * TOP_K, 2 * INTERMEDIATE))
        self.invoke(
            "expert_gemm1",
            f"{variant}_m16_grouped_tile",
            KernelFamily.EXPERT_GEMM1,
            ExpertGemmWork(
                variant,
                "gemm1",
                "input",
                "output",
                b * TOP_K,
                2 * INTERMEDIATE,
                HIDDEN,
                tuple(groups),
            ),
            tuple(
                [self.read("input", expert_input), self.write("output", expert_gate_up)]
                + [
                    self.read(
                        group.gate_up_weight,
                        self.weight(
                            f"expert.{group.expert}.gate_up", (HIDDEN, 2 * INTERMEDIATE)
                        ),
                    )
                    for group in groups
                ]
            ),
        )

        swiglu = self.tensor("moe.swiglu", (b * TOP_K, INTERMEDIATE))
        self.invoke(
            "swiglu",
            "silu_gate_times_up",
            KernelFamily.SWIGLU,
            VectorWork(
                ("gate_up",),
                ("output",),
                b * TOP_K,
                2 * INTERMEDIATE,
                1,
                2 * INTERMEDIATE,
                (
                    VectorStage(FlowOperator.SILU, INTERMEDIATE, input_indices=(0,)),
                    VectorStage(
                        FlowOperator.MULTIPLY,
                        INTERMEDIATE,
                        dependencies=(0,),
                        input_indices=(0,),
                    ),
                ),
                output_width=INTERMEDIATE,
            ),
            (self.read("gate_up", expert_gate_up), self.write("output", swiglu)),
        )

        expert_output = self.tensor("moe.expert_output", (b * TOP_K, HIDDEN))
        self.invoke(
            "expert_gemm2",
            f"{variant}_m16_grouped_tile",
            KernelFamily.EXPERT_GEMM2,
            ExpertGemmWork(
                variant,
                "gemm2",
                "input",
                "output",
                b * TOP_K,
                HIDDEN,
                INTERMEDIATE,
                tuple(groups),
            ),
            tuple(
                [self.read("input", swiglu), self.write("output", expert_output)]
                + [
                    self.read(
                        group.down_weight,
                        self.weight(
                            f"expert.{group.expert}.down", (INTERMEDIATE, HIDDEN)
                        ),
                    )
                    for group in groups
                ]
            ),
        )

        if variant == "grouped":
            assignment_for_sum, probabilities_for_sum = (
                sorted_assignments,
                sorted_probabilities,
            )
            direct_order = False
        else:
            assignment_for_sum, probabilities_for_sum = expert_ids, expert_probs
            direct_order = True
        moe_output = self.tensor("moe.weighted_output", (b, HIDDEN))
        self.invoke(
            "weighted_moe_sum",
            variant,
            KernelFamily.WEIGHTED_MOE_SUM,
            WeightedSumWork(
                "expert_output",
                "expert_probabilities",
                "assignments",
                "output",
                b,
                direct_order=direct_order,
                grouped_assignments=(
                    tuple(a for group in groups for a in group.assignments)
                    if not direct_order
                    else ()
                ),
            ),
            (
                self.read("expert_output", expert_output),
                self.read("expert_probabilities", probabilities_for_sum),
                self.read("assignments", assignment_for_sum),
                self.write("output", moe_output),
            ),
        )

        layer_output = self.tensor("output", (b, HIDDEN))
        self.invoke(
            "final_residual_add",
            "residual_add",
            KernelFamily.VECTOR,
            VectorWork(
                ("residual", "moe"),
                ("output",),
                b,
                HIDDEN,
                1,
                HIDDEN,
                (VectorStage(FlowOperator.ADD, HIDDEN, input_indices=(0, 1)),),
                output_width=HIDDEN,
            ),
            (
                self.read("residual", x),
                self.read("moe", moe_output),
                self.write("output", layer_output),
            ),
        )

        padded_rows = sum(
            ((len(group.assignments) + 15) // 16) * 16 - len(group.assignments)
            for group in groups
        )
        physical_rows = sum(
            ((len(group.assignments) + 15) // 16) * 16 for group in groups
        )
        expert_alignment = {
            str(group.expert): {
                "assignments": len(group.assignments),
                "physical_rows": ((len(group.assignments) + 15) // 16) * 16,
                "padded_rows": ((len(group.assignments) + 15) // 16) * 16
                - len(group.assignments),
                "row_utilization": len(group.assignments)
                / (((len(group.assignments) + 15) // 16) * 16),
            }
            for group in groups
        }
        self.metadata = {
            "layer": l,
            "fa3_variant": attention_variant,
            "fa3_split_count": split,
            "fa3_query_ctas": b * Q_HEADS * split,
            "moe_variant": variant,
            "active_experts": len(groups),
            "expert_assignment_count": len(assignments),
            "expert_token_counts": {str(k): v for k, v in expert_counts.items()},
            "grouped_physical_rows": physical_rows if grouped else 0,
            "grouped_padded_rows": padded_rows if grouped else 0,
            "grouped_row_utilization": (len(assignments) / physical_rows)
            if grouped
            else None,
            "grouped_alignment_amplification": (physical_rows / len(assignments))
            if grouped
            else None,
            "expert_alignment": expert_alignment,
            "direct_physical_rows": physical_rows if not grouped else 0,
            "direct_padded_rows": padded_rows if not grouped else 0,
            "direct_row_utilization": (len(assignments) / physical_rows)
            if not grouped
            else None,
            "direct_alignment_amplification": (physical_rows / len(assignments))
            if not grouped
            else None,
            "new_tokens": b,
            "layer_output_shape": [b, HIDDEN],
        }
        return layer_output


def _normalize_routes(routes, layers, batch):
    parsed = []
    if len(routes) != layers:
        raise ValueError("routing trace must contain one route list per layer")
    for layer_index, layer_routes in enumerate(routes):
        if len(layer_routes) != batch:
            raise ValueError(f"routing trace layer {layer_index} must contain B routes")
        tokens = []
        for token_index, route in enumerate(layer_routes):
            ids = tuple(int(value) for value in route["expert_ids"])
            probabilities = tuple(float(value) for value in route["probabilities"])
            total = sum(probabilities)
            if len(ids) != TOP_K or len(probabilities) != TOP_K or total <= 0:
                raise ValueError(
                    "each routing record must contain 8 IDs and probabilities"
                )
            tokens.append(
                Qwen3RoutingToken(
                    token_index, ids, tuple(p / total for p in probabilities)
                )
            )
        parsed.append(tuple(tokens))
    return tuple(parsed)


def synthetic_routing_trace(batch, layers, seed=0):
    rng = random.Random(seed)
    routes = []
    for _ in range(layers):
        layer = []
        for token in range(batch):
            experts = tuple(rng.sample(range(EXPERTS), TOP_K))
            raw = [rng.random() + 1e-6 for _ in range(TOP_K)]
            total = sum(raw)
            layer.append(
                Qwen3RoutingToken(token, experts, tuple(value / total for value in raw))
            )
        routes.append(tuple(layer))
    return tuple(routes)


def load_routing_trace(path, layers, batch):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("layers")
    if not isinstance(data, list):
        raise ValueError("routing trace must be a layer-indexed JSON array")
    return _normalize_routes(data, layers, batch)


def build(
    batch_size=1,
    layers=1,
    context_lengths=None,
    split_count=None,
    routing_seed=0,
    routing_trace=None,
):
    if batch_size not in (1, 8, 16) or layers not in (1, 2):
        raise ValueError("Qwen3 decode supports batch size 1/8/16 and 1/2 layers")
    if context_lengths is None:
        context_lengths = tuple(
            (10000, 16384, 24576, 32000)[i % 4] for i in range(batch_size)
        )
    context_lengths = tuple(int(length) for length in context_lengths)
    if len(context_lengths) != batch_size or any(
        not 10000 <= n <= 32000 for n in context_lengths
    ):
        raise ValueError(
            "provide exactly B context lengths in the inclusive 10000-32000 range"
        )
    if split_count is None:
        split_count = 4 if batch_size == 1 else 1
    if not 1 <= split_count <= 32:
        raise ValueError("split count must be between 1 and 32")
    trace = routing_trace or synthetic_routing_trace(batch_size, layers, routing_seed)
    if not all(
        isinstance(token, Qwen3RoutingToken) for layer in trace for token in layer
    ):
        trace = _normalize_routes(trace, layers, batch_size)
    root = Tensor("input.hidden", (batch_size, HIDDEN), read_only=True)
    tensors = {root.id: root}
    layer_metadata = []
    x = root.id
    launches = []
    commands = []
    for layer in range(layers):
        builder = LayerBuilder(
            batch_size, context_lengths, split_count, layer, trace[layer], tensors
        )
        x = builder.build(x)
        for identity, tensor in builder.tensors.items():
            if identity in tensors and tensors[identity] != tensor:
                raise ValueError(f"duplicate incompatible global tensor: {identity}")
            tensors[identity] = tensor
        launches.extend(builder.launches)
        commands.extend(builder.commands)
        layer_metadata.append(builder.metadata)
    program = DecodeProgram(
        symbol=f"qwen3_decode_b{batch_size}_l{layers}",
        batch_size=batch_size,
        layers=layers,
        context_lengths=context_lengths,
        routing_seed=routing_seed,
        routing_trace=trace,
        tensors=tuple(tensors.values()),
        launches=tuple(launches),
        commands=tuple(commands),
    )
    program.verify()
    stats = {
        "model": "Qwen3-30B-A3B",
        "mode": "single_decode_step",
        "batch_size": batch_size,
        "layers": layers,
        "context_lengths": list(context_lengths),
        "effective_context_lengths": [length + 1 for length in context_lengths],
        "routing_source": "provided_trace"
        if routing_trace is not None
        else "synthetic_seeded",
        "routing_seed": routing_seed,
        "layer_summaries": layer_metadata,
        "tensor_count": len(tensors),
        "kernel_count": len(launches),
    }
    return program, stats


def parse_trace(path, layers, batch):
    return load_routing_trace(path, layers, batch)
