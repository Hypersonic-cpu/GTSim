"""Qwen3-30B-A3B layer composition using only canonical kernel Algorithms."""

from collections import Counter
from dataclasses import replace
from ..ir.algorithm import Algorithm
from ..ir.common import Tensor, MatrixGroup
from ..ir.program import Program, Invocation, Command
from ..ir.primitives import (
    NormSpec,
    AddSpec,
    SwiGLUSpec,
    RopeSpec,
    ExtractSpec,
    KVUpdateSpec,
    RouterSpec,
    HistogramSpec,
    GatherSpec,
    WeightedSpec,
    RouteAssignment,
)
from ..kernels.signatures import signature
from .routing import normalize, synthetic_routing_trace, load_routing_trace

HIDDEN = 2048
Q_HEADS = 32
KV_HEADS = 4
HEAD_DIM = 128
EXPERTS = 128
TOP_K = 8
INTERMEDIATE = 768


class LayerBuilder:
    """Declare tensors, choose Algorithms, and bind ordered invocations."""

    def __init__(self, layer, batch, contexts, splits, routes, tensors):
        self.layer = layer
        self.batch = batch
        self.contexts = contexts
        self.splits = splits
        self.routes = routes
        self.tensors = tensors
        self.launches = []
        self.metadata = {}

    def tensor(self, name):
        return f"l{self.layer}.{name}"

    def invoke(self, name, algorithm, bindings, variant=None):
        identity = f"l{self.layer}_{name}"
        algorithm = replace(algorithm, symbol=identity)
        specs = signature(algorithm)
        for spec in specs:
            target = bindings[spec.id]
            if target not in self.tensors:
                self.tensors[target] = replace(spec, id=target)
            else:
                current = self.tensors[target]
                if (current.shape, current.dtype, current.tier) != (
                    spec.shape,
                    spec.dtype,
                    spec.tier,
                ):
                    raise ValueError(f"incompatible workload tensor binding {target}")
                if not spec.read_only and current.read_only:
                    raise ValueError(f"write to workload constant {target}")
        self.launches.append(Invocation(identity, algorithm, tuple(bindings.items())))
        self.metadata[identity] = {
            "layer": self.layer,
            "name": name,
            "variant": variant or algorithm.schedule_id,
        }

    def vector(self, name, spec, bindings, variant=None):
        self.invoke(
            name,
            Algorithm(
                symbol=name,
                family="vector",
                dtype="bf16",
                schedule_id="simt_vector",
                primitive=spec,
            ),
            bindings,
            variant,
        )

    def projection(self, name, input, weight, output, n, k):
        self.invoke(
            name,
            Algorithm(
                symbol=name,
                dimensions=(self.batch, n, k),
                dtype="bf16",
                schedule_id="small_m_mma",
            ),
            {"A": input, "B": weight, "C": output},
        )

    def build(self, x):
        b = self.batch
        t = self.tensor
        normalized = t("input_norm")
        self.vector(
            "input_rmsnorm",
            NormSpec(b, HIDDEN),
            {"X": x, "Weight": t("weight.input_norm"), "Output": normalized},
        )
        qkv = t("qkv_projection")
        self.projection(
            "qkv_projection", normalized, t("weight.qkv"), qkv, 5120, HIDDEN
        )
        q = t("query")
        k = t("key_new")
        v = t("value_new")
        self.vector(
            "q_norm_rope",
            RopeSpec(b, Q_HEADS, HEAD_DIM, 5120, 0, self.contexts),
            {"Input": qkv, "Weight": t("weight.q_norm"), "Output": q},
        )
        self.vector(
            "k_norm_rope",
            RopeSpec(b, KV_HEADS, HEAD_DIM, 5120, 4096, self.contexts),
            {"Input": qkv, "Weight": t("weight.k_norm"), "Output": k},
        )
        self.vector(
            "v_extract",
            ExtractSpec(b, KV_HEADS, HEAD_DIM, 5120, 4608),
            {"Input": qkv, "Output": v},
        )
        caches = {
            f"{letter}_r{request}_h{head}": t(f"kv.{letter}.r{request}.h{head}")
            for request in range(b)
            for head in range(KV_HEADS)
            for letter in ("K", "V")
        }
        self.vector(
            "kv_cache_update",
            KVUpdateSpec(b, KV_HEADS, HEAD_DIM, self.contexts),
            {"Key": k, "Value": v, **caches},
        )
        attention = t("attention_output")
        algorithm = Algorithm(
            symbol="attention",
            family="fa3",
            dimensions=(b, Q_HEADS, KV_HEADS, 1, max(self.contexts) + 1),
            dtype="bf16",
            schedule_id="decode_simt",
            valid_lengths=tuple(n + 1 for n in self.contexts),
            split_count=self.splits,
        )
        binding = {"Q": q, **caches}
        if self.splits > 1:
            binding.update(Partial=t("attention_partial"), LSE=t("attention_lse"))
        else:
            binding["O"] = attention
        self.invoke(
            "fa3_decode",
            algorithm,
            binding,
            "split_kv" if self.splits > 1 else "non_split",
        )
        if self.splits > 1:
            self.invoke(
                "fa3_combine",
                replace(algorithm, schedule_id="decode_combine", contract=None),
                {
                    "Partial": t("attention_partial"),
                    "LSE": t("attention_lse"),
                    "O": attention,
                },
                "stable_lse_combine",
            )
        projected = t("attention_projection")
        self.projection(
            "attention_output_projection",
            attention,
            t("weight.attention_output"),
            projected,
            HIDDEN,
            Q_HEADS * HEAD_DIM,
        )
        residual = t("attention_residual")
        self.vector(
            "attention_residual_add",
            AddSpec(b, HIDDEN),
            {"X": x, "Y": projected, "Output": residual},
        )
        post = t("post_attention_norm")
        self.vector(
            "post_attention_rmsnorm",
            NormSpec(b, HIDDEN),
            {"X": residual, "Weight": t("weight.post_attention_norm"), "Output": post},
        )
        logits = t("router_logits")
        ids = t("route.ids")
        prob = t("route.probabilities")
        self.projection(
            "router_projection", post, t("weight.router"), logits, EXPERTS, HIDDEN
        )
        self.vector(
            "router_softmax_topk",
            RouterSpec(b, EXPERTS, TOP_K),
            {"Logits": logits, "ExpertIDs": ids, "Probabilities": prob},
        )
        assignments = tuple(
            RouteAssignment(token, slot, expert)
            for token, route in enumerate(self.routes)
            for slot, expert in enumerate(route.expert_ids)
        )
        grouped = b > 1
        ordered = (
            tuple(sorted(assignments, key=lambda a: (a.expert, a.token, a.slot)))
            if grouped
            else assignments
        )
        counts = Counter(a.expert for a in ordered)
        input = post
        if grouped:
            self.vector(
                "expert_histogram_prefix",
                HistogramSpec(b, EXPERTS, TOP_K),
                {
                    "ExpertIDs": ids,
                    "Counts": t("moe.counts"),
                    "Offsets": t("moe.offsets"),
                },
            )
            self.vector(
                "expert_assignment_gather",
                GatherSpec(b, HIDDEN, EXPERTS, TOP_K, assignments),
                {
                    "X": post,
                    "ExpertIDs": ids,
                    "Probabilities": prob,
                    "Offsets": t("moe.offsets"),
                    "SortedX": t("moe.sorted_input"),
                    "Assignments": t("moe.assignments"),
                    "SortedProbabilities": t("moe.sorted_probabilities"),
                },
            )
            input = t("moe.sorted_input")
        groups = []
        for expert in sorted(counts):
            rows = tuple(i for i, a in enumerate(ordered) if a.expert == expert)
            inputs = rows if grouped else tuple(ordered[i].token for i in rows)
            groups.append(MatrixGroup(expert, inputs, rows, f"W_e{expert}"))
        groups = tuple(groups)
        variant = "grouped_small_m_mma_baseline" if grouped else "direct"
        gate_up = t("moe.gate_up")
        self.invoke(
            "expert_gemm1",
            Algorithm(
                symbol="experts",
                dimensions=(b * TOP_K, 2 * INTERMEDIATE, HIDDEN),
                dtype="bf16",
                schedule_id="grouped_small_m_mma",
                groups=groups,
            ),
            {
                "A": input,
                "C": gate_up,
                **{g.weight: t(f"weight.expert{g.identity}.gate_up") for g in groups},
            },
            variant,
        )
        activated = t("moe.activated")
        self.vector(
            "swiglu",
            SwiGLUSpec(b * TOP_K, INTERMEDIATE),
            {"X": gate_up, "Output": activated},
        )
        down = t("moe.expert_output")
        down_groups = tuple(replace(g, input_rows=g.output_rows) for g in groups)
        self.invoke(
            "expert_gemm2",
            Algorithm(
                symbol="experts",
                dimensions=(b * TOP_K, HIDDEN, INTERMEDIATE),
                dtype="bf16",
                schedule_id="grouped_small_m_mma",
                groups=down_groups,
            ),
            {
                "A": activated,
                "C": down,
                **{g.weight: t(f"weight.expert{g.identity}.down") for g in groups},
            },
            variant,
        )
        rows_by_token = tuple(
            tuple(
                next(
                    i
                    for i, a in enumerate(ordered)
                    if a.token == token and a.slot == slot
                )
                for slot in range(TOP_K)
            )
            for token in range(b)
        )
        weighted = t("moe.weighted_output")
        binding = {"ExpertOutput": down, "Probabilities": prob, "Output": weighted}
        if grouped:
            binding["Assignments"] = t("moe.assignments")
        self.vector(
            "weighted_moe_sum",
            WeightedSpec(b, HIDDEN, TOP_K, rows_by_token, grouped),
            binding,
            variant,
        )
        output = t("output")
        self.vector(
            "final_residual_add",
            AddSpec(b, HIDDEN),
            {"X": residual, "Y": weighted, "Output": output},
        )
        physical = sum(((count + 15) // 16) * 16 for count in counts.values())
        self.summary = {
            "layer": self.layer,
            "new_tokens": b,
            "layer_output_shape": [b, HIDDEN],
            "fa3_variant": "split_kv" if self.splits > 1 else "non_split",
            "fa3_split_count": self.splits,
            "fa3_query_ctas": b * Q_HEADS * self.splits,
            "moe_variant": variant,
            "active_experts": len(counts),
            "expert_assignment_count": b * TOP_K,
            "expert_token_counts": {str(e): counts[e] for e in sorted(counts)},
            "physical_rows": physical,
            "padded_rows": physical - b * TOP_K,
            "row_utilization": b * TOP_K / physical,
            "padding_amplification": physical / (b * TOP_K),
            "assignments": [
                {"token": a.token, "slot": a.slot, "expert": a.expert} for a in ordered
            ],
        }
        return output


def build(
    batch_size=1,
    layers=1,
    context_lengths=None,
    split_count=None,
    routing_seed=0,
    routing_trace=None,
):
    if batch_size not in (1, 8, 16) or layers not in (1, 2):
        raise ValueError("supported batch sizes1/8/16 and layer counts1/2")
    contexts = (
        tuple(context_lengths)
        if context_lengths is not None
        else tuple((10000, 16384, 24576, 32000)[i % 4] for i in range(batch_size))
    )
    if len(contexts) != batch_size or any(
        type(n) is not int or not 10000 <= n <= 32000 for n in contexts
    ):
        raise ValueError("provide exactly B context lengths in10000..32000")
    splits = split_count if split_count is not None else 4 if batch_size == 1 else 1
    if type(splits) is not int or not 1 <= splits <= 32:
        raise ValueError("split count must be1..32")
    trace = (
        normalize(routing_trace, layers, batch_size)
        if routing_trace is not None
        else synthetic_routing_trace(batch_size, layers, routing_seed)
    )
    tensors = {
        "input.hidden": Tensor(
            "input.hidden", (batch_size, HIDDEN), "bf16", read_only=True
        )
    }
    x = "input.hidden"
    launches = []
    metadata = {}
    summaries = []
    for layer in range(layers):
        builder = LayerBuilder(
            layer, batch_size, contexts, splits, trace[layer], tensors
        )
        x = builder.build(x)
        launches.extend(builder.launches)
        metadata.update(builder.metadata)
        summaries.append(builder.summary)
    program = Program(
        symbol=f"qwen3_decode_b{batch_size}_l{layers}",
        tensors=tuple(tensors.values()),
        launches=tuple(launches),
        commands=tuple(Command("LaunchKernel", inv.id) for inv in launches)
        + (Command("StreamSynchronize"),),
    )
    program.verify()
    summary = {
        "model": "Qwen3-30B-A3B",
        "mode": "single_decode_step",
        "batch_size": batch_size,
        "layers": layers,
        "context_lengths": list(contexts),
        "effective_context_lengths": [n + 1 for n in contexts],
        "routing_source": "provided_trace"
        if routing_trace is not None
        else "synthetic_seeded",
        "routing_seed": routing_seed,
        "routing_trace": [
            [
                {
                    "expert_ids": list(r.expert_ids),
                    "probabilities": list(r.probabilities),
                }
                for r in layer
            ]
            for layer in trace
        ],
        "layer_summaries": summaries,
        "kernel_metadata": metadata,
        "kernel_count": len(launches),
        "tensor_count": len(tensors),
    }
    return program, summary


parse_trace = load_routing_trace
