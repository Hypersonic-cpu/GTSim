"""Workload producer/consumer identities independently encode pre-norm Qwen3."""

from dataclasses import replace
from collections import Counter
import unittest
from gtsim_frontend.ir.algorithm import Algorithm
from gtsim_frontend.ir.program import Program
from gtsim_frontend.programs.qwen3_decode import build
from gtsim_frontend.programs.routing import RoutingToken
from gtsim_frontend.serialization.json_codec import loads, dumps


def layer_bindings(program, layer):
    return {
        inv.id.split("_", 1)[1]: dict(inv.bindings)
        for inv in program.launches
        if inv.id.startswith(f"l{layer}_")
    }


def assert_prenorm_graph(test, program, layers):
    prior = "input.hidden"
    for layer in range(layers):
        b = layer_bindings(program, layer)
        test.assertEqual(b["input_rmsnorm"]["X"], prior)
        test.assertEqual(b["qkv_projection"]["A"], b["input_rmsnorm"]["Output"])
        test.assertEqual(b["attention_residual_add"]["X"], prior)
        test.assertEqual(
            b["attention_residual_add"]["Y"], b["attention_output_projection"]["C"]
        )
        residual = b["attention_residual_add"]["Output"]
        test.assertEqual(b["post_attention_rmsnorm"]["X"], residual)
        test.assertEqual(
            b["router_projection"]["A"], b["post_attention_rmsnorm"]["Output"]
        )
        test.assertEqual(b["final_residual_add"]["X"], residual)
        test.assertEqual(b["final_residual_add"]["Y"], b["weighted_moe_sum"]["Output"])
        prior = b["final_residual_add"]["Output"]


class Qwen3DecodeContracts(unittest.TestCase):
    def test_all_six_are_canonical_and_follow_decoder_dataflow(self):
        for batch in (1, 8, 16):
            contexts = tuple(10000 + i * 1300 for i in range(batch))
            for layers in (1, 2):
                program, summary = build(batch, layers, contexts)
                self.assertIs(type(program), Program)
                restored = loads(dumps(program))
                self.assertTrue(
                    all(type(k.algorithm) is Algorithm for k in restored.launches)
                )
                self.assertEqual(
                    len(program.launches), (18 if batch == 1 else 19) * layers
                )
                self.assertEqual(summary["context_lengths"], list(contexts))
                assert_prenorm_graph(self, program, layers)
                for layer in summary["layer_summaries"]:
                    self.assertEqual(layer["new_tokens"], batch)
                    self.assertEqual(layer["layer_output_shape"], [batch, 2048])
                    self.assertEqual(
                        layer["fa3_variant"], "split_kv" if batch == 1 else "non_split"
                    )
                    self.assertEqual(
                        layer["moe_variant"],
                        "direct" if batch == 1 else "grouped_small_m_mma_baseline",
                    )

    def test_original_input_cannot_substitute_for_final_residual(self):
        program, _ = build(1, 1, (10000,))
        final = next(k for k in program.launches if k.id == "l0_final_residual_add")
        bad = replace(
            final,
            bindings=tuple(
                (key, "input.hidden" if key == "X" else value)
                for key, value in final.bindings
            ),
        )
        changed = replace(
            program,
            launches=tuple(bad if k.id == final.id else k for k in program.launches),
        )
        # Shapes alone accept this mutation; the semantic producer test rejects it.
        changed.verify()
        with self.assertRaises(AssertionError):
            assert_prenorm_graph(self, changed, 1)

    def test_per_request_per_layer_kv_lengths_and_qkv_regions(self):
        lengths = (10000, 11001, 17003, 22007, 27011, 30013, 31517, 32000)
        program, _ = build(8, 2, lengths)
        tensors = {t.id: t for t in program.tensors}
        for layer in range(2):
            inv = {
                k.id.split("_", 1)[1]: k
                for k in program.launches
                if k.id.startswith(f"l{layer}_")
            }
            self.assertEqual(
                inv["fa3_decode"].algorithm.valid_lengths, tuple(n + 1 for n in lengths)
            )
            self.assertEqual(
                inv["kv_cache_update"].algorithm.primitive.context_lengths, lengths
            )
            q = inv["q_norm_rope"].algorithm.primitive
            k = inv["k_norm_rope"].algorithm.primitive
            v = inv["v_extract"].algorithm.primitive
            self.assertEqual((q.heads, k.heads, v.heads), (32, 4, 4))
            self.assertEqual(
                (q.input_offset, k.input_offset, v.input_offset), (0, 4096, 4608)
            )
            self.assertEqual(q.positions, lengths)
            self.assertEqual(k.positions, lengths)
            for request, n in enumerate(lengths):
                for head in range(4):
                    for letter in ("K", "V"):
                        self.assertEqual(
                            tensors[f"l{layer}.kv.{letter}.r{request}.h{head}"].shape,
                            (n + 1, 128),
                        )
        self.assertEqual(tensors["input.hidden"].dtype, "bf16")

    def test_expert_group_rows_weight_identities_and_ordered_routing(self):
        for batch in (1, 8, 16):
            program, summary = build(batch, 2)
            for layer in range(2):
                inv = {
                    k.id.split("_", 1)[1]: k
                    for k in program.launches
                    if k.id.startswith(f"l{layer}_")
                }
                order = list(inv)
                self.assertLess(
                    order.index("router_projection"), order.index("router_softmax_topk")
                )
                self.assertLess(
                    order.index("router_softmax_topk"), order.index("expert_gemm1")
                )
                records = summary["layer_summaries"][layer]["assignments"]
                one = inv["expert_gemm1"]
                two = inv["expert_gemm2"]
                for group, down in zip(one.algorithm.groups, two.algorithm.groups):
                    self.assertEqual(down.input_rows, group.output_rows)
                    self.assertEqual(down.output_rows, group.output_rows)
                    self.assertTrue(
                        all(
                            records[r]["expert"] == group.identity
                            for r in group.output_rows
                        )
                    )
                    self.assertEqual(
                        dict(one.bindings)[group.weight],
                        f"l{layer}.weight.expert{group.identity}.gate_up",
                    )
                    self.assertEqual(
                        dict(two.bindings)[down.weight],
                        f"l{layer}.weight.expert{group.identity}.down",
                    )
                    if batch == 1:
                        self.assertEqual(group.input_rows, (0,))
                    else:
                        self.assertEqual(group.input_rows, group.output_rows)
                weighted = inv["weighted_moe_sum"].algorithm.primitive
                for token, rows in enumerate(weighted.rows_by_token):
                    for slot, row in enumerate(rows):
                        self.assertEqual(
                            (records[row]["token"], records[row]["slot"]), (token, slot)
                        )
                self.assertEqual("expert_histogram_prefix" in inv, batch > 1)
                self.assertEqual("expert_assignment_gather" in inv, batch > 1)
                self.assertEqual(
                    len(one.algorithm.groups),
                    summary["layer_summaries"][layer]["active_experts"],
                )
            self.assertNotEqual(
                summary["routing_trace"][0], summary["routing_trace"][1]
            )

    def test_zero_one_multiple_expert_occupancies_and_split_override(self):
        routes = []
        for token in range(8):
            base = 0 if token < 2 else 8 + (token - 2) * 8
            routes.append(
                RoutingToken(token, tuple(range(base, base + 8)), (0.125,) * 8)
            )
        program, summary = build(8, 1, routing_trace=(tuple(routes),), split_count=4)
        counts = Counter(
            int(e)
            for e, n in summary["layer_summaries"][0]["expert_token_counts"].items()
            for _ in range(n)
        )
        self.assertEqual(
            counts, {**{e: 2 for e in range(8)}, **{e: 1 for e in range(8, 56)}}
        )
        self.assertNotIn(100, counts)
        self.assertTrue(any(k.id == "l0_fa3_combine" for k in program.launches))
        self.assertEqual(summary["layer_summaries"][0]["expert_assignment_count"], 64)
        self.assertEqual(summary["layer_summaries"][0]["padded_rows"], 48 * 15 + 8 * 14)


if __name__ == "__main__":
    unittest.main()
