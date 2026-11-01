from collections import Counter
import math
import unittest
from gtsim_frontend.ir.algorithm import Algorithm
from gtsim_frontend.ir.primitives import (
    RouterSpec,
    HistogramSpec,
    GatherSpec,
    WeightedSpec,
    RouteAssignment,
    ExtractSpec,
    KVUpdateSpec,
)
from gtsim_frontend.passes.instantiate import instantiate
from gtsim_frontend.passes.hopper.lower import lower
from gtsim_frontend.passes.lower_execution import lower as expand
from gtsim_frontend.serialization.json_codec import dumps, loads
from native_math_oracle import evaluate_vector


def native(spec):
    a = Algorithm(
        symbol="auxiliary",
        family="vector",
        dtype="bf16",
        schedule_id="simt_vector",
        primitive=spec,
    )
    return loads(dumps(lower(loads(dumps(instantiate(a))))))


class MoeMemoryTests(unittest.TestCase):
    def test_softmax_top8_indices_and_selected_probability_renormalization(self):
        logits = [[math.sin((b + e) / 7) for e in range(128)] for b in range(2)]
        logits[0][3] = logits[0][9] = 3.0
        compiled = native(RouterSpec(2, 128, 8))
        actual = evaluate_vector(compiled, {"Logits": logits})
        for token, row in enumerate(logits):
            exps = [math.exp(x - max(row)) for x in row]
            probabilities = [x / sum(exps) for x in exps]
            indices = sorted(range(128), key=lambda e: (-probabilities[e], e))[:8]
            self.assertEqual(actual["ExpertIDs"][token], indices)
            denominator = sum(probabilities[e] for e in indices)
            for expected, got in zip(
                (probabilities[e] / denominator for e in indices),
                actual["Probabilities"][token],
            ):
                self.assertAlmostEqual(expected, got, places=13)
        graph = expand(compiled)
        self.assertEqual(sum(op.op == "reduce_max" for op in graph.ops), 2 * 9)
        self.assertTrue(
            all(
                op.dtype == "int32"
                for op in graph.ops
                if op.op == "cp.sram2dram_tma"
                and graph.tensors[op.tensor].name == "ExpertIDs"
            )
        )

    def test_histogram_prefix_gather_and_weighted_reduction(self):
        ids = [[3, 1], [1, 6], [3, 6]]
        probabilities = [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]]
        assignments = tuple(
            RouteAssignment(t, s, expert)
            for t, row in enumerate(ids)
            for s, expert in enumerate(row)
        )
        ordered = sorted(assignments, key=lambda a: (a.expert, a.token, a.slot))
        expected_counts = Counter(expert for row in ids for expert in row)
        hist = native(HistogramSpec(3, 8, 2))
        histogram = evaluate_vector(hist, {"ExpertIDs": ids})
        self.assertEqual(histogram["Counts"], [[expected_counts[e] for e in range(8)]])
        self.assertEqual(
            histogram["Offsets"],
            [[sum(expected_counts[i] for i in range(e)) for e in range(9)]],
        )
        x = [[100 * t + d for d in range(16)] for t in range(3)]
        gather = native(GatherSpec(3, 16, 8, 2, assignments))
        gathered = evaluate_vector(
            gather,
            {
                "X": x,
                "ExpertIDs": ids,
                "Probabilities": probabilities,
                "Offsets": histogram["Offsets"],
            },
        )
        self.assertEqual(
            gathered["Assignments"], [[a.token, a.slot, a.expert] for a in ordered]
        )
        self.assertEqual(gathered["SortedX"], [x[a.token] for a in ordered])
        self.assertEqual(
            gathered["SortedProbabilities"],
            [[probabilities[a.token][a.slot]] for a in ordered],
        )
        rows_by_token = tuple(
            tuple(
                next(i for i, a in enumerate(ordered) if a.token == t and a.slot == s)
                for s in range(2)
            )
            for t in range(3)
        )
        expert_outputs = [
            [a.expert * 100 + a.token * 10 + d for d in range(16)] for a in ordered
        ]
        weighted = native(WeightedSpec(3, 16, 2, rows_by_token, True))
        actual = evaluate_vector(
            weighted,
            {
                "ExpertOutput": expert_outputs,
                "Probabilities": probabilities,
                "Assignments": gathered["Assignments"],
            },
        )["Output"]
        for token in range(3):
            expected = [
                sum(
                    probabilities[token][slot]
                    * (ids[token][slot] * 100 + token * 10 + d)
                    for slot in range(2)
                )
                for d in range(16)
            ]
            for x, y in zip(expected, actual[token]):
                self.assertAlmostEqual(x, y, places=11)
        self.assertTrue(any(op.op == "shfl" for op in expand(hist).ops))

    def test_direct_reduction_preserves_slot_rows_without_grouping_metadata(self):
        compiled = native(WeightedSpec(1, 16, 8, (tuple(range(8)),), False))
        output = [[10 * slot + d for d in range(16)] for slot in range(8)]
        probs = [[0.125] * 8]
        actual = evaluate_vector(
            compiled, {"ExpertOutput": output, "Probabilities": probs}
        )["Output"][0]
        self.assertEqual(actual, [35 + d for d in range(16)])
        self.assertNotIn("Assignments", {t.id for t in compiled.tensors})

    def test_v_extract_and_kv_append_preserve_history_and_request_offsets(self):
        spec = ExtractSpec(2, 4, 128, 5120, 4608)
        projection = [[b * 10000 + d for d in range(5120)] for b in range(2)]
        extracted = evaluate_vector(native(spec), {"Input": projection})["Output"]
        self.assertEqual(
            extracted,
            [
                projection[b][4608 + h * 128 : 4608 + (h + 1) * 128]
                for b in range(2)
                for h in range(4)
            ],
        )
        update = native(KVUpdateSpec(2, 4, 128, (3, 5)))
        keys = [[v + 1 for v in row] for row in extracted]
        supplied = {"Key": keys, "Value": extracted}
        for b, length in enumerate((3, 5)):
            for h in range(4):
                for name in ("K", "V"):
                    supplied[f"{name}_r{b}_h{h}"] = [
                        [b * 1000 + h * 100 + r] * 128 for r in range(length + 1)
                    ]
        actual = evaluate_vector(update, supplied)
        for b, length in enumerate((3, 5)):
            for h in range(4):
                for name, source in (("K", keys), ("V", extracted)):
                    tensor = f"{name}_r{b}_h{h}"
                    self.assertEqual(actual[tensor][:length], supplied[tensor][:length])
                    self.assertEqual(actual[tensor][length], source[b * 4 + h])
        graph = expand(update)
        self.assertEqual(len(graph.partial_outputs), 16)
        self.assertEqual(
            sum(
                op.tile[2] * op.tile[3]
                for op in graph.ops
                if op.op == "cp.sram2dram_tma"
            ),
            2 * 4 * 2 * 128,
        )


if __name__ == "__main__":
    unittest.main()
