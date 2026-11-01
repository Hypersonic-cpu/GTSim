import math
from dataclasses import replace
import unittest
from gtsim_frontend.ir.algorithm import Algorithm
from gtsim_frontend.ir.flow import FlowKernel
from gtsim_frontend.ir.arch import ArchKernel
from gtsim_frontend.passes.instantiate import instantiate
from gtsim_frontend.passes.lower_hopper_fa3 import lower
from gtsim_frontend.passes.lower_execution import lower as expand
from gtsim_frontend.serialization.json_codec import dumps, loads
from native_math_oracle import evaluate_partition


def attention(lengths, splits=1):
    return Algorithm(
        symbol="decode",
        family="fa3",
        dimensions=(len(lengths), 32, 4, 1, max(lengths)),
        dtype="bf16",
        schedule_id="decode_simt",
        valid_lengths=tuple(lengths),
        split_count=splits,
    )


class CanonicalAttentionTests(unittest.TestCase):
    def test_native_math_matches_independent_full_attention_and_lse_merge(self):
        length = 519
        q = [math.sin(d / 13) for d in range(128)]
        keys = [[math.cos((t + d) / 31) for d in range(128)] for t in range(length)]
        values = [[math.sin((t - d) / 17) for d in range(128)] for t in range(length)]
        scores = [sum(x * y for x, y in zip(q, k)) / math.sqrt(128) for k in keys]
        maximum = max(scores)
        weights = [math.exp(s - maximum) for s in scores]
        expected = [
            sum(w * v[d] for w, v in zip(weights, values)) / sum(weights)
            for d in range(128)
        ]
        for splits in (1, 4):
            flow = loads(dumps(instantiate(attention((length,), splits))))
            self.assertIs(type(flow), FlowKernel)
            native = loads(dumps(lower(flow)))
            self.assertIs(type(native), ArchKernel)
            parts = [p for p in native.domain.partitions if p.query_head == 0]
            partials = [evaluate_partition(native, p, q, keys, values) for p in parts]
            m = max(lse for _, lse in partials)
            ws = [math.exp(lse - m) for _, lse in partials]
            actual = [
                sum(w * o[d] for w, (o, _) in zip(ws, partials)) / sum(ws)
                for d in range(128)
            ]
            if splits > 1:
                algo = replace(
                    attention((length,), splits),
                    schedule_id="decode_combine",
                    contract=None,
                )
                combine = loads(dumps(lower(instantiate(algo))))
                combined, _ = evaluate_partition(
                    combine,
                    combine.domain.partitions[0],
                    None,
                    None,
                    None,
                    supplied={
                        "Partial": [o for o, _ in partials],
                        "LSE": [[lse] for _, lse in partials],
                    },
                )
                for x, y in zip(expected, combined):
                    self.assertAlmostEqual(x, y, places=11)
            for x, y in zip(expected, actual):
                self.assertAlmostEqual(x, y, places=11)

    def test_serialized_flow_and_native_dependencies_are_authoritative(self):
        from gtsim_frontend.ir.schedule import Dependency, OpRole as R, Relation, Phase

        flow = instantiate(attention((513,)))
        edge = Dependency(R.LOAD_V, Phase.COMPLETE, Relation.EXPLICIT)
        changed = replace(
            flow,
            ops=tuple(
                replace(op, dependencies=op.dependencies + (edge,))
                if op.role == R.QK
                else op
                for op in flow.ops
            ),
        )
        before = expand(lower(flow))
        after = expand(lower(loads(dumps(changed))))
        self.assertNotEqual(before.ops, after.ops)
        native = lower(flow)
        changed_native = replace(
            native,
            ops=tuple(
                replace(
                    pattern,
                    steps=(
                        replace(
                            pattern.steps[0],
                            dependencies=pattern.steps[0].dependencies + (edge,),
                        ),
                        *pattern.steps[1:],
                    ),
                )
                if pattern.source.role == R.QK
                else pattern
                for pattern in native.ops
            ),
        )
        self.assertNotEqual(before.ops, expand(loads(dumps(changed_native))).ops)
        renamed = replace(
            flow,
            ops=tuple(
                replace(op, id=f"unrelated_{i}") for i, op in enumerate(flow.ops)
            ),
        )
        self.assertEqual(before.ops, expand(lower(loads(dumps(renamed)))).ops)
        bad = replace(
            native,
            ops=tuple(
                replace(
                    pattern,
                    steps=(
                        replace(pattern.steps[0], opcode="unsupported"),
                        *pattern.steps[1:],
                    ),
                )
                if pattern.source.role == R.QK
                else pattern
                for pattern in native.ops
            ),
        )
        with self.assertRaises(ValueError):
            bad.verify()

    def test_gqa_heterogeneous_memory_and_split_regions(self):
        for splits in (1, 4):
            native = lower(instantiate(attention((513, 777), splits)))
            for part in native.domain.partitions:
                self.assertEqual(part.kv_head, part.query_head // 8)
            graph = expand(native)
            requested = sum(
                op.tile[2] * op.tile[3] * 2
                for op in graph.ops
                if op.op == "cp.dram2sram_tma"
            )
            expected_kv = 2 * 32 * (513 + 777) * 128 * 2
            expected_q = 2 * 32 * splits * 128 * 2
            self.assertEqual(requested, expected_kv + expected_q)
            self.assertEqual(len(graph.ctas), 2 * 32 * splits)
            for op in graph.ops:
                if op.op.startswith("cp."):
                    t = graph.tensors[op.tensor]
                    self.assertLessEqual(op.tile[0] + op.tile[2], t.shape[0])
                    self.assertLessEqual(op.tile[1] + op.tile[3], t.shape[1])


if __name__ == "__main__":
    unittest.main()
