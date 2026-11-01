import math
import unittest
from dataclasses import replace
from gtsim_frontend.ir.algorithm import Algorithm
from gtsim_frontend.ir.primitives import NormSpec, AddSpec, SwiGLUSpec, RopeSpec
from gtsim_frontend.passes.instantiate import instantiate
from gtsim_frontend.passes.hopper.lower import lower
from gtsim_frontend.passes.lower_execution import lower as expand
from gtsim_frontend.serialization.json_codec import loads, dumps
from native_math_oracle import evaluate_vector


def compile_spec(spec):
    algorithm = Algorithm(
        symbol="primitive",
        family="vector",
        dtype="bf16",
        schedule_id="simt_vector",
        primitive=spec,
    )
    flow = loads(dumps(instantiate(algorithm)))
    native = loads(dumps(lower(flow)))
    return flow, native


class CanonicalVectorTests(unittest.TestCase):
    def assert_matrix(self, expected, actual):
        for row, got in zip(expected, actual):
            for x, y in zip(row, got):
                self.assertAlmostEqual(x, y, places=10)

    def test_rmsnorm_epsilon_learned_scale_and_residual(self):
        x = [[math.sin((r + c) / 7) for c in range(128)] for r in range(2)]
        w = [[1 + c / 1000 for c in range(128)]]
        _, native = compile_spec(NormSpec(2, 128, 1e-6))
        actual = evaluate_vector(native, {"X": x, "Weight": w})["Output"]
        expected = [
            [
                v / math.sqrt(sum(a * a for a in row) / 128 + 1e-6) * weight
                for v, weight in zip(row, w[0])
            ]
            for row in x
        ]
        self.assert_matrix(expected, actual)
        y = [[1 - v for v in row] for row in x]
        _, native = compile_spec(AddSpec(2, 128))
        self.assert_matrix(
            [[1] * 128] * 2, evaluate_vector(native, {"X": x, "Y": y})["Output"]
        )

    def test_swiglu_uses_distinct_gate_up_regions(self):
        data = [[math.sin((r + c) / 11) for c in range(256)] for r in range(2)]
        _, native = compile_spec(SwiGLUSpec(2, 128))
        actual = evaluate_vector(native, {"X": data})["Output"]
        expected = [
            [row[c] / (1 + math.exp(-row[c])) * row[128 + c] for c in range(128)]
            for row in data
        ]
        self.assert_matrix(expected, actual)
        graph = expand(native)
        regions = {op.tile for op in graph.ops if op.op == "cp.dram2sram_tma"}
        self.assertEqual(
            regions,
            {(0, 0, 1, 128), (0, 128, 1, 128), (1, 0, 1, 128), (1, 128, 1, 128)},
        )

    def test_qk_norm_rope_positions_theta_and_rotate_half(self):
        data = [[math.sin((r * 37 + c) / 23) for c in range(512)] for r in range(2)]
        weight = [[1 + c / 500 for c in range(128)]]
        spec = RopeSpec(2, 2, 128, 512, 64, (10000, 32000))
        _, native = compile_spec(spec)
        actual = evaluate_vector(native, {"Input": data, "Weight": weight})["Output"]
        expected = []
        for request in range(2):
            for head in range(2):
                x = data[request][64 + head * 128 : 64 + (head + 1) * 128]
                scale = 1 / math.sqrt(sum(v * v for v in x) / 128 + 1e-6)
                x = [v * scale * w for v, w in zip(x, weight[0])]
                angles = [
                    spec.positions[request] * spec.theta ** (-2 * d / 128)
                    for d in range(64)
                ]
                expected.append(
                    [
                        x[d] * math.cos(a) - x[d + 64] * math.sin(a)
                        for d, a in enumerate(angles)
                    ]
                    + [
                        x[d + 64] * math.cos(a) + x[d] * math.sin(a)
                        for d, a in enumerate(angles)
                    ]
                )
        self.assert_matrix(expected, actual)
        _, different = compile_spec(replace(spec, positions=(10001, 32001)))
        self.assertNotEqual(
            actual,
            evaluate_vector(different, {"Input": data, "Weight": weight})["Output"],
        )
        _, different = compile_spec(replace(spec, theta=10000.0))
        self.assertNotEqual(
            actual,
            evaluate_vector(different, {"Input": data, "Weight": weight})["Output"],
        )
        graph = expand(native)
        self.assertEqual(sum(o.op == "sfu.sin" for o in graph.ops), 4)
        self.assertEqual(sum(o.op == "sfu.cos" for o in graph.ops), 4)
        self.assertEqual(
            sum(o.tile[2] * o.tile[3] for o in graph.ops if o.op == "cp.sram2dram_tma"),
            4 * 128,
        )

    def test_rope_position_parameter_cannot_be_unbound(self):
        flow, _ = compile_spec(RopeSpec(2, 2, 128, 512, 64, (10000, 32000)))
        bad = replace(
            flow,
            domain=replace(
                flow.domain,
                workgroups=tuple(
                    replace(g, constants=()) for g in flow.domain.workgroups
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "unbound mathematical group parameter"):
            bad.verify()

    def test_vector_dependency_mutations_round_trip_and_names(self):
        from gtsim_frontend.ir.schedule import Dependency, Phase, Relation

        flow, native = compile_spec(NormSpec(1, 128))
        edge = Dependency("load_w", Phase.COMPLETE, Relation.EXPLICIT)
        changed = replace(
            flow,
            ops=tuple(
                replace(op, dependencies=(edge,)) if op.role == "x_fp32" else op
                for op in flow.ops
            ),
        )
        self.assertNotEqual(
            expand(native).ops, expand(lower(loads(dumps(changed)))).ops
        )
        renamed = replace(
            flow,
            ops=tuple(replace(op, id=f"display_{i}") for i, op in enumerate(flow.ops)),
        )
        self.assertEqual(expand(native).ops, expand(lower(loads(dumps(renamed)))).ops)
        bad = replace(
            flow,
            ops=tuple(
                replace(
                    op,
                    operands=(replace(op.operands[0], offsets=(0, 127), shape=(1, 2)),),
                )
                if op.role == "x_fp32"
                else op
                for op in flow.ops
            ),
        )
        with self.assertRaisesRegex(ValueError, "register view"):
            bad.verify()


if __name__ == "__main__":
    unittest.main()
