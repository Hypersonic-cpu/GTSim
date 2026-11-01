from collections import Counter
import unittest
from gtsim_frontend.ir.algorithm import Algorithm
from gtsim_frontend.ir.common import MatrixGroup
from gtsim_frontend.ir.flow import FlowKernel
from gtsim_frontend.ir.arch import ArchKernel
from gtsim_frontend.passes.instantiate import instantiate
from gtsim_frontend.passes.lower_hopper_gemm import lower
from gtsim_frontend.passes.lower_execution import lower as expand
from gtsim_frontend.serialization.json_codec import dumps, loads


class GroupedGemmTests(unittest.TestCase):
    def test_direct_and_sparse_nonuniform_groups_preserve_rows_and_weights(self):
        variants = (
            tuple(
                MatrixGroup(e, (0,), (slot,), f"W{e}")
                for slot, e in enumerate((1, 3, 7, 11, 19, 27, 51, 99))
            ),
            (
                MatrixGroup(1, (3,), (0,), "W1"),
                MatrixGroup(7, (2, 0), (1, 2), "W7"),
                MatrixGroup(99, tuple(range(17)), tuple(range(3, 20)), "W99"),
            ),
        )
        for groups in variants:
            rows = sum(len(g.input_rows) for g in groups)
            algorithm = Algorithm(
                symbol="experts",
                dimensions=(rows, 130, 24),
                dtype="bf16",
                schedule_id="grouped_small_m_mma",
                groups=groups,
            )
            flow = loads(dumps(instantiate(algorithm)))
            native = loads(dumps(lower(flow)))
            self.assertIs(type(flow), FlowKernel)
            self.assertIs(type(native), ArchKernel)
            graph = expand(native)
            writes = Counter()
            reads = Counter()
            for op in graph.ops:
                if op.op == "cp.sram2dram_tma":
                    for r in range(op.tile[0], op.tile[0] + op.tile[2]):
                        writes[r] += op.tile[3]
                if op.op == "cp.dram2sram_tma":
                    reads[graph.tensors[op.tensor].name] += op.tile[2] * op.tile[3] * 2
            self.assertEqual(writes, Counter({r: 130 for r in range(rows)}))
            self.assertEqual(reads["A"], rows * 24 * 2 * 3)
            for group in groups:
                self.assertEqual(
                    reads[group.weight],
                    ((len(group.input_rows) + 15) // 16) * 24 * 130 * 2,
                )
            self.assertEqual(set(reads), {"A"} | {g.weight for g in groups})
            expected_tiles = sum((len(g.input_rows) + 15) // 16 for g in groups) * 3
            self.assertEqual(len(graph.ctas), expected_tiles)
            self.assertEqual(
                sum(op.op == "mma.sync.m16n8k16" for op in graph.ops),
                expected_tiles * 2 * 8,
            )
            cursor = 0
            for group in groups:
                count = ((len(group.input_rows) + 15) // 16) * 3
                weight_id = next(t.id for t in graph.tensors if t.name == group.weight)
                for cta in range(cursor, cursor + count):
                    ids = {
                        op.tensor
                        for op in graph.ops
                        if op.cta == cta and op.op == "cp.dram2sram_tma"
                    }
                    self.assertEqual(ids, {0, weight_id})
                cursor += count

    def test_duplicate_or_missing_logical_output_is_rejected(self):
        algorithm = Algorithm(
            symbol="bad",
            dimensions=(2, 64, 16),
            dtype="bf16",
            schedule_id="grouped_small_m_mma",
            groups=(MatrixGroup(0, (0, 1), (0, 0), "W0"),),
        )
        with self.assertRaisesRegex(ValueError, "each logical row exactly once"):
            algorithm.verify()


if __name__ == "__main__":
    unittest.main()
