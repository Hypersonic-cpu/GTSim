"""Small-M uses canonical GEMM math, Flow, Native patterns, and emitter."""

import unittest
from gtsim_frontend.ir.algorithm import Algorithm
from gtsim_frontend.ir.flow import FlowKernel
from gtsim_frontend.ir.arch import ArchKernel
from gtsim_frontend.passes.instantiate import instantiate
from gtsim_frontend.passes.lower_hopper_gemm import lower
from gtsim_frontend.passes.lower_execution import lower as expand
from gtsim_frontend.serialization.json_codec import dumps, loads


class CanonicalSmallMTests(unittest.TestCase):
    def test_masked_tails_and_traffic_oracle(self):
        for m in (1, 8, 16):
            algorithm = loads(
                dumps(
                    Algorithm(
                        symbol="skinny",
                        dimensions=(m, 130, 24),
                        dtype="bf16",
                        schedule_id="small_m_mma",
                    )
                )
            )
            flow = loads(dumps(instantiate(algorithm, 132)))
            self.assertIs(type(flow), FlowKernel)
            self.assertEqual(
                {
                    result.dtype
                    for op in flow.ops
                    if op.kind.value == "async_copy"
                    for result in op.results
                },
                {"bf16"},
            )
            native = loads(dumps(lower(flow)))
            self.assertIs(type(native), ArchKernel)
            graph = expand(native)
            loads_by_tensor = [0, 0, 0]
            stores = 0
            matrix = 0
            for op in graph.ops:
                if op.op == "cp.dram2sram_tma":
                    loads_by_tensor[op.tensor] += op.tile[2] * op.tile[3] * 2
                elif op.op == "cp.sram2dram_tma":
                    stores += op.tile[2] * op.tile[3] * 2
                elif op.op == "mma.sync.m16n8k16":
                    matrix += 1
            # Three N tiles each consume A; each valid weight element is read once.
            self.assertEqual(loads_by_tensor, [3 * m * 24 * 2, 24 * 130 * 2, 0])
            self.assertEqual(stores, m * 130 * 2)
            self.assertEqual(matrix, 3 * 2 * 4 * 2)
            self.assertEqual({t.dtype for t in graph.tensors}, {"bf16"})


if __name__ == "__main__":
    unittest.main()
