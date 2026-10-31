import unittest
from dataclasses import replace
from gtsim_frontend.ir.common import VerificationError
from gtsim_frontend.ir.algorithm import MathKind
from gtsim_frontend.programs.gemm_single import build
from gtsim_frontend.passes.instantiate import instantiate
from gtsim_frontend.passes.lower_hopper_gemm import lower
from gtsim_frontend.passes.lower_execution import lower as expand
from gtsim_frontend.serialization.json_codec import dumps


class NativeContractTests(unittest.TestCase):
    def test_structured_native_and_only_execution_expands(self):
        native = lower(instantiate(build().launches[0].algorithm))
        execution = expand(native)
        self.assertEqual(len(native.ops), 6)
        self.assertEqual(len(execution.ops), 9568)
        self.assertLess(len(dumps(native)), 40000)
        self.assertNotIn('"record": "ArchInstruction"', dumps(native))
        patterns = list(native.ops)
        patterns[1] = replace(
            patterns[1], steps=(replace(patterns[1].steps[0], opcode="unknown"),)
        )
        with self.assertRaises(VerificationError):
            expand(replace(native, ops=tuple(patterns)))

    def test_typed_algorithm_reductions_and_outputs(self):
        algorithm = build().launches[0].algorithm
        for operations in (
            algorithm.contract.operations[:-1],
            tuple(
                replace(op, reduction_axes=()) if op.kind == MathKind.MATMUL else op
                for op in algorithm.contract.operations
            ),
        ):
            with self.assertRaises(VerificationError):
                replace(
                    algorithm,
                    contract=replace(algorithm.contract, operations=operations),
                ).verify()


if __name__ == "__main__":
    unittest.main()
