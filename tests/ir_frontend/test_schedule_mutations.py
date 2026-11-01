"""Canonical structured IR, independently serialized, controls DAG edges."""

from dataclasses import replace
import unittest

from gtsim_frontend.ir.schedule import Dependency, OpRole, Phase, Relation
from gtsim_frontend.passes.instantiate import instantiate
from gtsim_frontend.passes.lower_hopper_gemm import lower
from gtsim_frontend.passes.lower_execution import lower as expand
from gtsim_frontend.programs.gemm_single import build
from gtsim_frontend.serialization.json_codec import dumps, loads


class ScheduleMutationTests(unittest.TestCase):
    def setUp(self):
        self.flow = instantiate(build(k=64).launches[0].algorithm, 2)
        self.edge = Dependency(OpRole.LOAD_A, Phase.COMPLETE, Relation.EXPLICIT)

    def test_serialized_flow_dependency_changes_execution(self):
        ops = tuple(
            replace(op, dependencies=op.dependencies + (self.edge,))
            if op.role == OpRole.LOAD_B
            else op
            for op in self.flow.ops
        )
        changed = loads(dumps(replace(self.flow, ops=ops)))
        before = expand(lower(self.flow))
        after = expand(lower(changed))
        self.assertEqual(len(before.ops), len(after.ops))
        self.assertTrue(
            any(
                a.after_complete != b.after_complete
                for a, b in zip(before.ops, after.ops)
            )
        )

    def test_serialized_native_dependency_changes_execution(self):
        native = lower(self.flow)
        patterns = tuple(
            replace(
                pattern,
                steps=tuple(
                    replace(step, dependencies=step.dependencies + (self.edge,))
                    for step in pattern.steps
                ),
            )
            if pattern.source.role == OpRole.LOAD_B
            else pattern
            for pattern in native.ops
        )
        changed = loads(dumps(replace(native, ops=patterns)))
        self.assertNotEqual(expand(native).ops, expand(changed).ops)

    def test_forward_explicit_dependencies_rejected(self):
        bad = Dependency(OpRole.STORE, Phase.COMPLETE, Relation.EXPLICIT)
        ops = tuple(
            replace(op, dependencies=op.dependencies + (bad,))
            if op.role == OpRole.LOAD_A
            else op
            for op in self.flow.ops
        )
        with self.assertRaisesRegex(ValueError, "earlier scheduled producer"):
            replace(self.flow, ops=ops).verify()
        native = lower(self.flow)
        patterns = tuple(
            replace(
                pattern,
                steps=tuple(
                    replace(step, dependencies=step.dependencies + (bad,))
                    for step in pattern.steps
                ),
            )
            if pattern.source.role == OpRole.LOAD_A
            else pattern
            for pattern in native.ops
        )
        with self.assertRaisesRegex(ValueError, "illegal explicit native"):
            replace(native, ops=patterns).verify()


if __name__ == "__main__":
    unittest.main()
