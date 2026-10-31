import unittest
from dataclasses import replace
from gtsim_frontend.ir.common import VerificationError
from gtsim_frontend.ir.schedule import OpRole, Phase, IndexExpr, LoopVar
from gtsim_frontend.programs.gemm_single import build as gemm
from gtsim_frontend.programs.fa3_single import build as fa3
from gtsim_frontend.passes.instantiate import instantiate
from gtsim_frontend.passes.lower_hopper_gemm import lower
from gtsim_frontend.passes.lower_execution import lower as expand
from gtsim_frontend.serialization.json_codec import dumps, loads


class ReferenceFlowTests(unittest.TestCase):
    def test_gemm_regions_odd_even_slots_and_compactness(self):
        for k in (64, 128, 192, 256):
            flow = instantiate(gemm(k=k).launches[0].algorithm, 2)
            self.assertEqual(len(flow.ops), 6)
            self.assertEqual(flow.domain.cta_count, 2)
            self.assertEqual(flow.domain.waves, 4)
            self.assertEqual(flow.domain.pipeline.loop.count, k // 64)
            stage = flow.domain.pipeline.stage
            self.assertEqual(
                stage.generation_counts_per_wave, ((k // 64 + 1) // 2, (k // 64) // 2)
            )
            self.assertEqual(flow, loads(dumps(flow)))
            a = next(op for op in flow.ops if op.role == OpRole.LOAD_A)
            self.assertEqual(
                a.region.offsets_at({LoopVar.ROW: 2, LoopVar.K: 1}), (256, 64)
            )

    def test_fa3_gqa_compact_body(self):
        small = instantiate(fa3().launches[0].algorithm)
        large = instantiate(
            fa3(heads_q=16, heads_kv=8, seq_q=1536, seq_kv=1536).launches[0].algorithm
        )
        self.assertEqual(len(small.ops), len(large.ops))
        self.assertLess(len(large.ops), 40)
        self.assertEqual(large.domain.cta_count, 192)
        self.assertEqual(large.domain.pipeline.loop.count, 12)
        self.assertEqual(large, loads(dumps(large)))

    def test_dependency_slot_region_and_executor_mutations_rejected(self):
        flow = instantiate(gemm().launches[0].algorithm)
        index = next(i for i, op in enumerate(flow.ops) if op.role == OpRole.LOAD_A)
        op = flow.ops[index]
        variants = [
            replace(
                op,
                dependencies=(replace(op.dependencies[0], phase="complete"),)
                + op.dependencies[1:],
            ),
            replace(
                op,
                dependencies=(replace(op.dependencies[0], phase=Phase.ISSUE),)
                + op.dependencies[1:],
            ),
            replace(op, slot=replace(op.slot, stage=replace(op.slot.stage, modulus=3))),
            replace(
                op,
                region=replace(op.region, offsets=(IndexExpr(1), op.region.offsets[1])),
            ),
            replace(op, executor=flow.ops[3].executor),
        ]
        store_index = next(
            i for i, item in enumerate(flow.ops) if item.role == OpRole.STORE
        )
        store = flow.ops[store_index]
        changed = list(flow.ops)
        changed[store_index] = replace(
            store, region=replace(store.region, shape=(64, 256))
        )
        with self.assertRaisesRegex(VerificationError, "region/output"):
            lower(replace(flow, ops=tuple(changed)))
        with self.assertRaisesRegex(VerificationError, "pipeline depth"):
            lower(
                replace(
                    flow,
                    domain=replace(
                        flow.domain, pipeline=replace(flow.domain.pipeline, depth=3)
                    ),
                )
            )
        for variant in variants:
            ops = list(flow.ops)
            ops[index] = variant
            with self.assertRaises(VerificationError):
                lower(replace(flow, ops=tuple(ops)))

    def test_display_names_do_not_drive_lowering(self):
        flow = instantiate(gemm(m=128, n=256, k=64).launches[0].algorithm)
        renamed = replace(
            flow,
            ops=tuple(
                replace(op, id=f"user_label_{i}") for i, op in enumerate(flow.ops)
            ),
        )
        self.assertEqual(expand(lower(flow)).ops, expand(lower(renamed)).ops)


if __name__ == "__main__":
    unittest.main()
