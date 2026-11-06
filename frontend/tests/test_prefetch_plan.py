"""Check capacity-driven FlowIR slicing without changing the compute graph."""

from collections import Counter
from pathlib import Path
import tempfile
import unittest

from compile import compile_program, lower_arch, lower_flow
from compile.opt.prefetch import Residency, optimize


class PrefetchPlanTests(unittest.TestCase):
    def test_rotating_buffer_capacity_release_and_dataflow(self):
        library = Path(__file__).resolve().parents[1] / "lib"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "buffered.progir"
            source.write_text(
                """prog.program @buffered {
  prog.tensor @A : tensor<128x64xfp16> storage<HBM> external readonly
  prog.tensor @B : tensor<64x16384xfp16> storage<HBF> external readonly
  prog.tensor @C : tensor<128x16384xfp16> storage<HBM>
  prog.stream @compute ordered {
    %C, !done = prog.launch @project
        callee<@gemm_reference> implementation<@gemm_reference_flow>
        inputs[@A, @B] outputs[@C]
        attrs{M=128, N=16384, K=64, NUM_SMS=132}
    prog.synchronize @compute after[!done]
  }
  prog.return @C after[!done]
}
"""
            )
            flow = compile_program(
                source,
                {},
                list(library.glob("*.algoir")),
                list(library.glob("*.flowir")),
            )
        original = lower_arch(lower_flow(flow))

        def compute_signature(execution):
            return Counter((op.opcode, op.tensor, op.tile) for op in execution.ops)

        for count in (2, 4):
            with self.subTest(buffers=count):
                plan = optimize(
                    flow, Residency(2, 8, 96, 32, 4096, 0, 524288, buffer_count=count)
                )
                self.assertEqual(
                    compute_signature(original),
                    compute_signature(lower_arch(lower_flow(plan.flow))),
                )
                uses = {}
                for item in plan.queue:
                    self.assertEqual(item.buffer.slot, item.buffer.epoch % count)
                    uses.setdefault(item.buffer.epoch, []).append(item)
                self.assertGreater(len(uses), count)
                positions = {
                    region.symbol: i for i, region in enumerate(plan.flow.regions)
                }
                for epoch, items in uses.items():
                    self.assertLessEqual(
                        sum(item.span.bytes for item in items), 1048576 // count
                    )
                    if epoch >= count:
                        previous = uses[epoch - count]
                        released = (
                            max(
                                positions[c]
                                for item in previous
                                for c in item.consumers
                            )
                            + 1
                        )
                        self.assertTrue(
                            all(item.enqueue_at >= released for item in items)
                        )
                self.assertGreater(len({item.enqueue_at for item in plan.queue}), 1)

    def test_full_reduction_cta_slicing_and_lifetimes(self):
        library = Path(__file__).resolve().parents[1] / "lib"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "wide.progir"
            source.write_text(
                """prog.program @wide {
  prog.tensor @A : tensor<128x64xfp16> storage<HBM> external readonly
  prog.tensor @B : tensor<64x4096xfp16> storage<HBF> external readonly
  prog.tensor @C : tensor<128x4096xfp16> storage<HBM>
  prog.stream @compute ordered {
    %C, !done = prog.launch @project
        callee<@gemm_reference> implementation<@gemm_reference_flow>
        inputs[@A, @B] outputs[@C]
        attrs{M=128, N=4096, K=64, NUM_SMS=132}
    prog.synchronize @compute after[!done]
  }
  prog.return @C after[!done]
}
"""
            )
            flow = compile_program(
                source,
                {},
                list(library.glob("*.algoir")),
                list(library.glob("*.flowir")),
            )
        plan = optimize(flow, Residency(2, 8, 96, 32, 4096, 0, 131072))
        self.assertEqual(len(plan.flow.regions), 2)
        self.assertEqual(len(plan.queue), 128)
        self.assertTrue(
            all(p.enqueue_at == 0 and len(p.consumers) == 1 for p in plan.queue)
        )

        def signature(execution):
            def op_signature(op):
                return op.opcode, op.tensor, op.dtype, op.tile

            return Counter(
                (
                    op_signature(op),
                    tuple(op_signature(execution.ops[d]) for d in op.after_issue),
                    # Launch ordering introduces edges between CTA slices.
                    tuple(
                        op_signature(execution.ops[d])
                        for d in op.after_complete
                        if execution.ops[d].cta == op.cta
                    ),
                )
                for op in execution.ops
            )

        self.assertEqual(
            signature(lower_arch(lower_flow(flow))),
            signature(lower_arch(lower_flow(plan.flow))),
        )
        launches = [{"id": r.symbol} for r in plan.flow.regions]
        manifest = plan.decorate(
            {"tensors": [{"name": "B", "id": 1}], "launches": launches}
        )
        active = set()
        for command in manifest["commands"]:
            kind, target = command["kind"], command["target"]
            if kind == "QueuePrefetch":
                active.add(target)
            elif kind == "LaunchKernel":
                self.assertTrue(
                    set(next(x for x in launches if x["id"] == target)["read_aliases"])
                    <= active
                )
            elif kind == "Release":
                active.remove(target)
        self.assertFalse(active)
        with self.assertRaisesRegex(ValueError, "full-reduction CTA exceeds"):
            optimize(flow, Residency(2, 8, 96, 32, 4096, 0, 65536))


if __name__ == "__main__":
    unittest.main()
