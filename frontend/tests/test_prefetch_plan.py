"""Check capacity-driven FlowIR slicing without changing the compute graph."""

from collections import Counter
from pathlib import Path
import tempfile
import unittest

from compile import compile_program, lower_flow
from compile.opt.prefetch import Residency, optimize
from native_helpers import expand_arch


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
        original = expand_arch(lower_flow(flow))[2]

        def compute_signature(ops):
            return Counter((op.opcode, op.tensor, op.tile[2:]) for op in ops)

        for count in (2, 4):
            with self.subTest(buffers=count):
                plan = optimize(
                    flow, Residency(2, 8, 96, 32, 4096, 0, 524288, buffer_count=count)
                )
                self.assertEqual(
                    compute_signature(original),
                    compute_signature(expand_arch(lower_flow(plan.flow))[2]),
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

        def signature(ops):
            def op_signature(op):
                return op.opcode, op.tensor, op.dtype, (op.tile[2], op.tile[3])

            return Counter(
                (
                    op_signature(op),
                    tuple(op_signature(ops[d]) for d in op.after_issue),
                    # Launch ordering introduces edges between CTA slices.
                    tuple(
                        op_signature(ops[d])
                        for d in op.after_complete
                        if ops[d].cta == op.cta
                    ),
                )
                for op in ops
            )

        self.assertEqual(
            signature(expand_arch(lower_flow(flow))[2]),
            signature(expand_arch(lower_flow(plan.flow))[2]),
        )
        self.assertEqual(
            {consumer for item in plan.queue for consumer in item.consumers},
            {region.symbol for region in plan.flow.regions},
        )
        with self.assertRaisesRegex(ValueError, "full-reduction CTA exceeds"):
            optimize(flow, Residency(2, 8, 96, 32, 4096, 0, 65536))


if __name__ == "__main__":
    unittest.main()
