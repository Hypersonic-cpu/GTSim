"""Streaming panels preserve compute and obey exact per-slot physical page budgets."""

from collections import Counter
from pathlib import Path
import tempfile
import unittest

from compile import compile_program, lower_flow, lower_arch
from compile.opt.panels import lower_panels
from compile.opt.prefetch import PageBudget, Residency


class PrefetchPanelTests(unittest.TestCase):
    def test_k_windows_preserve_native_compute_and_cover_actual_consumers(self):
        library = Path(__file__).resolve().parents[1] / "lib"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.progir"
            source.write_text("""prog.program @probe {
  prog.tensor @A : tensor<4x256xbf16> storage<HBM> external readonly
  prog.tensor @B : tensor<256x512xbf16> storage<HBF> external readonly
  prog.tensor @C : tensor<4x512xbf16> storage<HBM>
  prog.stream @compute ordered {
    %C, !done = prog.launch @gemm
        callee<@small_m_gemm> implementation<@small_m_gemm_flow>
        inputs[@A,@B] outputs[@C]
        attrs{M=4,N=512,K=256,NUM_SMS=132}
    prog.synchronize @compute after[!done]
  }
  prog.return @C after[!done]
}
""")
            kernel = lower_arch(
                lower_flow(
                    compile_program(
                        source,
                        {},
                        [library / "small_m_gemm.algoir"],
                        [library / "small_m_gemm.flowir"],
                    )
                )
            )
        config = Residency(2, 8, 96, 32, 4096, 0, 32768, buffer_count=2, streaming=True)
        result = lower_panels(kernel, config, {"A": 0, "B": 0, "C": 0})
        self.assertEqual(result.ops, kernel.ops)
        self.assertEqual(result.ctas, kernel.ctas)
        self.assertGreater(len(result.prefetch_windows), 2)
        expected = {
            n.id for n in kernel.ops if n.opcode == "cp.dram2sram_tma" and n.tensor == 1
        }
        counts = Counter(n for w in result.prefetch_windows for n in w.consumers)
        self.assertEqual(set(counts), expected)
        self.assertTrue(all(c == 1 for c in counts.values()))
        for i, window in enumerate(result.prefetch_windows):
            self.assertEqual(window.buffer.slot, i % 2)
            budget = PageBudget(config, {"B": 0}, i % 2)
            pages = {
                ("B", p)
                for r in window.ranges
                for p in range(r.offset // 4096, (r.offset + r.bytes) // 4096)
            }
            self.assertTrue(budget.add(pages))
        iterations = [
            kernel.ops[n].pipeline_iteration
            for w in result.prefetch_windows
            for n in w.consumers
        ]
        self.assertEqual(iterations, sorted(iterations))

    def test_one_page_planes_are_shared_as_whole_pages(self):
        config = Residency(2, 8, 96, 32, 4096, 4096, 0, buffer_count=2, streaming=True)
        for slot in range(2):
            budget = PageBudget(config, {"B": 0}, slot)
            self.assertTrue(budget.add({("B", slot)}))
            self.assertFalse(budget.add({("B", 1 - slot)}))
