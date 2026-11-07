"""Verify wide TMA boxes and full-K fragment accumulation in the shared lowering."""

from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compile import compile_program, lower_flow, lower_arch
from run_gemm_tiling_search import flow_variant


class SkinnyTilingTests(unittest.TestCase):
    def test_wide_copy_and_k_fragments(self):
        library = Path(__file__).resolve().parents[1] / "lib"
        for bn, bk in ((512, 64), (32, 256)):
            with self.subTest(BN=bn, BK=bk), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                flow = folder / "variant.flowir"
                flow.write_text(flow_variant("small_m_gemm", bn, bk))
                source = folder / "input.progir"
                source.write_text(f"""prog.program @probe {{
  prog.tensor @A : tensor<4x{bk}xbf16> storage<HBM> external readonly
  prog.tensor @B : tensor<{bk}x512xbf16> storage<HBM> external readonly
  prog.tensor @C : tensor<4x512xbf16> storage<HBM>
  prog.stream @compute ordered {{
    %C, !done = prog.launch @gemm
        callee<@small_m_gemm> implementation<@small_m_gemm_flow_n{bn}_k{bk}>
        inputs[@A,@B] outputs[@C]
        attrs{{M=4,N=512,K={bk},NUM_SMS=132}}
    prog.synchronize @compute after[!done]
  }}
  prog.return @C after[!done]
}}
""")
                execution = lower_arch(
                    lower_flow(
                        compile_program(
                            source, {}, [library / "small_m_gemm.algoir"], [flow]
                        )
                    )
                )
                tensors = {t.name: t.id for t in execution.tensors}
                loads = [n for n in execution.ops if n.opcode == "cp.dram2sram_tma"]
                stores = [n for n in execution.ops if n.opcode == "cp.sram2dram_tma"]
                matrix = [n for n in execution.ops if n.opcode == "mma.sync.m16n8k16"]
                self.assertEqual(len(matrix), 512 // 8 * (bk // 16))
                self.assertTrue(
                    all(
                        0 < n.tile[2] <= 256 and 0 < n.tile[3] <= 256
                        for n in (*loads, *stores)
                    )
                )
                self.assertEqual(
                    sum(
                        n.tile[2] * n.tile[3] for n in loads if n.tensor == tensors["B"]
                    ),
                    bk * 512,
                )
                self.assertEqual(sum(n.tile[2] * n.tile[3] for n in stores), 4 * 512)
                for cta in execution.ctas:
                    weights = {
                        n.id
                        for n in loads
                        if n.cta == cta.id and n.tensor == tensors["B"]
                    }
                    accumulators = {}
                    for n in matrix:
                        if n.cta != cta.id:
                            continue
                        self.assertTrue(weights.issubset(n.after_complete))
                        if n.warp in accumulators:
                            self.assertIn(accumulators[n.warp], n.after_complete)
                        accumulators[n.warp] = n.id
                if bn == 512:
                    self.assertEqual(
                        len([n for n in loads if n.tensor == tensors["B"]]), 2
                    )
                    self.assertEqual(len(stores), 2)


if __name__ == "__main__":
    unittest.main()
