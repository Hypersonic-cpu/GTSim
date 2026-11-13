"""Prefetch windows are generated from streaming NativeExpander callbacks."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from compile import compile_program, lower_flow
from compile.expand_native import NativeExpander
from compile.opt.prefetch import PageBudget, Residency, optimize
from compile.static import EMPTY, Resolver, bound_resolver
from compile.streaming import ExecutionIRProducer, ProgramPlan, _strides
from parser import ast as A


class PrefetchPanelTests(unittest.TestCase):
    def test_multiple_ctas_and_pipeline_iterations_use_cta_wave_index(self):
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
            flow = compile_program(source, {}, [library / "small_m_gemm.algoir"],
                                   [library / "small_m_gemm.flowir"])
        config = Residency(2, 8, 96, 32, 4096, 0, 32768, buffer_count=2, streaming=True)
        plan = optimize(flow, config, {"A": None, "B": None, "C": None})
        arch = lower_flow(plan.flow)
        root = bound_resolver(arch.static, Resolver(EMPTY, {})).materialize_constants()
        tensors = []
        for spec in arch.tensors:
            shape = tuple(root.value(d) for d in spec.shape)
            tensors.append(A.KernelTensor(len(tensors), spec.id, shape, _strides(shape),
                                          spec.dtype, spec.tier or A.MemorySpace.HBM,
                                          spec.external, spec.read_only, None))
        region = arch.regions[0]
        args = {a.id: root.value(a.expression) for a in region.static.arguments}
        resolver = Resolver(replace(region.static, parameters=(), constraints=()),
                            {**root.values, **args}).materialize_constants()
        ctas, loads = [], []
        def emit_instruction(node):
            if node.opcode == A.NativeOpcode.TMA_LOAD.value:
                loads.append((node.cta, node.pipeline_iteration, node.id, node.tensor, node.tile))
        NativeExpander(region, resolver, {t.name: t for t in tensors}).expand_to(ctas.append,
                                                                                   emit_instruction)
        producer = ExecutionIRProducer(arch, ProgramPlan("compute", (), (), ()),
                                       {"prefetch_config": config, "prefetch_bases": plan.bases})
        windows = producer._prefetch_windows(loads, ctas, tensors)
        self.assertGreater(len(ctas), 1)
        self.assertGreater(len({item[1] for item in loads if item[1] is not None}), 1)
        eligible = {node for cta, _, node, tensor, _ in loads
                    if tensors[tensor].tier == A.MemorySpace.HBF
                    and tensors[tensor].preloaded and tensors[tensor].read_only}
        consumers = [node for window in windows for node in window.consumers]
        self.assertEqual(len(consumers), len(set(consumers)))
        self.assertEqual(eligible, set(consumers))
        for index, window in enumerate(windows):
            self.assertEqual(window.buffer.slot, index % config.buffer_count)
            budget = PageBudget(config, plan.bases, index % config.buffer_count)
            names = {tensor.id: tensor.name for tensor in tensors}
            self.assertTrue(budget.add({(names[t.tensor], page)
                                        for t in window.ranges
                                        for page in range(t.offset // config.page_bytes,
                                                           (t.offset + t.bytes) // config.page_bytes)}))


if __name__ == "__main__":
    unittest.main()
