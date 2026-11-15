import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from compile import compile_program
from compile.opt.partition import PartitionPolicy, PhysicalLayout, partition_flow
from compile.opt.prefetch import Residency, optimize
from parser import ast as A


class FlashAccelBarrierTests(unittest.TestCase):
    def test_expert_prefetch_waits_for_router_completion(self):
        root = Path(__file__).resolve().parents[1]
        flow = compile_program(
            root / "asm/qwen3/flash-accel/one_block.progir",
            json.loads((root / "asm/qwen3/b1.json").read_text()),
            list((root / "lib").glob("*.algoir")),
            list((root / "lib").glob("*.flowir")),
        )
        flow = replace(
            flow,
            tensors=tuple(
                replace(tensor, tier=A.MemorySpace.HBF)
                if tensor.external and tensor.read_only
                else tensor
                for tensor in flow.tensors
            ),
        )
        partition = partition_flow(
            flow,
            PartitionPolicy(),
            PhysicalLayout.from_memory_yaml(
                Path("configs/memconf/4F1M_grouped_planes.yaml")
            ),
        )
        layout = PhysicalLayout.from_memory_yaml(
            Path("configs/memconf/4F1M_grouped_planes.yaml")
        )
        plan = optimize(
            partition.flow,
            Residency(
                layout.stacks,
                layout.dies_per_stack,
                layout.planes_per_die,
                layout.planes_per_command_group,
                layout.page_bytes,
                layout.sram_bytes_per_plane,
                layout.base_die_sram_bytes_per_stack,
                buffer_count=2,
                streaming=True,
            ),
            origin_launches=partition.origin_launches,
        )
        router = next(
            index
            for index, region in enumerate(plan.flow.regions)
            if region.symbol == "router_softmax_topk"
        )
        expert = [
            item
            for item in plan.queue
            if item.span.tensor in {"W_gate_up", "W_down"}
        ]
        self.assertTrue(expert)
        self.assertTrue(all(item.enqueue_at > router for item in expert))
        self.assertFalse(
            any(item.enqueue_at <= router for item in expert)
        )

        # A partition is a view into the original row-major allocation.  Its
        # logical N is 64, but TMA must retain the parent row stride (1536 for
        # GateUp and 2048 for Down), including non-zero slab offsets.
        for prefix, tensor, stride in (
            ("expert_gemm1_direct", "W_gate_up", 1536),
            ("expert_gemm2_direct", "W_down", 2048),
        ):
            regions = [
                region for region in partition.flow.regions
                if region.symbol.startswith(prefix)
            ]
            weights = [
                op.region
                for region in regions
                for op in region.ops
                if op.region is not None and op.region.tensor == tensor
            ]
            self.assertTrue(weights)
            self.assertTrue(
                all(
                    view.parent_row_stride is not None
                    and view.parent_row_stride.expression == str(stride)
                    for view in weights
                )
            )

    def test_partition_preserves_parent_stride_at_nonzero_slab(self):
        root = Path(__file__).resolve().parents[1]
        source_text = """prog.program @partition_stride {
  prog.tensor @A : tensor<1x16xbf16> storage<HBM> external readonly
  prog.tensor @B : tensor<16x4096xbf16> storage<HBF> external readonly
  prog.tensor @C : tensor<1x4096xbf16> storage<HBM>
  prog.stream @compute ordered {
    %C, !done = prog.launch @project
        callee<@small_m_gemm> implementation<@small_m_gemm_flow>
        inputs[@A, @B] outputs[@C]
        attrs{M=1, N=4096, K=16, NUM_SMS=1}
    prog.synchronize @compute after[!done]
  }
  prog.return @C after[!done]
}
"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "partition_stride.progir"
            source.write_text(source_text)
            flow = compile_program(
                source,
                {},
                list((root / "lib").glob("*.algoir")),
                list((root / "lib").glob("*.flowir")),
            )
        flow = replace(
            flow,
            tensors=tuple(
                replace(tensor, tier=A.MemorySpace.HBF)
                if tensor.id == "B"
                else tensor
                for tensor in flow.tensors
            ),
        )
        result = partition_flow(
            flow,
            PartitionPolicy(),
            PhysicalLayout(
                stacks=1,
                dies_per_stack=1,
                planes_per_die=1,
                planes_per_command_group=1,
                page_bytes=4096,
                sram_bytes_per_plane=131072,
                base_die_sram_bytes_per_stack=0,
            ),
        )
        views = [
            op.region
            for region in result.flow.regions
            for op in region.ops
            if op.region is not None and op.region.tensor == "B"
        ]
        self.assertGreater(len(views), 1)
        self.assertTrue(any("+2048" in view.offsets[1].expression for view in views))
        self.assertTrue(
            all(
                view.parent_row_stride is not None
                and view.parent_row_stride.expression == "4096"
                for view in views
            )
        )

    def test_megaplane_mapping_matches_residency_for_on_and_off(self):
        layout = PhysicalLayout.from_memory_yaml(
            Path("configs/memconf/4F1M_grouped_planes.yaml")
        )
        for lanes in (32, 1):
            with self.subTest(planes_per_command_group=lanes):
                candidate = replace(layout, planes_per_command_group=lanes)
                residency = Residency(
                    candidate.stacks,
                    candidate.dies_per_stack,
                    candidate.planes_per_die,
                    lanes,
                    candidate.page_bytes,
                    candidate.sram_bytes_per_plane,
                    candidate.base_die_sram_bytes_per_stack,
                    address_mapping_policy=candidate.address_mapping_policy,
                )
                for page in (0, 1, 31, 32, 95, 96, 191, 192, 193, 4095):
                    self.assertEqual(candidate.location(page), residency.location(page))
        with self.assertRaises(ValueError):
            replace(layout, address_mapping_policy="linear_page").location(0)


if __name__ == "__main__":
    unittest.main()
