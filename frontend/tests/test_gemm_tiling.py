"""Tests for the structured Qwen small-M/grouped GEMM tiling pass."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compile import compile_program, lower_flow
from compile.opt.gemm_tiling import (
    apply_gemm_tiling,
    scratch_bytes,
    validate_gemm_flow,
)
from compile.static import EMPTY, Resolver, bound_resolver
from parser import ast as A


class StructuredGemmTilingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.source = root / "frontend/asm/qwen3/flash-accel/one_block.progir"
        cls.library = root / "frontend/lib"
        params = json.loads(
            (root / "frontend/asm/qwen3/qwen3-235b-a22b-b1.json").read_text()
        )
        cls.flow = compile_program(
            cls.source,
            params,
            list(cls.library.glob("*.algoir")),
            list(cls.library.glob("*.flowir")),
        )
        cls.root = bound_resolver(
            cls.flow.static, Resolver(EMPTY, {})
        ).materialize_constants()

    def test_t0_keeps_reference_geometry(self):
        tiled = apply_gemm_tiling(self.flow, 16, 64, 16, 2)
        self.assertEqual(tiled, self.flow)
        validate_gemm_flow(tiled, self.root)
        self.assertEqual(scratch_bytes(16, 64, 16), 7168)

    def test_variant_changes_structured_shapes_and_resources(self):
        tiled = apply_gemm_tiling(self.flow, 32, 128, 64, 2)
        region = next(item for item in tiled.regions if item.symbol == "qkv_projection")
        self.assertEqual(region.config.tile_shape, (32, 128, 64))
        self.assertEqual(region.buffers[0].bytes_per_slot, scratch_bytes(32, 128, 64))
        self.assertEqual(region.domain.pipeline.loop.stop, "ceildiv(%K,64)")
        self.assertEqual(
            next(item for item in region.ops if item.id == "load_B").region.shape,
            (64, 128),
        )
        self.assertEqual(
            next(item for item in region.ops if item.id == "matmul").results[0].shape,
            (32, 128),
        )
        arch = lower_flow(tiled)
        lowered = next(item for item in arch.regions if item.symbol == "qkv_projection")
        self.assertEqual(lowered.resources.shared_bytes, scratch_bytes(32, 128, 64))
        self.assertLessEqual(lowered.resources.shared_bytes, 227 * 1024)

    def test_illegal_candidate_is_rejected_before_simulation(self):
        tiled = apply_gemm_tiling(self.flow, 16, 256, 16, 2)
        with self.assertRaisesRegex(ValueError, "router_projection.*BN=256"):
            validate_gemm_flow(tiled, self.root)


if __name__ == "__main__":
    unittest.main()
