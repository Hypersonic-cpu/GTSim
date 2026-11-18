"""Audit the executable SM90 packed-attention graph and independent math."""

from collections import Counter
from pathlib import Path
import sys
import unittest

from native_helpers import expand_arch
from parser import ast as A


FRONTEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRONTEND / "tools"))
from qwen_fa3_oracle import dense_gqa, split_gqa
from run_qwen_simulations import plan_program

try:
    import torch
except ImportError:
    torch = None


def attention_arch(**overrides):
    params = dict(B=1, HQ=8, HKV=2, D=32, CONTEXTS=[256], R=3, S=2,
                  TILE=32, NUM_SMS=132, STAGES=2)
    params.update(overrides)
    _, _, arch, _, _, _ = plan_program(
        FRONTEND / "asm/flash-accel/fa3_decode.progir", FRONTEND / "lib", params
    )
    return arch


class PackedDecodeTests(unittest.TestCase):
    def test_program_overrides_packed_default_and_stage_count(self):
        legacy = attention_arch(PACK_GQA=0)
        self.assertEqual([region.symbol for region in legacy.regions],
                         ["legacy_split", "legacy_combine"])
        direct = attention_arch(PACK_GQA=0, S=1)
        self.assertEqual([region.symbol for region in direct.regions], ["legacy_direct"])
        packed = attention_arch(STAGES=3)
        self.assertEqual(packed.regions[0].domain.pipeline.depth, 3)

    def test_native_addresses_and_dependencies(self):
        for lengths, hq, hkv, dim, packed, dtype in (
            ([97, 131], 12, 2, 24, 4, "fp16"),
            ([257], 10, 2, 64, 3, "bf16"),
            ([129], 64, 4, 128, 16, "bf16"),
        ):
            with self.subTest(lengths=lengths, hq=hq, d=dim):
                s, tile, stages = 2, 32, 2
                arch = attention_arch(B=len(lengths), HQ=hq, HKV=hkv, D=dim,
                                      CONTEXTS=[n - 1 for n in lengths],
                                      R=packed, QKV_DTYPE=dtype, OUTPUT_DTYPE=dtype)
                tensors, ctas, ops = expand_arch(arch)
                names = {tensor.id: tensor.name for tensor in tensors}
                split_ops = [op for op in ops if op.name.startswith("packed_split_")]
                combine_ops = [op for op in ops if op.name.startswith("packed_combine_")]
                group, qblocks = hq // hkv, (hq // hkv + packed - 1) // packed
                self.assertEqual(len(ctas), len(lengths) * (hkv * qblocks * s + hq))
                by_cta = {}
                for op in split_ops:
                    by_cta.setdefault(op.cta, []).append(op)
                expected_rows = Counter()
                for cta, nodes in by_cta.items():
                    split = cta % s
                    block = (cta // s) % qblocks
                    kvhead = (cta // (s * qblocks)) % hkv
                    b = cta // (s * qblocks * hkv)
                    first_head = kvhead * group + block * packed
                    rows = min(packed, group - block * packed)
                    begin = lengths[b] * split // s
                    end = lengths[b] * (split + 1) // s
                    base = sum(lengths[:b]) * hkv + kvhead * lengths[b]
                    by_id = {op.id: op for op in nodes}
                    loads = [op for op in nodes if op.opcode == A.NativeOpcode.TMA_LOAD.value]
                    # No duplicate K/V load per query head or padded row.
                    self.assertEqual([op.tile for op in loads if names[op.tensor] == "Q"],
                                     [(b * hq + first_head, 0, rows, dim)])
                    for tensor, prior in (("K", "_qk_"), ("V", "_pv_")):
                        accesses = [op for op in loads if names[op.tensor] == tensor]
                        self.assertEqual([op.tile for op in accesses],
                            [(base + j, 0, min(tile, end - j), dim)
                             for j in range(begin, end, tile)])
                        for k, op in enumerate(accesses):
                            predecessors = [by_id[i] for i in op.after_complete]
                            if k < stages:
                                self.assertFalse(predecessors)
                            else:
                                self.assertTrue(predecessors)
                                self.assertTrue(all(prior in p.name and p.pipeline_iteration == k - stages
                                                    for p in predecessors))
                    for pattern in ("_qk_", "_pv_"):
                        matrix = [op for op in nodes if pattern in op.name]
                        self.assertTrue(matrix)
                        self.assertTrue(all(op.opcode == A.NativeOpcode.MATRIX.value
                                            for op in matrix))
                    cast = [op for op in nodes if "_prob_cast_" in op.name]
                    self.assertTrue(cast)
                    self.assertTrue(all(op.opcode == A.NativeOpcode.CAST.value and
                                        op.dtype.value == dtype for op in cast))
                    pv = next(op for op in nodes if "_pv_" in op.name)
                    self.assertTrue(any("_prob_cast_" in by_id[j].name
                                        for j in pv.after_complete))
                    expected_base = (b * s + split) * hq + first_head
                    for op in nodes:
                        if op.opcode == A.NativeOpcode.TMA_STORE.value:
                            self.assertEqual(op.tile[:3], (expected_base, 0, rows))
                            expected_rows.update((names[op.tensor], row) for row in
                                                 range(expected_base, expected_base + rows))
                self.assertEqual(expected_rows, Counter({
                    (tensor, row): 1 for tensor in ("Partial", "LSE")
                    for row in range(len(lengths) * hq * s)
                }))
                combine_reads = Counter()
                for op in combine_ops:
                    if op.opcode == A.NativeOpcode.TMA_LOAD.value:
                        batch, head = divmod(op.cta, hq)
                        tensor = names[op.tensor]
                        self.assertIn(op.tile[0], [(batch * s + split) * hq + head
                                                   for split in range(s)])
                        combine_reads[tensor, op.tile[0]] += 1
                self.assertEqual(combine_reads, expected_rows)

    def test_reject_empty_split_and_excess_smem(self):
        with self.assertRaisesRegex(ValueError, "nonempty splits"):
            attention_arch(CONTEXTS=[0], S=2)
        with self.assertRaisesRegex(ValueError, "shared memory"):
            attention_arch(D=128, TILE=256, STAGES=3)

    def test_qwen_physical_stage_budget(self):
        split = attention_arch(HQ=64, HKV=4, D=128, R=16, S=32,
                               TILE=128, CONTEXTS=[10000]).regions[0]
        buffers = {buffer.buffer: buffer for buffer in split.layouts}
        self.assertEqual(buffers["q"].slots, 1)
        self.assertEqual(buffers["k_stage"].slots, 2)
        self.assertEqual(buffers["v_stage"].slots, 2)
        self.assertEqual(sum(buffer.slots * buffer.bytes_per_slot for name, buffer
                             in buffers.items() if name != "barriers"), 147456)
        self.assertEqual(split.resources.shared_bytes, 147456 + buffers["barriers"].bytes_per_slot)


@unittest.skipIf(torch is None, "PyTorch CPU is required for the independent oracle")
class OracleTests(unittest.TestCase):
    def test_qwen_model_dimensions(self):
        torch.set_num_threads(2)
        torch.manual_seed(235)
        q = torch.randn(1, 64, 128).bfloat16()
        for context in (10000, 100000, 200000):
            keys = [torch.randn(4, context + 1, 128).bfloat16()]
            values = [torch.randn_like(keys[0])]
            dense = dense_gqa(q, keys, values)
            for dtype, rtol, atol in ((None, 2e-5, 1e-5), (torch.bfloat16, 0.025, 0.004)):
                actual, _, _ = split_gqa(q, keys, values, block_heads=16, splits=32,
                                         tile=128, probability_dtype=dtype)
                torch.testing.assert_close(actual, dense, rtol=rtol, atol=atol)

    def test_online_and_dense_on_split_and_tile_boundaries(self):
        from itertools import product

        torch.set_num_threads(2)
        torch.manual_seed(123)
        q = torch.randn(2, 32, 24).bfloat16()
        worst = {None: (0.0, 0.0), torch.bfloat16: (0.0, 0.0)}
        for length in (1, 17, 127, 128, 129, 255, 256, 257, 10000, 10001, 100000, 200000):
            lengths = (length, length + 3)
            keys = [torch.randn(2, n, 24).bfloat16() for n in lengths]
            values = [torch.randn_like(k) for k in keys]
            dense = dense_gqa(q, keys, values)
            for splits, tile, probability_dtype in product(
                (1, 4, 16, 32), (64, 128), (None, torch.bfloat16)
            ):
                if splits > length:
                    continue
                actual, _, lse = split_gqa(q, keys, values, block_heads=16, splits=splits,
                                          tile=tile, probability_dtype=probability_dtype)
                error = (actual - dense).abs()
                relative = (error / dense.abs().clamp_min(1e-3)).max().item()
                worst[probability_dtype] = tuple(map(max, zip(
                    worst[probability_dtype], (error.max().item(), relative)
                )))
                self.assertTrue(torch.isfinite(lse).all())
                tolerance = (2e-5, 1e-5) if probability_dtype is None else (0.025, 0.004)
                torch.testing.assert_close(actual, dense, rtol=tolerance[0], atol=tolerance[1])
        for dtype, (absolute, relative) in worst.items():
            print(f"Probability dtype {dtype or 'FP32'}: max abs {absolute:.6g}, "
                  f"max relative (denominator floor 1e-3) {relative:.6g}")
