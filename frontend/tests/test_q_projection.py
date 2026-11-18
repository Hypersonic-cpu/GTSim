"""Structural and numerical oracles for the timing-only Q projection runner."""

from dataclasses import replace
import math
from pathlib import Path
import re
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_q_projection import compile_case, region_resolver
from compile.expand_native import NativeExpander
from compile.opt.q_projection import K, N, ORDERINGS, QSchedule, task_order
from parser import ast as A


def expand(arch, region, tensors, cta_id):
    domain = region.domain
    cta_loop = replace(domain.loops[0], start=cta_id, stop=cta_id + 1)
    sample = replace(region, domain=replace(domain, loops=(cta_loop, *domain.loops[1:])))
    expander = NativeExpander(sample, region_resolver(arch, region),
                              {t.name: t for t in tensors})
    instructions = []
    expander.expand_to(lambda _: None, instructions.append)
    return instructions


class QProjectionTests(unittest.TestCase):
    def test_order_families_match_spec_and_are_bijective(self):
        expected = {
            "n_major": [(0, 0), (1, 0), (0, 1), (1, 1)],
            "split_major": [(0, 0), (0, 1), (0, 2), (0, 3)],
            "n_grouped_2": [(0, 0), (0, 1), (1, 0), (1, 1)],
            "n_grouped_4": [(0, 0), (0, 1), (0, 2), (0, 3)],
            "n_grouped_8": [(0, 0), (0, 1), (0, 2), (0, 3)],
            "staggered": [(0, 0), (1, 0), (1, 1), (0, 1)],
        }
        for name in ORDERINGS:
            self.assertEqual(task_order(9, 2, name)[:4], tuple(expected[name]))
            self.assertEqual(task_order(64, 1, name), task_order(64, 1, "n_major"))
        for n in (9, 32, 64):
            for split in (1, 2, 4, 8):
                orders = {task_order(n, split, name) for name in ORDERINGS}
                self.assertEqual(len(orders), 1 if split == 1 else 6)
                for tasks in orders:
                    self.assertEqual(sorted(tasks), [(s, t) for s in range(split)
                                                     for t in range(n)])

    def test_native_pipeline_guards_masked_io_and_tail_waves(self):
        for count in (1, 3, 4):
            for cta in (0, 131):
                with self.subTest(k_iterations=count, cta=cta):
                    self.check_cta(count, cta)

    def check_cta(self, count, cta):
        schedule = QSchedule(128, 32, 8, "staggered")
        _, _, _, arch, tensors, _, summary = compile_case(
            schedule, "4f1m", diagnostic={"K": count * 32 * schedule.split_k})
        instructions = expand(arch, arch.regions[0], tensors, cta)
        by_wave = {}
        by_name = {t.name: t.id for t in tensors}
        for node in instructions:
            wave = int(re.search(r"_wave(\d+)_", node.name)[1])
            by_wave.setdefault(wave, []).append(node)
        wanted_waves = list(range(4 if cta == 0 else 3))
        self.assertEqual(list(by_wave), wanted_waves)
        previous_loads = previous_stores = set()
        stage_users = [set(), set()]
        by_id = {node.id: node for node in instructions}
        def ancestors(node, goals):
            seen, pending = set(), list(node.after_complete)
            while pending and not goals <= seen:
                current = pending.pop()
                if current not in seen:
                    seen.add(current)
                    pending.extend(by_id[current].after_complete)
            return seen

        for wave, nodes in by_wave.items():
            s, n = schedule.tasks[wave * summary["persistent_ctas"] + cta]
            accumulators = {}
            for k in range(count):
                group = [node for node in nodes if node.pipeline_iteration == k]
                copies = [node for node in group if node.opcode == A.NativeOpcode.TMA_LOAD.value]
                self.assertEqual([node.tile for node in copies], [
                    (0, (s * count + k) * 32, 1, 32),
                    ((s * count + k) * 32, n * 128, 32, 64),
                    ((s * count + k) * 32, n * 128 + 64, 32, 64),
                ])
                self.assertEqual([node.tensor for node in copies],
                                 [by_name["X_norm"], by_name["W_q"], by_name["W_q"]])
                for copy in copies:
                    self.assertEqual(copy.warp, 8)
                    needed = previous_loads | stage_users[k % 2]
                    self.assertTrue(needed <= ancestors(copy, needed))
                previous_loads = {node.id for node in copies}
                mma = [node for node in group if node.opcode == A.NativeOpcode.MATRIX.value]
                self.assertEqual(len(mma), 8 * 2)
                self.assertEqual({node.warp for node in mma}, set(range(8)))
                for node in mma:
                    needed = previous_loads | (previous_stores if k == 0 else set())
                    if node.warp in accumulators:
                        needed = needed | {accumulators[node.warp]}
                    self.assertTrue(needed <= set(node.after_complete))
                    self.assertEqual(node.dtype, A.DataType.FP32)
                    accumulators[node.warp] = node.id
                stage_users[k % 2] = {node.id for node in mma}
            stores = [node for node in nodes if node.opcode == A.NativeOpcode.TMA_STORE.value]
            self.assertEqual([node.tile for node in stores],
                             [(s, n * 128, 1, 64), (s, n * 128 + 64, 1, 64)])
            self.assertTrue(all(node.dtype == A.DataType.FP32 for node in stores))
            self.assertEqual(stores[1].after_issue, (stores[0].id,))
            previous_stores = {node.id for node in stores}
        reduction = expand(arch, arch.regions[1], tensors, 0)
        self.assertEqual([node.opcode for node in reduction], [
            "cp.dram2sram_tma", "reduce_sum", "simd.cast", "st.reg2sram", "cp.sram2dram_tma"])
        self.assertEqual(reduction[0].tile, (0, 0, 8, 64))
        self.assertEqual(reduction[-1].tile, (0, 0, 1, 64))
        self.assertEqual(reduction[0].dtype, A.DataType.FP32)
        self.assertEqual(reduction[-1].dtype, A.DataType.BF16)
        for a, b in zip(reduction, reduction[1:]):
            self.assertIn(a.id, b.after_complete)

    def test_prefetch_overlap_and_independent_boundaries(self):
        schedule = QSchedule(256, 128, 2, "n_grouped_4", 53, 117, 2)
        _, _, _, arch, tensors, plan, summary = compile_case(schedule, "4f1m")
        self.assertGreater(summary["prefetch"]["shared_pages"], 0)
        self.assertEqual(sum(length for _, length in plan.spans), N * K * 2)
        load = next(node for node in expand(arch, arch.regions[0], tensors, 0)
                    if node.opcode == A.NativeOpcode.TMA_LOAD.value and node.tensor == plan.tensor)
        consumer = plan.consumer(load)
        self.assertEqual(len(consumer["regions"]), 4)
        for i in consumer["regions"]:
            self.assertTrue(set(consumer["spans"]).intersection(plan.regions[i]))
        self.assertIsNone(compile_case(replace(schedule, ahead=0), "4f1m")[5])

    def test_full_shape_numerical_oracle_all_splits_and_orderings(self):
        # BF16 inputs are exactly representable dyadic values with sparse X.
        # Include split boundaries to make missing/duplicated K partitions visible.
        rows = sorted(set(range(0, K, 233)) | {K - 1} |
                      {edge + delta for edge in range(512, K, 512) for delta in (-1, 0)})
        x = {row: bf16(((row * 17) % 71 - 34) / 16) for row in rows}
        products = {row: tuple(x[row] * bf16(((col * 73 + row * 37 + (col ^ row)) % 211
                                              - 101) / 64) for col in range(N))
                    for row in rows}
        reference = [bf16(math.fsum(products[row][col] for row in rows)) for col in range(N)]
        for split in (1, 2, 4, 8):
            width = K // split
            partial = []
            for s in range(split):
                terms = [products[row] for row in rows if s * width <= row < (s + 1) * width]
                partial.append([sum(values) for values in zip(*terms)])
            for ordering in ORDERINGS:
                schedule = QSchedule(256, 64, split, ordering)
                for ahead in (0, 2):
                    compiled = compile_case(replace(schedule, ahead=ahead, pf_bn=1024, pf_bk=1024),
                                            "4f1m")
                    arch = compiled[3]
                    resolver = region_resolver(arch, arch.regions[0])
                    outputs = [[None] * N for _ in range(split)]
                    ctas = resolver.value("%ctas")
                    for rank in range(len(schedule.tasks)):
                        point = resolver.at(dict(cta=rank % ctas, wave=rank // ctas, k=0))
                        s, col = point.value("%s"), point.value("%column")
                        self.assertIsNone(outputs[s][col])
                        self.assertEqual(point.value("%k_row"), s * width)
                        outputs[s][col:col + 256] = partial[s][col:col + 256]
                    output = [bf16(sum(values)) for values in zip(*outputs)]
                    # One BF16 rounding unit of relative error, with an
                    # absolute floor for near-zero results. Dyadic sums here
                    # are exact in FP32, so the stricter equality also holds.
                    for actual, expected in zip(output, reference):
                        self.assertLessEqual(abs(actual - expected), max(1e-5, abs(expected) / 128))
                    self.assertEqual(output, reference)
                    self.assertEqual("Partial" in {t.name for t in compiled[4]}, split > 1)


def bf16(value):
    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (raw + 0x7fff + ((raw >> 16) & 1)) & 0xffff0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]
