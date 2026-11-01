import math
import unittest

from gtsim_frontend.ir.common import Tensor
from gtsim_frontend.ir.decode import (
    DecodeAlgorithm,
    GemmWork,
    KernelFamily,
    Qwen3RoutingToken,
    TensorUse,
    AccessMode,
)
from gtsim_frontend.passes.instantiate import instantiate
from gtsim_frontend.passes.lower_execution import lower as lower_execution
from gtsim_frontend.passes.lower_hopper_fa3 import lower as lower_fa3
from gtsim_frontend.passes.lower_hopper_gemm import lower as lower_gemm
from gtsim_frontend.programs.qwen3_decode import build


def _attention(q, keys, values):
    scores = [sum(a * b for a, b in zip(q, key)) / math.sqrt(len(q)) for key in keys]
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    normalizer = sum(weights)
    result = [
        sum(weight * value[d] for weight, value in zip(weights, values)) / normalizer
        for d in range(len(q))
    ]
    return result, maximum + math.log(normalizer)


def _split_attention(q, keys, values, chunks):
    partials = [
        _attention(q, keys[start:end], values[start:end]) for start, end in chunks
    ]
    maximum = max(lse for _, lse in partials)
    weights = [math.exp(lse - maximum) for _, lse in partials]
    normalizer = sum(weights)
    output = [
        sum(
            weight * partial[dimension]
            for weight, (partial, _) in zip(weights, partials)
        )
        / normalizer
        for dimension in range(len(q))
    ]
    return output


class Qwen3DecodeContracts(unittest.TestCase):
    def test_all_six_shapes_and_dispatches(self):
        for batch in (1, 8, 16):
            lengths = tuple(10000 + index * 1300 for index in range(batch))
            for layers in (1, 2):
                program, summary = build(batch, layers, lengths)
                self.assertEqual(program.batch_size, batch)
                self.assertEqual(program.layers, layers)
                self.assertEqual(len(program.launches), 15 * layers)
                self.assertEqual(program.context_lengths, lengths)
                for layer_summary in summary["layer_summaries"]:
                    self.assertEqual(layer_summary["new_tokens"], batch)
                    self.assertEqual(layer_summary["layer_output_shape"], [batch, 2048])
                    self.assertEqual(
                        layer_summary["fa3_variant"],
                        "split_kv" if batch == 1 else "non_split",
                    )
                    self.assertEqual(
                        layer_summary["moe_variant"],
                        "direct" if batch == 1 else "grouped",
                    )

    def test_heterogeneous_kv_regions_include_appended_token(self):
        contexts = (10000, 11001, 17003, 22007, 27011, 30013, 31517, 32000)
        program, _ = build(8, 1, contexts)
        attention = next(
            launch.algorithm.work
            for launch in program.launches
            if launch.name == "fa3_decode"
        )
        for request, context in zip(attention.requests, contexts):
            expected = context + 1
            self.assertEqual(attention.context_lengths[request.request], context)
            self.assertEqual(
                program.tensors[
                    next(
                        i
                        for i, tensor in enumerate(program.tensors)
                        if tensor.id.endswith(f"kv.key.r{request.request}.h0")
                    )
                ].shape[0],
                expected,
            )
            ranges = sorted(
                (chunk.start, chunk.end)
                for chunk in attention.chunks
                if chunk.request == request.request
            )
            self.assertEqual(ranges, [(0, expected)])

    def test_small_m_gemm_masks_rows_without_extra_outputs(self):
        for rows in (1, 8, 16):
            algorithm = DecodeAlgorithm(
                symbol=f"skinny_m{rows}",
                family=KernelFamily.SKINNY_GEMM,
                variant="skinny_m16_mma_sync",
                tensors=(
                    TensorUse(
                        "a",
                        Tensor("a", (rows, 24), "bf16", read_only=True),
                        AccessMode.READ,
                    ),
                    TensorUse(
                        "w",
                        Tensor("w", (24, 130), "bf16", read_only=True),
                        AccessMode.READ,
                    ),
                    TensorUse("c", Tensor("c", (rows, 130), "bf16"), AccessMode.WRITE),
                ),
                work=GemmWork("a", "w", "c", rows, 130, 24),
            )
            flow = instantiate(algorithm, 132)
            graph = lower_execution(lower_gemm(flow))
            stores = [op for op in graph.ops if op.op == "cp.sram2dram_tma"]
            self.assertEqual(sum(op.tile[2] * op.tile[3] for op in stores), rows * 130)
            self.assertEqual({op.tile[2] for op in stores}, {rows})
            self.assertTrue(
                all(op.tile[3] == 8 for op in graph.ops if op.op == "mma.sync.m16n8k16")
            )
            input_load_rows = {
                op.tile[2]
                for op in graph.ops
                if op.op == "cp.dram2sram_tma" and op.tensor == 0
            }
            self.assertEqual(input_load_rows, {rows})

    def test_decode_attention_uses_shared_fa3_lowering_path(self):
        program, _ = build(1, 1, (10000,))
        invocation = next(
            launch for launch in program.launches if launch.name == "fa3_decode"
        )
        flow = instantiate(invocation.algorithm, 132)
        native = lower_fa3(flow)
        graph = lower_execution(native)
        self.assertEqual(graph.symbol, invocation.algorithm.symbol)
        self.assertEqual(
            len(graph.ctas), 1 * 32 * invocation.algorithm.work.split_count
        )
        self.assertIn("cp.dram2sram_tma", {op.op for op in graph.ops})
        self.assertIn("simd.fma", {op.op for op in graph.ops})

    def test_grouped_assignment_zero_one_and_multiple_rows(self):
        routes = []
        for token in range(8):
            base = 0 if token < 2 else 8 + (token - 2) * 8
            ids = tuple(range(base, base + 8))
            routes.append(Qwen3RoutingToken(token, ids, (1.0 / 8,) * 8))
        program, _ = build(8, 1, routing_trace=(tuple(routes),))
        assignment = next(
            launch.algorithm.work
            for launch in program.launches
            if launch.name == "expert_assignment"
        )
        counts = {group.expert: len(group.assignments) for group in assignment.groups}
        self.assertEqual(
            counts,
            {
                **{expert: 2 for expert in range(8)},
                **{expert: 1 for expert in range(8, 56)},
            },
        )
        self.assertNotIn(100, counts)
        self.assertEqual(sum(counts.values()), 64)
        padded = sum(((count + 15) // 16) * 16 - count for count in counts.values())
        self.assertEqual(padded, 48 * 15 + 8 * 14)

    def test_split_lse_merge_matches_full_attention(self):
        query = (0.5, -1.0, 0.25, 0.75)
        keys = tuple(
            (i / 7 - 1.0, (i % 5) / 3, 0.25 - i / 11, (i % 3) / 4) for i in range(19)
        )
        values = tuple((i / 13, 1.0 - i / 17, (i % 7) / 5, -i / 19) for i in range(19))
        full, _ = _attention(query, keys, values)
        split = _split_attention(query, keys, values, ((0, 4), (4, 11), (11, 19)))
        for expected, actual in zip(full, split):
            self.assertAlmostEqual(expected, actual, places=12)


if __name__ == "__main__":
    unittest.main()
