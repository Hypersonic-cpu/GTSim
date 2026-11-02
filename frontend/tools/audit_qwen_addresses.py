"""Independent request/head/expert association checks on compact Native IR."""

from bisect import bisect_right
from pathlib import Path
from compile import compile_program, lower_flow
from compile.static import Resolver, EMPTY, bound_resolver
from parser import ast as A


def audit(parameters):
    lib = Path("frontend/lib")
    arch = lower_flow(
        compile_program(
            "frontend/asm/qwen3/one_block.progir",
            parameters,
            list(lib.glob("*.algoir")),
            list(lib.glob("*.flowir")),
        )
    )
    root = bound_resolver(arch.static, Resolver(EMPTY, {})).materialize_constants()
    ids = parameters["ROUTE_IDS"]
    topk = parameters["TK"]
    experts = sorted(set(ids))
    order = sorted(range(len(ids)), key=ids.__getitem__)
    rank = {slot: i for i, slot in enumerate(order)}
    for token in range(parameters["B"]):
        route = ids[token * topk : (token + 1) * topk]
        assert len(set(route)) == topk and all(0 <= e < parameters["E"] for e in route)
        assert (
            abs(sum(parameters["ROUTE_PROBS"][token * topk : (token + 1) * topk]) - 1)
            < 1e-12
        )
    heads_checked = expert_tiles = position_args = 0
    for region in arch.regions:
        args = {v.id: root.value(v.expression) for v in region.static.arguments}
        context = Resolver(
            region.static, {**root.values, **args}
        ).materialize_constants()
        if any(
            not context.value(g.condition)
            for g in region.static_ifs
            if not g.iterations
        ):
            continue
        ctas = context.value(region.domain.loops[0].stop)
        if region.origin in ("fa3_decode", "fa3_decode_split"):
            query, key, value = region.tensors[:3]
            steps = [step for op in region.ops for step in op.steps]
            q = next(
                step
                for step in steps
                if step.opcode == A.NativeOpcode.TMA_LOAD
                and step.region.tensor == query.id
            )
            k = next(
                step
                for step in steps
                if step.opcode == A.NativeOpcode.TMA_LOAD
                and step.region.tensor == key.id
            )
            v = next(
                step
                for step in steps
                if step.opcode == A.NativeOpcode.TMA_LOAD
                and step.region.tensor == value.id
            )
            for cta in range(ctas):
                scope = context.at({"cta": cta, "wave": 0}).materialize_constants()
                request = cta // (parameters["HQ"] * parameters["S"])
                head = (cta // parameters["S"]) % parameters["HQ"]
                split = cta % parameters["S"]
                kv_head = head // (parameters["HQ"] // parameters["HKV"])
                length = parameters["CONTEXTS"][request] + 1
                start = length * split // parameters["S"]
                end = length * (split + 1) // parameters["S"]
                base = (sum(parameters["CONTEXTS"][:request]) + request) * parameters[
                    "HKV"
                ] + kv_head * length
                assert tuple(scope.value(v.expression) for v in q.region.offsets) == (
                    request * parameters["HQ"] + head,
                    0,
                )
                for iteration in range(
                    (end - start + parameters["TILE"] - 1) // parameters["TILE"]
                ):
                    point = scope.at({"k": iteration}).materialize_constants()
                    for step in (k, v):
                        assert tuple(
                            point.value(x.expression) for x in step.region.offsets
                        ) == (base + start + iteration * parameters["TILE"], 0)
                        assert tuple(point.value(x) for x in step.region.shape) == (
                            min(
                                parameters["TILE"],
                                end - start - iteration * parameters["TILE"],
                            ),
                            parameters["D"],
                        )
                heads_checked += 1
        elif region.origin == "norm_rope":
            args = [
                v
                for pattern in region.ops
                for step in pattern.steps
                for v in step.static_args
                if v.id == "position"
            ]
            assert len(args) == 1
            for cta in range(ctas):
                point = context.at({"cta": cta, "wave": 0}).materialize_constants()
                heads = point.value("%HEADS")
                assert (
                    point.value(args[0].expression)
                    == parameters["CONTEXTS"][cta // heads]
                )
                position_args += 1
        elif region.origin == "grouped_gemm":
            red = context.value("%K")
            columns = context.value("%N")
            n_tiles = (columns + 63) // 64
            inputs, weights, output = region.tensors
            steps = [step for op in region.ops for step in op.steps]
            a = next(
                s
                for s in steps
                if s.opcode == A.NativeOpcode.TMA_LOAD and s.region.tensor == inputs.id
            )
            b = next(
                s
                for s in steps
                if s.opcode == A.NativeOpcode.TMA_LOAD and s.region.tensor == weights.id
            )
            c = next(
                s
                for s in steps
                if s.opcode == A.NativeOpcode.TMA_STORE and s.region.tensor == output.id
            )
            expected = []
            for expert in experts:
                slots = [s for s in order if ids[s] == expert]
                for block in range((len(slots) + 15) // 16):
                    subset = slots[block * 16 : (block + 1) * 16]
                    if parameters["B"] > 1:
                        in_rows = out_rows = [rank[s] for s in subset]
                    else:
                        out_rows = subset
                        in_rows = (
                            [s // topk for s in subset]
                            if region.symbol.startswith("expert_gemm1")
                            else subset
                        )
                    for tile in range(n_tiles):
                        expected.append((expert, in_rows, out_rows, tile))
            assert len(expected) == ctas
            for cta, (expert, in_rows, out_rows, tile) in enumerate(expected):
                scope = context.at({"cta": cta, "wave": 0}).materialize_constants()
                for iteration in (0, red // 16 - 1):
                    point = scope.at({"k": iteration})
                    assert tuple(
                        point.value(x.expression) for x in a.region.offsets
                    ) == (in_rows[0], iteration * 16)
                    assert tuple(point.value(x) for x in a.region.shape) == (
                        len(in_rows),
                        16,
                    )
                    assert tuple(
                        point.value(x.expression) for x in b.region.offsets
                    ) == (expert * red + iteration * 16, tile * 64)
                assert tuple(scope.value(x.expression) for x in c.region.offsets) == (
                    out_rows[0],
                    tile * 64,
                )
                assert tuple(scope.value(x) for x in c.region.shape) == (
                    len(out_rows),
                    64,
                )
                expert_tiles += 1
    return dict(
        batch=parameters["B"],
        attention_head_splits_checked=heads_checked,
        rope_positions_checked=position_args,
        expert_tiles_checked=expert_tiles,
        outcome="PASS",
    )
