"""Small independent CPU math/dataflow oracle; GTSim does not execute values."""

import argparse, json, math, random
from pathlib import Path
from parser import parse_file


def softmax(scores):
    m = max(scores)
    p = [math.exp(v - m) for v in scores]
    s = sum(p)
    return [v / s for v in p]


def attention(q, keys, values):
    p = softmax([sum(a * b for a, b in zip(q, k)) / math.sqrt(len(q)) for k in keys])
    return [sum(weight * v[j] for weight, v in zip(p, values)) for j in range(len(q))]


def online(q, keys, values, tile):
    m = -math.inf
    l = 0.0
    o = [0.0] * len(q)
    for start in range(0, len(keys), tile):
        scores = [
            sum(a * b for a, b in zip(q, k)) / math.sqrt(len(q))
            for k in keys[start : start + tile]
        ]
        new_m = max(m, max(scores))
        alpha = math.exp(m - new_m)
        p = [math.exp(s - new_m) for s in scores]
        o = [
            alpha * o[j]
            + sum(w * v[j] for w, v in zip(p, values[start : start + tile]))
            for j in range(len(q))
        ]
        l = alpha * l + sum(p)
        m = new_m
    return [v / l for v in o], m + math.log(l)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rng = random.Random(27)
    d = 128
    length = 513
    q = [rng.uniform(-1, 1) for _ in range(d)]
    keys = [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(length)]
    values = [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(length)]
    reference = attention(q, keys, values)
    actual, lse = online(q, keys, values, 256)
    pieces = [
        online(
            q,
            keys[length * s // 4 : length * (s + 1) // 4],
            values[length * s // 4 : length * (s + 1) // 4],
            256,
        )
        for s in range(4)
    ]
    weights = softmax([lse for _, lse in pieces])
    combined = [
        sum(w * part[j] for w, (part, _) in zip(weights, pieces)) for j in range(d)
    ]
    online_error = max(abs(x - y) for x, y in zip(reference, actual))
    split_error = max(abs(x - y) for x, y in zip(reference, combined))
    assert online_error < 1e-12 and split_error < 1e-12
    x = [rng.uniform(-2, 2) for _ in range(d)]
    gamma = [rng.uniform(0.5, 1.5) for _ in range(d)]
    norm = [
        v / math.sqrt(sum(t * t for t in x) / d + 1e-6) * g for v, g in zip(x, gamma)
    ]
    norm_unit = [v / g for v, g in zip(norm, gamma)]
    norm_error = abs(
        sum(v * v for v in norm_unit) / d
        - (sum(v * v for v in x) / d) / (sum(v * v for v in x) / d + 1e-6)
    )
    rope_errors = []
    for position in (10000, 32000):
        angles = [position * (1000000.0 ** (-2 * j / d)) for j in range(d // 2)]
        first = [
            norm[j] * math.cos(t) - norm[j + d // 2] * math.sin(t)
            for j, t in enumerate(angles)
        ]
        second = [
            norm[j + d // 2] * math.cos(t) + norm[j] * math.sin(t)
            for j, t in enumerate(angles)
        ]
        rope_errors.append(
            abs(sum(v * v for v in norm) - sum(v * v for v in first + second))
        )
    assert norm_error < 1e-12 and max(rope_errors) < 1e-10
    # A heterogeneous expert assignment checks row/slot identity independently.
    ids = (3, 1, 1, 4, 3, 4)
    order = sorted(range(len(ids)), key=ids.__getitem__)
    inverse = [order.index(i) for i in range(len(ids))]
    expert_values = [[rng.uniform(-1, 1) for _ in range(4)] for _ in ids]
    probabilities = [0.3, 0.7, 0.8, 0.2, 0.6, 0.4]
    direct = [
        [
            sum(
                probabilities[t * 2 + s] * expert_values[t * 2 + s][j] for s in range(2)
            )
            for j in range(4)
        ]
        for t in range(3)
    ]
    sorted_values = [expert_values[i] for i in order]
    grouped = [
        [
            sum(
                probabilities[t * 2 + s] * sorted_values[inverse[t * 2 + s]][j]
                for s in range(2)
            )
            for j in range(4)
        ]
        for t in range(3)
    ]
    assert direct == grouped
    program = parse_file("frontend/asm/qwen3/one_block.progir")
    launches = {inv.id: inv for inv in program.launches}
    assert launches["post_attention_rmsnorm"].inputs[0] == "Attention_residual"
    assert launches["final_residual_add"].inputs == ("Attention_residual", "Weighted")
    assert launches["q_norm_rope"].attributes != launches["k_norm_rope"].attributes
    assert dict(launches["q_norm_rope"].attributes)["CONTEXTS"] == "%CONTEXTS"
    assert launches["v_extract"].callee == "extract"
    swiglu = parse_file("frontend/lib/swiglu.flowir").regions[0]
    gate, up = swiglu.ops[:2]
    assert (
        gate.region.offsets[1].expression == "0"
        and up.region.offsets[1].expression == "%WIDTH"
    )
    result = dict(
        online_max_abs_error=online_error,
        split_lse_max_abs_error=split_error,
        rmsnorm_error=norm_error,
        rope_norm_errors=rope_errors,
        grouped_weighted_mapping="PASS",
        residual_producer_consumer="PASS",
        gate_up_subregions="PASS",
        gqa_mapping=[h // 8 for h in range(32)],
        scope="independent CPU operator math and typed Program dataflow; not numerical GTSim inference",
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
