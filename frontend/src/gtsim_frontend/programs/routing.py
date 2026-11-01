"""Deterministic routing input for timing workloads; no predicted execution."""

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path


@dataclass(frozen=True)
class RoutingToken:
    token: int
    expert_ids: tuple[int, ...]
    probabilities: tuple[float, ...]


def normalize(routes, layers, batch, experts=128, top_k=8):
    if len(routes) != layers:
        raise ValueError("routing trace requires one array per layer")
    result = []
    for layer in routes:
        if len(layer) != batch:
            raise ValueError("routing trace requires exactly B rows per layer")
        tokens = []
        for token, record in enumerate(layer):
            ids = tuple(
                record.expert_ids
                if isinstance(record, RoutingToken)
                else record["expert_ids"]
            )
            probabilities = tuple(
                float(p)
                for p in (
                    record.probabilities
                    if isinstance(record, RoutingToken)
                    else record["probabilities"]
                )
            )
            if (
                len(ids) != top_k
                or len(set(ids)) != top_k
                or any(type(e) is not int or not 0 <= e < experts for e in ids)
            ):
                raise ValueError("invalid TopK expert IDs")
            if (
                len(probabilities) != top_k
                or any(not math.isfinite(p) or p < 0 for p in probabilities)
                or sum(probabilities) <= 0
            ):
                raise ValueError("invalid routing probabilities")
            total = sum(probabilities)
            tokens.append(
                RoutingToken(token, ids, tuple(p / total for p in probabilities))
            )
        result.append(tuple(tokens))
    return tuple(result)


def synthetic_routing_trace(batch, layers, seed=0):
    rng = random.Random(seed)
    routes = []
    for _ in range(layers):
        tokens = []
        for token in range(batch):
            ids = tuple(rng.sample(range(128), 8))
            raw = [rng.random() + 1e-6 for _ in range(8)]
            tokens.append(RoutingToken(token, ids, tuple(p / sum(raw) for p in raw)))
        routes.append(tuple(tokens))
    return tuple(routes)


def load_routing_trace(path, layers, batch):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("layers")
    if not isinstance(data, list):
        raise ValueError("routing trace must be layer-indexed JSON")
    return normalize(data, layers, batch)
