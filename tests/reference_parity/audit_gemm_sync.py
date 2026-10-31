#!/usr/bin/env python3
"""Check required buffer-consumer completion paths in reference graph dumps.

This is a graph-contract audit, not proof of a particular runtime race. Scheduler
and implicit collective constraints need separate review before a correctness fix.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def audit(executable, reduction):
    raw = subprocess.check_output(
        [str(executable), "gemm", "512", "512", str(reduction), "2"], text=True
    )
    nodes = {}
    for line in raw.splitlines():
        fields = line.split()
        if fields[0] != "op":
            continue
        c, i = fields.index("complete"), fields.index("issue")
        nodes[int(fields[1])] = {
            "name": fields[2],
            "complete": tuple(map(int, fields[c + 1 : i])),
            "issue": tuple(map(int, fields[i + 1 :])),
        }
    groups = {}
    for nid, n in nodes.items():
        m = re.match(r"TB(\d+)_Tile(\d+)_WGMMA_w\d+_k(\d+)_", n["name"])
        if m:
            groups.setdefault(tuple(map(int, m.groups())), []).append(nid)
    hazards = []
    for nid, n in nodes.items():
        m = re.match(r"TB(\d+)_Tile(\d+)_TMA_Load[AB]_k(\d+)", n["name"])
        if not m:
            continue
        tb, tile, k = map(int, m.groups())
        if k >= 2:
            consumer_key = (tb, tile, k - 2)
        elif tile >= 2:
            previous_k = max(
                (j for j in range(reduction // 64) if j % 2 == k % 2), default=-1
            )
            if previous_k < 0:
                continue
            consumer_key = (tb, tile - 2, previous_k)
        else:
            continue
        todo = [(nid, "issue")]
        seen = set()
        while todo:
            event = todo.pop()
            if event in seen:
                continue
            seen.add(event)
            node, phase = event
            if phase == "complete":
                todo.append((node, "issue"))
            else:
                todo.extend((d, "complete") for d in nodes[node]["complete"])
                todo.extend((d, "issue") for d in nodes[node]["issue"])
        missing = [
            nodes[c]["name"]
            for c in groups[consumer_key]
            if (c, "complete") not in seen
        ]
        if missing:
            hazards.append(
                {"load": n["name"], "slot": k % 2, "missing_consumers": missing}
            )
    return {
        "K": reduction,
        "sms": 2,
        "shape": [512, 512, reduction],
        "graph_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "nodes": len(nodes),
        "unguarded_loads": len(hazards),
        "hazards": hazards,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-safe", action="store_true")
    args = parser.parse_args()
    report = {
        "interpretation": "Required graph completion paths; not a measured runtime race",
        "cases": [audit(args.oracle, k) for k in (64, 128, 192, 256)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for c in report["cases"]:
        print(
            f"K={c['K']}: {c['unguarded_loads']} loads without all required consumer-completion paths"
        )
    if args.require_safe and any(c["unguarded_loads"] for c in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
