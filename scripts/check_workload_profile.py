#!/usr/bin/env python3
"""Reject timing/resource overrides in public workload builders."""

import sys
from pathlib import Path


FORBIDDEN = (
    "setup_latency",
    "tma_setup_latency",
    "tma_issue_interval",
    "tma_issue_rate",
    "throughput",
    "bandwidth",
    "startup_delay",
    "block_schedule",
)
ALLOWED_CONFIG_LINES = ("config.silence_mode", "config.workload_type")
ALLOWED_MEMORY_READS = (
    "const bool segmented = config.memory.address_layout.enabled ||",
    "config.memory.placement != TensorPlacementProfile::AllHBM;",
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    violations = []
    for source in sorted((root / "workloads").rglob("*.cpp")):
        for number, line in enumerate(source.read_text().splitlines(), start=1):
            stripped = line.strip()
            if any(token in stripped for token in FORBIDDEN):
                violations.append(f"{source.relative_to(root)}:{number}: {stripped}")
            if ("config." in stripped and stripped not in ALLOWED_MEMORY_READS
                    and not any(item in stripped for item in ALLOWED_CONFIG_LINES)):
                violations.append(f"{source.relative_to(root)}:{number}: {stripped}")
    if violations:
        print("Workload builders may not override hardware-profile timing:", *violations,
              sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
