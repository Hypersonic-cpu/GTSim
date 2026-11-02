"""Summarize four completed Qwen simulations; never polls running jobs."""

import argparse
import json
from pathlib import Path


def name(value):
    return (
        value.removeprefix("l0_")
        .removesuffix("_direct")
        .removesuffix("_grouped")
        .replace("fa3_decode_split", "fa3_decode")
    )


def duration(kernel):
    return kernel["complete_cycle"] - kernel["launch_cycle"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = []
    sections = []
    for batch in (1, 16):
        runs = {}
        provenance = {}
        for version in ("legacy", "new"):
            directory = args.root / f"{version}-b{batch}"
            result = json.loads((directory / "result.json").read_text())
            assert result["returncode"] == 0 and not result["errors"]
            timing = result["timing"]
            assert all(
                timing[k]
                for k in (
                    "persistent_gpu",
                    "persistent_memory",
                    "persistent_l2",
                    "stable_allocations",
                )
            )
            previous = 0
            for kernel in timing["per_kernel"]:
                assert kernel["state"] == "COMPLETED"
                assert previous == kernel["launch_cycle"]
                assert kernel["launch_cycle"] <= kernel["start_cycle"]
                assert kernel["start_cycle"] <= kernel["device_complete_cycle"]
                assert kernel["device_complete_cycle"] <= kernel["complete_cycle"]
                previous = kernel["complete_cycle"]
            assert previous == timing["total_cycles"]
            runs[version] = result
            provenance[version] = json.loads(
                (directory / "provenance.json").read_text()
            )
        for key in (
            "params",
            "runner_sha256",
            "architecture_sha256",
            "memory_sha256",
            "git_commit",
            "launch_overhead_cycles",
            "source_sha256",
        ):
            assert provenance["legacy"][key] == provenance["new"][key], key
        old, new = (runs[v]["timing"] for v in ("legacy", "new"))
        assert [name(k["id"]) for k in old["per_kernel"]] == [
            name(k["id"]) for k in new["per_kernel"]
        ]
        kernels = []
        lines = [
            f"## B={batch}",
            "",
            "| Kernel | Legacy cycles | New cycles | Change | Legacy HBM read | New HBM read |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for before, after in zip(old["per_kernel"], new["per_kernel"]):
            delta = duration(after) - duration(before)
            change = 100 * delta / duration(before)
            kernels.append(
                dict(
                    name=name(after["id"]),
                    legacy=before,
                    new=after,
                    delta_cycles=delta,
                    change_percent=change,
                )
            )
            lines.append(
                f"| {name(after['id'])} | {duration(before):,} | {duration(after):,} | {change:+.2f}% | {before['hbm_read_bytes']:,} | {after['hbm_read_bytes']:,} |"
            )
        old_cycles, new_cycles = old["total_cycles"], new["total_cycles"]
        change = 100 * (new_cycles / old_cycles - 1)
        row = dict(
            batch_size=batch,
            legacy_cycles=old_cycles,
            new_cycles=new_cycles,
            change_percent=change,
            legacy_traffic=old["traffic_bytes"],
            new_traffic=new["traffic_bytes"],
            legacy_nodes=sum(k["nodes"] for k in runs["legacy"]["kernels"]),
            new_nodes=sum(k["nodes"] for k in runs["new"]["kernels"]),
            kernels=kernels,
        )
        rows.append(row)
        lines += [
            "",
            "Global cycle intervals (launch/start/device complete/drain complete):",
            "",
            "| Kernel | Legacy | New |",
            "| --- | --- | --- |",
        ]
        for kernel in kernels:
            intervals = [
                " / ".join(
                    str(kernel[v][f])
                    for f in (
                        "launch_cycle",
                        "start_cycle",
                        "device_complete_cycle",
                        "complete_cycle",
                    )
                )
                for v in ("legacy", "new")
            ]
            lines.append(f"| {kernel['name']} | {intervals[0]} | {intervals[1]} |")
        sections.append("\n".join(lines))
    document = [
        "# Four-run simulator comparison",
        "",
        "All four runs completed on one ordered stream with persistent GPU, memory and L2. Launch overhead is zero. Latencies are simulator estimates, not calibrated H100 measurements. HBM byte counts are actual simulator traffic counters, not logical tensor bytes.",
        "",
        "| Batch | Legacy cycles | New cycles | Change | Legacy us | New us |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        document.append(
            f"| {row['batch_size']} | {row['legacy_cycles']:,} | {row['new_cycles']:,} | {row['change_percent']:+.2f}% | {row['legacy_cycles'] / 1980:.3f} | {row['new_cycles'] / 1980:.3f} |"
        )
    document += [
        "",
        "| Batch | Legacy/new HBM read bytes | Legacy/new HBM write bytes | HBF bytes |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        traffic = row["legacy_traffic"]
        assert traffic == row["new_traffic"]
        document.append(
            f"| {row['batch_size']} | {traffic['HBM_read']:,} | {traffic['HBM_write']:,} | {traffic['HBF_read'] + traffic['HBF_write']:,} |"
        )
    document += ["", "\n\n".join(sections), ""]
    (args.root / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.root / "comparison.md").write_text("\n".join(document))
    print(
        json.dumps(
            [{k: v for k, v in row.items() if k != "kernels"} for row in rows], indent=2
        )
    )


if __name__ == "__main__":
    main()
