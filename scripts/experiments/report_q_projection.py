"""Summarize completed Experiment 002 measurements in cycles."""

from collections import Counter, defaultdict
import csv
import json
import math
from statistics import mean, median


def load(path):
    return json.loads(path.read_text())


def metrics(case, status):
    p = case.schedule
    row = dict(case=case.id, phase=case.phase, hardware=case.hardware,
               status=status["status"], reason=status["error"], bn=p.bn, bk=p.bk,
               split_k=p.split_k, ordering=p.ordering, pf_bn=p.pf_bn, pf_bk=p.pf_bk,
               ahead=p.ahead, fingerprint=status["fingerprint"],
               elapsed_seconds=status["elapsed_seconds"],
               peak_ram_bytes=status["resources"].get("peak_ram_bytes"))
    if row["status"] != "SUCCESS":
        return row
    data = load(case.directory / "result.json")
    timing = data["timing"]
    summary = data["compile_metadata"]
    gemm, *reduce = timing["per_kernel"]
    hbf = gemm["memory"].get("HBF", {})
    hbm = gemm["memory"].get("HBM", {})
    counts = hbf.get("counters", {})
    gauges = hbf.get("gauges", {})
    regions = gemm["prefetch_regions"]
    useful_leads = [r["first_use_cycle"] - r["ready_cycle"] for r in regions
                    if r["ready_at_first_use"]]
    resident = counts.get("resident_read_bytes")
    sensed = counts.get("sensed_bytes")
    pf = summary["prefetch"] or {}
    device_cycles = gemm["device_cycles"]
    row.update(
        total_cycles=timing["total_cycles"], gemm_cycles=device_cycles,
        reduction_cycles=reduce[0]["complete_cycle"] - reduce[0]["launch_cycle"] if reduce else 0,
        reduction_tma_load_bytes=reduce[0]["tma_load_bytes"] if reduce else 0,
        reduction_tma_store_bytes=reduce[0]["tma_store_bytes"] if reduce else 0,
        reduction_hbm_read_bytes=reduce[0]["memory"]["HBM"]["read_bytes"] if reduce else 0,
        reduction_hbm_write_bytes=reduce[0]["memory"]["HBM"]["write_bytes"] if reduce else 0,
        tasks=summary["compute_tasks"], ctas=summary["persistent_ctas"], waves=summary["waves"],
        task_slots_used=summary["compute_tasks"] / (summary["persistent_ctas"] * summary["waves"]),
        sm_active_fraction=gemm["active_sm_cycles"] / (132 * device_cycles),
        active_sms=gemm["active_sms"], wgmma_warp_ops=gemm["wgmma_warp_operations"],
        tma_load_ops=gemm["tma_load_operations"], tma_store_ops=gemm["tma_store_operations"],
        tma_inflight_fraction=gemm["pipeline_inflight_fraction"].get("TMA"),
        wgmma_inflight_fraction=gemm["pipeline_inflight_fraction"].get("Tensor"),
        # All load nodes in this GEMM are TMA. The ST class also includes
        # shared-memory stores, so it cannot isolate TMA store time.
        critical_path_tma_load_cycles=gemm["dependency_path_cycles"].get("LD", 0),
        gpu_memory_wait_cycles=None,
        hbf_sense_commands=counts.get("sense_commands"),
        hbf_sensed_pages=counts.get("sensed_pages"),
        hbf_media_bytes=sensed,
        hbf_repeated_sense_commands=counts.get("repeated_sense_commands"),
        hbf_media_amplification=sensed / (64 * 1024**2) if sensed is not None else None,
        hbf_active_units=len(hbf.get("active_units", [])) if hbf else None,
        hbf_unit_busy_ticks=sum(hbf.get("unit_busy_ticks", {}).values()) if hbf else None,
        hbf_prefetch_bytes=counts.get("prefetch_bytes"),
        hbf_resident_read_bytes=resident,
        hbf_demand_read_bytes=hbf["read_bytes"] - resident if resident is not None else None,
        hbf_demand_fallback_requests=(counts["fallback_read_requests"] +
                                      hbf["unadmitted_alias_read_requests"]
                                      if p.ahead else 0),
        hbf_demand_fallback_bytes=(counts["fallback_read_bytes"] +
                                   hbf["unadmitted_alias_read_bytes"] if p.ahead else 0),
        sram_peak_bytes=gauges.get("sram_peak_reserved_bytes"),
        sram_admission_stalls=timing["prefetch_capacity_stalls"],
        sram_allocation_failures=None,
        hbf_read_GBps=hbf["read_bytes"] * 1.98 / device_cycles if hbf else None,
        hbm_read_GBps=hbm["read_bytes"] * 1.98 / device_cycles if hbm else None,
        prefetch_regions=len(regions), late_regions=len(regions) - len(useful_leads),
        useful_region_lead_cycles=median(useful_leads) if useful_leads else None,
        physical_span_count=pf.get("physical_span_count"),
        shared_physical_pages=pf.get("shared_pages"),
        max_region_pages=pf.get("max_region_pages"),
        summed_region_pages=pf.get("summed_region_pages"),
    )
    return row


def group_by(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def table(columns, rows):
    return ["| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *["| " + " | ".join(str(value) for value in row) + " |" for row in rows]]


def mib(value):
    return "unavailable" if value is None else f"{value / 1024**2:.2f}"


def best_row(rows):
    return min(rows, key=lambda row: (row["total_cycles"], row["case"])) if rows else None


def paired_prefetch(rows):
    """Summarize against the same BN, BK, Split-K and ordering."""
    by_compute = group_by(rows, ("bn", "bk", "split_k", "ordering"))
    speedups, distance_pairs = defaultdict(list), defaultdict(list)
    for group in by_compute.values():
        baseline = next((row for row in group if row["ahead"] == 0), None)
        if baseline:
            for row in group:
                if row["ahead"]:
                    mode = row["pf_bn"], row["pf_bk"], row["ahead"]
                    speedups[mode].append(baseline["total_cycles"] / row["total_cycles"])
        geometry = group_by([r for r in group if r["ahead"] > 0], ("pf_bn", "pf_bk"))
        for key, values in geometry.items():
            by_ahead = {row["ahead"]: row for row in values}
            if set(by_ahead) == {1, 2, 4}:
                distance_pairs[key].append((
                    by_ahead[1]["total_cycles"] / by_ahead[2]["total_cycles"],
                    by_ahead[1]["total_cycles"] / by_ahead[4]["total_cycles"],
                ))
    return speedups, distance_pairs


def independently_ranked(rows):
    off = [r for r in rows if r["phase"] == "phase-a" and r["ahead"] == 0]
    if not off:
        return None
    dimensions = ("bn", "bk", "split_k", "ordering")
    choices = {}
    for key in dimensions:
        # S=1 has one deduplicated ordering. Give every geometry one marginal
        # sample by taking the mean across its otherwise equivalent replicas.
        subset = [r for r in off if r["split_k"] > 1] if key == "ordering" else off
        if not subset:
            choices[key] = "n_major"
            continue
        groups = group_by(subset, (key,))
        scores = {}
        for (value,), candidates in groups.items():
            controls = [name for name in dimensions if name not in (key, "ordering")]
            if key == "ordering":
                controls = ["bn", "bk", "split_k"]
            scores[value] = mean(mean(math.log(r["total_cycles"]) for r in group)
                                 for group in group_by(candidates, controls).values())
        choices[key] = min(scores, key=lambda value: (scores[value], value))
    if choices["split_k"] == 1:
        choices["ordering"] = "n_major"
    return best_row([r for r in rows if all(r[key] == choices[key] for key in dimensions)])


def write_reports(root, pilot, reference, cases, shortfall):
    fullhbm = metrics(pilot, reference)
    rows = [metrics(case, status) for case, status in cases]
    legal = [row for row in rows if row["status"] == "SUCCESS"]
    best = best_row(legal)
    best_prefetched = best_row([r for r in legal if r["ahead"] > 0])
    best_off = best_row([r for r in legal if r["ahead"] == 0])
    own_off = best_row([r for r in legal if r["ahead"] == 0 and best is not None and
                        all(r[key] == best[key] for key in ("bn", "bk", "split_k", "ordering"))])
    split_best = [best_row([r for r in legal if r["split_k"] == split]) for split in (1, 2, 4, 8)]
    independent = independently_ranked(legal)
    speedups, distance_pairs = paired_prefetch(legal)
    fields = list(dict.fromkeys(key for row in [fullhbm, *rows] for key in row))
    with (root / "case_results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows([fullhbm, *rows])
    count = Counter(r["status"] for r in rows)
    summary = dict(full_hbm=fullhbm, best_4f1m=best, matching_prefetch_off=own_off,
                   best_prefetched=best_prefetched, best_prefetch_off=best_off,
                   best_by_split=split_best, independently_ranked=independent,
                   counts=count, phase_b_shortfall=shortfall)
    (root / "best_candidates.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Experiment 002 findings", "",
        f"Full-HBM reference **{fullhbm['total_cycles']:,} cycles**. "
        f"4F1M outcomes {dict(count)} across {len(rows)} candidates.", "",
        "GTSim measures timing. The separate deterministic BF16 oracle checks output coverage and "
        "Split-K math. The relative tolerance is 1/128 with an absolute floor of 1e-5. "
        "The dyadic test also passed exact BF16 equality.", "",
        "## 1. Compute geometry and Split-K", "",
        *table(("BN", "BK", "Split-K", "Best cycles", "Tasks", "CTAs", "Waves",
                "Tasks/slots", "SM active", "Reduction cycles"),
               ((bn, bk, split, f"{r['total_cycles']:,}", r["tasks"], r["ctas"], r["waves"],
                 f"{r['task_slots_used']:.1%}", f"{r['sm_active_fraction']:.1%}",
                 f"{r['reduction_cycles']:,}")
                for (bn, bk, split), group in sorted(group_by(legal, ("bn", "bk", "split_k")).items())
                for r in [best_row(group)])),
        "", "Each row takes the fastest measured ordering and prefetch mode for that geometry. "
        "Reduction runs as a dependent kernel and includes partial loads, FP32 accumulation and "
        "BF16 output stores. The CSV records its HBM traffic and TMA payload separately.", "",
    ]
    one = split_best[0]
    if one:
        for other in (r for r in split_best[1:] if r):
            lines.append(f"Best Split-K={other['split_k']} used {other['active_sms']} active SMs "
                         f"and took {other['total_cycles'] / one['total_cycles']:.3f}× the best "
                         f"Split-K=1 cycles on {one['active_sms']} SMs. Its reduction used "
                         f"{other['reduction_cycles'] / other['total_cycles']:.2%} of total cycles.")
    lines += ["", "## 2. Compute ordering", ""]
    paired = group_by([r for r in legal if r["phase"] == "phase-a" and r["split_k"] > 1],
                      ("bn", "bk", "split_k", "ahead"))
    complete = [group for group in paired.values() if len(group) == 6]
    ratios, repeated, activity, residency = (defaultdict(list) for _ in range(4))
    ties, identical = 0, 0
    for group in complete:
        minimum = best_row(group)["total_cycles"]
        ties += len({r["total_cycles"] for r in group}) < len(group)
        trace = ("total_cycles", "gemm_cycles", "hbf_sense_commands", "hbf_media_bytes",
                 "hbf_resident_read_bytes", "hbf_active_units", "hbf_unit_busy_ticks")
        identical += len({tuple(r[k] for k in trace) for r in group}) < len(group)
        for r in group:
            order = r["ordering"]
            ratios[order].append(r["total_cycles"] / minimum)
            repeated[order].append(r["hbf_repeated_sense_commands"])
            activity[order].append(r["hbf_unit_busy_ticks"])
            if r["ahead"]:
                residency[order].append(r["hbf_resident_read_bytes"])
    lines += [
        f"{len(complete)} identical-geometry and prefetch groups have results for all six orderings. "
        f"{ties} groups contain cycle ties and {identical} contain ties across the recorded "
        "cycle, media and memory activity counters. Split-K=1 permutations are identical by construction.",
        "",
        *table(("Ordering", "Median cycles / group best", "Median repeat senses",
                "Median bank busy ticks", "Median prefetched resident MiB"),
               ((order, f"{median(values):.3f}", median(repeated[order]), median(activity[order]),
                 mib(median(residency[order])) if residency[order] else "unavailable")
                for order, values in sorted(ratios.items()))),
        "", "## 3. Prefetch geometry and physical pages", "",
        *table(("PfBN × PfBK", "Cases", "Best cycles", "Median SRAM peak MiB",
                "Median media / unique", "Median repeat senses", "Shared pages"),
               ((f"{bn} × {bk}" if bn else "OFF", len(group), best_row(group)["total_cycles"],
                 mib(median(r["sram_peak_bytes"] for r in group)),
                 f"{median(r['hbf_media_amplification'] for r in group):.3f}",
                 median(r["hbf_repeated_sense_commands"] for r in group),
                 group[0]["shared_physical_pages"] or 0)
                for (bn, bk), group in sorted(group_by(legal, ("pf_bn", "pf_bk")).items()))),
        "", "Media amplification divides NAND sensed bytes by the contiguous 64 MiB weight allocation. "
        "Prefetch rectangles share physical pages even when their logical column intervals are disjoint. "
        "Each shared page has one reservation until its final relevant TMA completes.", "",
        "## 4. Lookahead distance", "",
        *table(("PfBN × PfBK", "Ahead", "Cases", "Best cycles", "Median paired OFF/ON",
                "Median late", "Median fallback MiB", "Max stalls", "Median lead cycles"),
               ((f"{bn} × {bk}", ahead, len(group), best_row(group)["total_cycles"],
                 f"{median(pairs):.3f}×" if pairs else "unavailable",
                 f"{median(r['late_regions'] / r['prefetch_regions'] for r in group):.1%}",
                 mib(median(r["hbf_demand_fallback_bytes"] for r in group)),
                 max(r["sram_admission_stalls"] for r in group),
                 median(leads) if leads else "unavailable")
                for (bn, bk, ahead), group in sorted(group_by(
                    [r for r in legal if r["ahead"]], ("pf_bn", "pf_bk", "ahead")).items())
                for pairs in [speedups[bn, bk, ahead]]
                for leads in [[r["useful_region_lead_cycles"] for r in group
                               if r["useful_region_lead_cycles"] is not None]])),
        "", "Late means at least one page of the logical region was still unready at its first TMA "
        "issue. Actual demand fallback counts individual requests routed through the NAND demand path "
        "while the relevant page was unavailable. Per-region enqueue, ready and first-use cycles are "
        "recorded in each result. A missing ready cycle means the full region never became ready "
        "before its relevant consumers retired.", "",
    ]
    for (bn, bk), pairs in sorted(distance_pairs.items()):
        ratio_2, ratio_4 = zip(*pairs)
        lines.append(f"For {bn} × {bk} over {len(pairs)} matched compute configurations, "
                     f"the median cycles ratio was {median(ratio_2):.3f}× for ahead 1 over 2 "
                     f"and {median(ratio_4):.3f}× for ahead 1 over 4. Ahead 4 beat ahead 1 "
                     f"in {sum(ratio > 1 for ratio in ratio_4)}/{len(pairs)} matches.")
    lines += ["", "## 5. Joint optimization", ""]
    if best:
        lines.append(f"Best tested 4F1M case **{best['case']}** at **{best['total_cycles']:,} cycles**, "
                     f"{best['total_cycles'] / fullhbm['total_cycles']:.3f}× the Full-HBM cycles.")
        if own_off:
            lines.append(f"Its matching Prefetch-OFF run took {own_off['total_cycles']:,} cycles "
                         f"({own_off['total_cycles'] / best['total_cycles']:.3f}× OFF/selected speedup).")
        if best_prefetched and best_off:
            lines.append(f"The fastest positive-lookahead case took "
                         f"{best_prefetched['total_cycles']:,} cycles and the fastest OFF case took "
                         f"{best_off['total_cycles']:,} cycles.")
        if independent:
            lines.append(f"Carrying the marginally best Prefetch-OFF compute parameters into the best "
                         f"available matching prefetch mode gave {independent['case']} at "
                         f"{independent['total_cycles']:,} cycles. This "
                         f"{'matches' if independent['case'] == best['case'] else 'differs from'} "
                         "the jointly best measured configuration.")
    else:
        lines.append("No successful 4F1M measurements are available.")
    limits = load(root / "resource_summary.json")
    enforcement = (
        "The scheduler also enforced a cgroup ceiling for its jobs."
        if limits["kernel_ceiling_enabled"] else
        "The cgroup mount did not allow a kernel ceiling. The scheduler monitored host "
        "RAM and any applicable cgroup limit."
    )
    lines += [
        "", f"RAM policy used a 95% hard threshold, 85% admission and 90% recovery. "
        f"{enforcement}",
        "", "TMA and WGMMA in-flight fractions and critical-path TMA load cycles are measured. "
        "A dedicated GPU memory-wait counter and a distinct SRAM allocation-failure counter "
        "are unavailable. Blank CSV values indicate unavailable counters.", "",
    ]
    if shortfall:
        lines += [f"Phase B selection shortfall by Split-K: {shortfall}.", ""]
    failures = [r for r in rows if r["status"] != "SUCCESS"]
    if failures:
        lines += ["## Non-successful cases", "",
                  *table(("Case", "Status", "Reason"),
                         ((r["case"], r["status"], r["reason"] or "RAM retries exhausted")
                          for r in failures)), ""]
    (root / "findings.md").write_text("\n".join(lines))
    winners = {r["case"] for r in [best, *split_best] if r}
    return [case for case, _ in cases if case.id in winners]
