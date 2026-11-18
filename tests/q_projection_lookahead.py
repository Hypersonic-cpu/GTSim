"""Run a small real GTSim graph to verify intrakernel lookahead and Split-K."""

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "frontend" / path) for path in ("src", "tools")]
from run_q_projection import QSchedule, compile_case, simulate, write_ir


def verify(output, runner):
    for ahead in (0, 2):
        args = SimpleNamespace(out=output / f"lookahead-{ahead}", runner=runner,
                               hardware="4f1m", fingerprint=None)
        args.out.mkdir(parents=True, exist_ok=True)
        schedule = QSchedule(256, 32, 2, "n_grouped_4", 1024, 32, ahead)
        compiled = compile_case(schedule, args.hardware, diagnostic={"K": 128})
        write_ir(args.out, compiled[2], compiled[3])
        result = simulate(args, schedule, compiled)
        timing = result["timing"]
        gemm, reduction = timing["per_kernel"]
        assert gemm["complete_cycle"] <= reduction["launch_cycle"]
        assert reduction["device_cycles"] > 0
        assert reduction["tma_load_bytes"] == 2 * 8192 * 4
        assert reduction["tma_store_bytes"] == 8192 * 2
        counters = gemm["memory"]["HBF"]["counters"]
        regions = gemm["prefetch_regions"]
        if ahead == 0:
            assert not regions
            assert counters["prefetch_queued"] == 0
            continue
        assert len(regions) == 4 * 8
        assert all(r["enqueue_cycle"] == gemm["launch_cycle"] for r in regions[:ahead])
        for index, region in enumerate(regions):
            if index >= ahead:
                assert region["enqueue_cycle"] == regions[index - ahead]["first_use_cycle"]
            assert region["first_use_cycle"] >= gemm["start_cycle"]
            assert region["last_consumer_complete_cycle"] >= region["first_use_cycle"]
            if region["ready_at_first_use"]:
                assert region["ready_cycle"] <= region["first_use_cycle"]
        # Overlapping logical rectangles can submit at most one prefetch per
        # physical page. At least one observed early demand uses the NAND path.
        assert 0 < counters["prefetch_bytes"] <= 128 * 8192 * 2
        assert counters["fallback_read_bytes"] + gemm["memory"]["HBF"][
            "unadmitted_alias_read_bytes"] > 0
        assert counters["resident_read_bytes"] > 0
        assert gemm["memory"]["HBF"]["gauges"]["sram_reserved_bytes"] == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runner = (args.build_dir / "gtsim_program").resolve()
    if args.output is not None:
        verify(args.output, runner)
    else:
        with TemporaryDirectory() as folder:
            verify(Path(folder), runner)


if __name__ == "__main__":
    main()
