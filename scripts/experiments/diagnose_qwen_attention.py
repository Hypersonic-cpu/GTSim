"""One-variable 100k attention sensitivity checks on the full decode block."""

import asyncio
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/results/qwen235b-attention-diagnosis"


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    architecture = (ROOT / "configs/v0.1.0/h100-sxm.yaml").read_text()
    memory = (ROOT / "configs/memconf/0F5M_channels.yaml").read_text()
    variants = {
        "baseline": (architecture, memory),
        "hbm_2x": (architecture, memory.replace("read_bw_GBps: 3100", "read_bw_GBps: 6200")),
        "hbm_half": (architecture, memory.replace("read_bw_GBps: 3100", "read_bw_GBps: 1550")),
        "l2_2x": (architecture.replace("bandwidth_bytes_per_cycle: 4300", "bandwidth_bytes_per_cycle: 8600"), memory),
        "l2_half": (architecture.replace("bandwidth_bytes_per_cycle: 4300", "bandwidth_bytes_per_cycle: 2150"), memory),
        "compute_2x": (architecture + "\nsimd_throughput: 64\nsimd_width: 64\nsimd_latency_cycles: 2\nsfu_throughput: 64\nsfu_width: 8\nsfu_latency_cycles: 15\nshfl_latency_cycles: 2\n", memory),
    }
    semaphore = asyncio.Semaphore(2)

    async def run(name, configs):
        async with semaphore:
            available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
            if available < 30 * 1024**3:
                raise RuntimeError("less than 30 GiB available; refusing diagnostic admission")
            folder = OUT / name
            folder.mkdir(exist_ok=True)
            arch_path, mem_path = folder / "architecture.yaml", folder / "memory.yaml"
            arch_path.write_text(configs[0])
            mem_path.write_text(configs[1])
            command = [sys.executable, str(ROOT / "frontend/tools/run_qwen_simulations.py"),
                       "--params", str(ROOT / "frontend/asm/qwen3/qwen3-235b-a22b-b1.json"),
                       "--source", str(ROOT / "frontend/asm/qwen3/flash-accel/one_block.progir"),
                       "--runner", str(ROOT / "build/gtsim_program"),
                       "--architecture", str(arch_path), "--memory", str(mem_path),
                       "--context-length", "100000", "--gemm-bm", "16", "--gemm-bn", "64",
                       "--gemm-bk", "256", "--gemm-stages", "2", "--sram-policy", "none",
                       "--prefetch", "off", "--max-cycles", "1000000000", "--out", str(folder)]
            with (folder / "driver.log").open("w") as log:
                process = await asyncio.create_subprocess_exec(*command, cwd=ROOT,
                    env={**os.environ, "PYTHONPATH": str(ROOT / "frontend/src")},
                    stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
                try:
                    code = await asyncio.wait_for(process.wait(), 1800)
                except BaseException:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
                    raise
            if code:
                print(name, "FAILED", code, flush=True)
                return {"case": name, "status": "FAILED"}
            timing = json.loads((folder / "result.json").read_text())["timing"]
            kernel = next(k for k in timing["per_kernel"] if k["id"] == "fa3_decode_split")
            hbm = kernel["memory"]["HBM"]
            result = {"case": name, "status": "SUCCESS", "attention_cycles": kernel["device_cycles"],
                "total_cycles": timing["total_cycles"], "hbm_read_bytes": hbm["read_bytes"],
                "hbm_read_gbps": hbm["read_bytes"] * 1.98 / kernel["device_cycles"],
                "hbm_mean_read_latency_cycles": hbm["read_latency_cycles"] / hbm["read_completions"]}
            print(json.dumps(result), flush=True)
            return result

    results = await asyncio.gather(*(run(name, configs) for name, configs in variants.items()))
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
