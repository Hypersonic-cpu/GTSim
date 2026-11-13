# GTSim v0.1.0

GTSim (GPU-Tile-Sim) is a tile-centric GPU simulation framework for LLM
hardware-software co-design. It executes a warp-centric tile graph whose nodes
describe tile-level operations and whose edges encode data and order
dependencies.

This repository accompanies [GPU-Tile-Sim: A Tile-Centric GPU Simulation
Framework for LLM Hardware-Software Co-Design](https://arxiv.org/abs/2607.11262). 
This paper is accepted and to appear in the proceedings of **MICRO 2026**.

This public core release provides the graph-driven simulator core and two H100
reference workload builders:

- a persistent two-stage H100 GEMM; and
- an H100 FA3 two-stage pipeline.

It is a scoped, runnable artifact rather than a complete release of every
frontend, workload, runtime optimization, or case-study feature described in
the GTSim paper. See [the v0.1.0 release scope](docs/release-scope.md) before
citing or extending the repository.

## Build and run

```bash
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# H100 persistent two-stage GEMM: M N K
./build/gtsim_gemm configs/v0.1.0/h100-sxm.yaml 512 512 512

# H100 FA3 reference: batch heads_q heads_kv sequence_q sequence_kv
./build/gtsim_fa3 configs/v0.1.0/h100-sxm.yaml 1 32 8 128 128
```

Each workload takes an architecture YAML patch as its first argument. The H100
profile starts from the simulator's shared H100 defaults; YAML values override
matching fields, omitted values retain those defaults, and unknown fields are
reported as configuration errors.

All memory backends are available in one simulator build by default. Select the
backend at runtime with `--memory-config`; no rebuild is needed when switching:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGTSIM_ENABLE_RAMULATOR2=ON
cmake --build build --target gtsim_gemm --parallel
./build/gtsim_gemm configs/v0.1.0/h100-sxm.yaml --memory-config configs/memconf/0F5M_channels.yaml 128 256 64
./build/gtsim_gemm configs/v0.1.0/h100-sxm.yaml --memory-config configs/memconf/4F1M_hbfine.yaml 128 256 64
./build/gtsim_gemm configs/v0.1.0/h100-sxm.yaml --memory-config configs/memconf/4F1M_eventlab.yaml 128 256 64
./build/gtsim_gemm configs/v0.1.0/h100-sxm.yaml --memory-config configs/memconf/4F1M_flashlab.yaml 128 256 64
./build/gtsim_gemm configs/v0.1.0/h100-sxm.yaml --memory-config configs/ramulator/h100-sxm-ramulator.yaml 128 256 64
```

In the YAML, `hbm.backend` selects `naive` or `ramulator2`; `hbf.backend`
selects `naive`, `hbfine`, `eventlab`, or `flashlab` (`bank_plane` remains a
compatibility alias for `hbfine`). The accompanying topology, capacities and
placement must match the selected device. `build/flashlab-native/` is an internal
third-party library build, linked into the same simulator. Ramulator2 may be
excluded explicitly with `-DGTSIM_ENABLE_RAMULATOR2=OFF` for smaller builds.

The [Ramulator2 HBM3 backend](docs/ramulator-hbm3.md) models H100 SXM
DRAM channels, queues, refresh, and command timing. It is included by default;
the default runtime memory backend remains naive.

The [HBFine Bank/Plane backend](docs/hbf-controller.md), supplied by
`3rdparty/hbfine`, models co-located HBM/HBF devices with explicit tensor
placement. Its portable C++ API and installable target are documented in the
[submodule README](3rdparty/hbfine/README.md).

The same build also includes EventLab and FlashLab memory backends. Select
`hbf.backend: hbfine`, `eventlab`, or `flashlab` in the memory YAML passed to
`--memory-config`. Native libraries use isolated build directories internally;
separate simulator builds are unnecessary.

The optional grouped-plane profile is
[`4F1M_grouped_planes.yaml`](configs/memconf/4F1M_grouped_planes.yaml).
It uses four HBF stacks, 32 planes per command, and 8 MiB SRAM per stack
(6 MiB across private plane SRAM plus 2 MiB on the base die). Enabling SRAM
makes it available; frontend prefetch planning is enabled separately through
`--prefetch-config`. The shared memory wrapper manages residency and lifetime;
native backends retain their hardware timing and transport resources. Existing
profiles retain their original geometry and do not enable these mechanisms.
EventLab's native grade-2 interface provides 94 GB/s per die, while the other
two profiles target 96 GB/s; the comparison retains this model difference.

Run the frontend IR comparison with:

```bash
python3 scripts/experiments/run_flashaccel_comparison.py --out validation/results/flashaccel-comparison
python3 scripts/experiments/report_flashaccel_comparison.py validation/results/flashaccel-comparison
```

Run the smoke tests with `ctest --test-dir build --output-on-failure`. The
checked reference outputs are in
[`configs/v0.1.0/references.json`](configs/v0.1.0/references.json). To
re-run and compare those outputs, use:

```bash
python3 scripts/run_reference.py --build-dir build --verify
```

Run the measured H100 validation suite (8 GEMM and 6 FA3 cases) with:

```bash
python3 scripts/run_validation.py --build-dir build
```

It reports MAPE and Pearson correlation and writes the per-case report to
`validation/results/`. The v0.1.0 result snapshot is in
[`validation/v0.1.0.md`](validation/v0.1.0.md).

The separate experimental [TVM automatic scheduling frontend](docs/tvm-auto-schedule.md)
accepts computation-only FP16 GEMM and uses native MetaSchedule with GTSim
performance feedback. See its guide for pinned TVM setup and reproducible search.

The experimental [A100 FA2 decode workload](docs/fa2-decode.md) supports
single-token BF16 GQA with small batches and split-KV merging using the
Simple memory model.

## License

The repository is distributed under the MIT license.
