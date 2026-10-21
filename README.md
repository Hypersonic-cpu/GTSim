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

An optional [Ramulator2 HBM3 backend](docs/ramulator-hbm3.md) models H100 SXM
DRAM channels, queues, refresh, and command timing. Enable it with
`-DGTSIM_ENABLE_RAMULATOR2=ON`; the default memory backend remains naive.

An optional [HBFine Bank/Plane backend](docs/hbf-controller.md), supplied by
`3rdparty/hbfine`, models co-located HBM/HBF devices with explicit tensor
placement. Its portable C++ API and installable target are documented in the
[submodule README](3rdparty/hbfine/README.md).

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

## License

The repository is distributed under the MIT license.
