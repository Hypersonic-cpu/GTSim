# Repository Guidelines

## Project Structure & Module Organization

GTSim is a C++23 tile-centric GPU simulator. `include/` contains public headers; `src/` implements scheduling, subpartitions, SMs, GPUs, and simulation. `workloads/gemm/` and `workloads/fa3/` contain the two H100 reference graph builders. `configs/v0.1.0/` holds the YAML hardware profile and deterministic reference manifest. `3rdparty/glaze/` is the YAML parser submodule. `scripts/` provides regression and validation tools; `validation/data/` contains measured H100 targets. Read `docs/release-scope.md` before extending or describing release capabilities.

You may read the paper [GPU-Tile-Sim: A Tile-Centric GPU Simulation Framework for LLM Hardware-Software Co-Design](https://arxiv.org/abs/2607.11262) if necessary for simulator algorithm confusion.

### Version Control
The `AGENTS.md` should be machine-independent, describing only about this project. Absolute path should NOT be used in this file.
Read and write machine-local agent information in `AGENTS.local.md` which is auto excluded by Git.

## Build, Test, and Development Commands

Use CMake 3.20+, a C++23 compiler, Ninja, and Python 3. Initialize submodules with `git submodule update --init --recursive`. CMake keeps CUDA targets at C++20. Run from the repository root:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/gtsim_gemm configs/v0.1.0/h100-sxm.yaml 512 512 512
./build/gtsim_fa3 configs/v0.1.0/h100-sxm.yaml 1 32 8 128 128
ctest --test-dir build --output-on-failure
python3 scripts/check_workload_profile.py
python3 scripts/run_reference.py --build-dir build --verify
python3 scripts/run_validation.py --build-dir build
```

These configure and build the simulator, run example workloads, execute checks, verify reference cycles, and compare against measured H100 data. Validation reports go to ignored `validation/results/`.

## Coding Style & Naming Conventions

Follow nearby code: four-space indentation, braces on the declaration or control-flow line, `GTSim` namespaces, PascalCase types, and snake_case functions and variables. Pair core `.cpp` files with matching `.h` headers. Python scripts use four-space indentation and snake_case names. No formatter or linter is configured; GCC/Clang builds enable `-Wall -Wextra -Wpedantic`.

## Testing Guidelines

CTest registers workload smoke checks, the workload-profile boundary check, and deterministic regression checks; Python checks require an available interpreter. Add checks in `CMakeLists.txt` using descriptive snake_case names such as `gemm_smoke`. No numerical coverage threshold is specified. Before proposing changes, run the build, CTest, profile check, and relevant validation sweep. Explain intentional cycle changes before updating `references.json`.

## Commit & Pull Request Guidelines

History uses short descriptive subjects such as `Update README.md`; no formal prefix convention is established. Keep commits focused. PRs should describe the change, link relevant issues, report checks performed, and explain timing or validation effects.

## Hardware Profiles & Contribution Boundaries

Justify timing changes with a documented shared hardware profile or microbenchmark. Workload builders must not override hardware timing or resource settings. Document redistribution terms before adding traces, proprietary profiler exports, or third-party code/data. Keep contributor guidance portable and avoid machine-specific paths.
