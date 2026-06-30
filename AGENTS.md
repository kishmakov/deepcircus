# Project

This is a research project to study ML approach to handle decision trees.

# Implementation Details

- Keep assertion checks simple `assert foo, bar`, don't use ifs
- In C++ use plain asserts `assert(condition)`
- Keep C++ generation deterministic from `(bitness, case_id)`
- Do not add package-presence guards (e.g. `assert torch is not None`)
- Do not generalize code for running in other environments, it is only run on this machine
- The generator API is bitness-based: use `uint16_t bitness`, not series ids or bit masks


# Code Layout

- `tmp` is the directory not indexed by git
- `bool-bench/decision_tree.{h,cpp}` owns `DecisionTree`, `Div`, `Node`, and tree evaluation/building
- `bool-bench/small_bitness.{h,cpp}` owns exact small-bitness solving and cache read/write
- `bool-bench/bool_bench.{h,cpp}` owns the public C API and main generation dispatch
- `bool-bench/bool_bench.py` owns generator loading, ctypes signatures, the Python generator wrapper, and sample generation helpers
- `src/experiment_*.py` should contain experiment logic only; do not put ctypes or shared-library details there
- `scripts/*.py` should stay thin entrypoints over experiment/generator helpers


# Building

- `bool-bench/build.sh` builds bool-bench in `build` directory;
   this is where generator is supposed to be stored

# Running

```bash
uv run scripts/run.py
```
