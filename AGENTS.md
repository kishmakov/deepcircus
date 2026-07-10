# Project

This is a research project to study ML approach to handle decision trees.

# Implementation Details

- Keep assertion checks simple `assert foo, bar`, don't use ifs
- In C++ use plain asserts `assert(condition)`
- Keep C++ generation deterministic from `(bitness, case_id)`
- Keep value-tensor input generation deterministic from `(seed, bitness, case_id)`
  and independent of case ordering or Python worker partitioning
- Do not add package-presence guards (e.g. `assert torch is not None`)
- Do not generalize code for running in other environments, it is only run on this machine
- The generator API is bitness-based: use `uint16_t bitness`, not series ids or bit masks


# Code Layout

- `tmp` is the directory not indexed by git
- `bool-bench/src/decision_tree.{h,cpp}` owns `DecisionTree`, `Div`, `Node`, tree evaluation/building, and exact small-bitness solving
- `bool-bench/src/bool_bench.h` declares the public C API; `tree.cpp`, `table.cpp`, and `bool_bench.cpp` implement the tree, table, and circuit portions
- `bool-bench/src/utils.{h,cpp}` owns SplitMix64-based value-input generation and `FlippingSampler`
- Value-tensor APIs accept `reps` and `seed`; the block-and-random input scheme is a C++ implementation detail, so do not expose an input policy or restore Python-generated packed inputs
- `bool-bench/bool_bench.py` owns generator loading, ctypes signatures, the Python generator wrapper, and sample generation helpers
- Value and restriction tensor inputs are generated in C++; do not add Python input-bit generation or packed-input payloads
- Multi-case value, restriction, depth, and node queries use batch C APIs; do not loop over scalar ctypes calls in Python workers
- `src/train.py` owns the bitness training loop, model construction/loading/saving, and per-epoch optimization
- `src/config.py` owns bitness config parsing plus snapshot/state/resume details; bitness training should use config methods instead of reading snapshot internals
- `src/experiment_*.py` should contain experiment logic only; do not put ctypes or shared-library details there
- `scripts/*.py` should stay thin entrypoints over experiment/generator helpers


# Building

- `bool-bench/build.sh` builds bool-bench in `build` directory;
   this is where generator is supposed to be stored

# Running

Verify by running:

```bash
uv run scripts/run.py
```
## Instructions for reporting

On my VM in Google Cloud there is a service listening on the port and redirecting
incoming messages to Telegram. It can be used from Python as following:
- `POST` request to the bot's `/notify` endpoint.
- plain text for simple messages, or JSON with `text`/`message`;

```python
import os
import urllib.request

url = f"http://{os.environ['GC_VM_IP']}:{os.environ['HEREYOUGOBOT_PORT']}/notify"
headers = {"Content-Type": "text/plain"}

request = urllib.request.Request(
    url,
    data=b"Job finished",
    headers=headers,
    method="POST",
)
with urllib.request.urlopen(request, timeout=10) as response:
    response.read()
```
