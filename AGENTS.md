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
- `cpp/` holds the C++ generator sources plus `CMakeLists.txt` and `.clang-format`
- `data/circuits/` holds the benchmark circuits (`*.aig`/`*.bench`); `data/dimensions.txt` records their sizes
- `cpp/decision_tree.{h,cpp}` owns `DecisionTree`, `Div`, `Node`, tree evaluation/building, and exact small-bitness solving
- `cpp/generator.h` declares the public C API; `tree.cpp`, `table.cpp`, and `generator.cpp` implement the tree, table, and circuit portions
- `cpp/aig.cpp` locates `data/circuits` relative to its own source path (falls back to walking up from the cwd)
- `cpp/utils.{h,cpp}` owns SplitMix64-based value-input generation and `FlippingSampler`
- `cpp/dataset.cpp` owns deterministic case-ID sampling, the persistent C++ worker pool, and the lifetime of generated data/restriction handles
- Value-tensor APIs accept `reps` and `seed`; the block-and-random input scheme is a C++ implementation detail, so do not expose an input policy or restore Python-generated packed inputs
- `src/generator.py` owns generator loading, ctypes signatures, the Python generator wrapper, and sample generation helpers
- Value and restriction tensor inputs are generated in C++; do not add Python input-bit generation or packed-input payloads
- Generated buffers transfer from C++ to NumPy at acquire time and are freed automatically when their NumPy/PyTorch views die; only recursive table source handles remain until restriction generation finishes
- `src/sampler.py` owns the thin generator wrapper and pipelined dataset orchestration; do not restore Python multiprocessing or case routing
- Python chooses table/tree batch counts from bitness, while C++ samples case IDs and generates each typed batch
- `src/train.py` owns the bitness training loop, model construction/loading/saving, and per-epoch optimization
- `src/config.py` owns bitness config parsing plus snapshot/state/resume details; bitness training should use config methods instead of reading snapshot internals
- `src/experiment_*.py` should contain experiment logic only; do not put ctypes or shared-library details there
- `scripts/*.py` should stay thin entrypoints over experiment/generator helpers


# Building

- `scripts/build.sh` builds the C++ generator into `cpp/build`;
   this is where the generator (`libgen.so`) is supposed to be stored

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

# Bool Bench Notes

- Keep the C++ generator (`cpp/`) small and dependency-light; it is used as a C/C++ generator with a thin Python helper (`src/generator.py`).
- Preserve the public C ABI in `cpp/generator.h`: keep exported functions `extern "C"` compatible and avoid C++-only types there.
- Prefer straightforward implementations over new abstractions unless they remove real duplication.
- Keep common functionality (such as RNG preparation or random bit sampling) in `cpp/utils.{h,cpp}`.
- When changing behavior, update nearby C++ and Python entry points together if they expose the same generator concept.
- Check builds through the local `cpp/CMakeLists.txt` path when edits touch compiled code.
