# Project

This is a research project to study ML approach to handle decision trees.

# Implementation Details

- Keep assertion checks simple `assert foo, bar`, don't use ifs
- In C++ use plain asserts `assert(condition)`
- Keep C++ generation deterministic from `(bitness, case_id)`
- Prefer to fail on assert than quiet ignoring of error
- Do not add package-presence guards (e.g. `assert torch is not None`)
- Do not generalize code for running in other environments, it is only run on this machine
- The generator API is bitness-based: use `uint16_t bitness`, not series ids or bit masks


# Code Layout

- `tmp` is the directory not indexed by git
- `cpp/generator/` holds synchronous generation sources and C API for `scripts/test.py`
- `cpp/producer/` holds the task queue, thread pool, and generator orchestration; it is a library independent of the daemon and is covered by `cpp/test`
- `cpp/server/` holds the daemon, socket protocol, shared-memory publication, and `main`
- `cpp/test/` holds the Google Test suite (fetched via CMake `FetchContent`), currently exercising `cpp/producer`
- `data/circuits/` holds the benchmark circuits (`*.aig`/`*.bench`); `data/dimensions.txt` records their sizes
- `cpp/generator/case.{h,cpp}` owns the `Case` base class: per-case deterministic randomness keyed by `(bitness, case_id)`, the fair-coin bit stream, and the block-and-random `InputShape` sampling (`SampleValues`/`SampleRestrictions`/`SampledValueString`); `TableCase`/`TreeCase` supply the virtual `Evaluate(std::vector<bool>)`
- `cpp/generator/tree.{h,cpp}` owns `TreeCase`, `Div`, `Node`, tree evaluation/building, and exact small-bitness solving
- `cpp/generator/generator.h` declares the public synchronous C API and the `InputShape` sampling shape
- `cpp/generator/aig.cpp` locates `data/circuits` relative to its own source path (falls back to walking up from the cwd)
- `cpp/generator/utils.{h,cpp}` owns deterministic case-ID sampling (`SampleCaseIds`/`DomainSeed`), the Gray-code `NextSequence` walk, and the bit-layout helpers (`SplitBitsInGroups`/`FullBitId`/`SizeScore`)
- `cpp/generator/generator.cpp` owns the compact bit-packed `BitMatrix` (`gen::Values`/`gen::Restrictions`) and the circuit forwarding; `tree.cpp`/`table.cpp` own their `*SampleCaseIds`/`*ForCases` batch entry points
- `cpp/producer/task_queue.{h,cpp}` owns the wire-level task/tensor types and the task queue; coordinates are produced strictly sequentially, chunking each coordinate's case ids across the thread pool and merging the chunks with `gen::Values::Concat`/`gen::Restrictions::Concat`
- `cpp/producer/thread_pool.{h,cpp}` owns the FIFO worker pool used to generate a coordinate's case-id chunks in parallel
- `cpp/server/daemon.{h,cpp}` owns the socket protocol, command loop, and shared-memory publication; `cpp/server/server.cpp` owns `main` and wires the daemon to a `TaskQueue`
- `scripts/test.sh` builds `cpp/test` under AddressSanitizer/UndefinedBehaviorSanitizer (in `cpp/build-asan`, separate from the Release `cpp/build`) and runs it with leak detection on
- Value-tensor APIs accept an `InputShape` (`batches` x `batch_size`); the block-and-random input scheme is a C++ implementation detail, so do not expose an input policy or restore Python-generated packed inputs
- `src/generator.py` owns the generator daemon client: server spawning, the task protocol, and shared-memory task views
- Value and restriction tensor inputs are generated in C++; do not add Python input-bit generation
- Generator values stay bit-packed end to end: the daemon publishes tensors packed (targets stay float32) and `src/generator.py` expands them to float32 with `np.unpackbits` (little-endian, bit `b` -> `2*b - 1`) -- value tensors eagerly at parse time, the large restrictions tensor lazily through `restriction_chunks()`
- `src/sampler.py` owns the thin generator wrapper and pipelined dataset orchestration; do not restore Python multiprocessing or case routing
- Python chooses table/tree batch counts from bitness, while C++ samples case IDs and generates each typed batch
- `src/train.py` owns the bitness training loop, model construction/loading/saving, and per-epoch optimization
- `src/config.py` owns config parsing plus snapshot/state/resume details; training should use config methods instead of reading snapshot internals
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
- Preserve the public C ABI in `cpp/generator/generator.h`: keep exported functions `extern "C"` compatible and avoid C++-only types there.
- Prefer straightforward implementations over new abstractions unless they remove real duplication.
- Keep common functionality (such as RNG preparation or random bit sampling) in `cpp/generator/utils.{h,cpp}`.
- When changing behavior, update nearby C++ and Python entry points together if they expose the same generator concept.
- Check builds through the local `cpp/CMakeLists.txt` path when edits touch compiled code.
