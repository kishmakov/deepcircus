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
- `cpp/generator/` holds synchronous generation sources
- `cpp/common/` holds the `common` library; `cpp/common/offline/read_write.{h,cpp}` is the shared reader and writer for the packed offline-data format, `cpp/common/tools/random.{h,cpp}` owns the shared SplitMix64 primitives and random stream, and `cpp/common/tools/solver.{h,cpp}` owns the exact truth-table solvers
- `cpp/producer/` holds the task queue, thread pool, and generator orchestration; it is a library independent of the daemon and is covered by `cpp/test`
- `cpp/preparation/` holds the `offline_train_data_generator` executable that writes offline training data one bitness at a time (`offline_train_data_generator <output_dir> <bitness> <seed> <entries> <small_size_from> <small_size_to>` -> `s{1,2}_<bitness>_{rand,small}.bin`, bitness zero-padded to two digits so a listing sorts in bitness order); it uses the specialized exact solvers for the two series, is a subdirectory of the `cpp/` CMake project, and is built by `scripts/build.sh` into `cpp/build/preparation/`; `scripts/preparation/build_offline_generator.sh` builds that one target and publishes it as the `execs/offline_train_data_generator` symlink, which is the name callers use
- `cpp/preparation/main.cpp` is the driver -- argument parsing, file naming, the source x series loop -- and `cpp/preparation/sampler.{h,cpp}` owns the two entry samplers behind the `_rand` and `_small` file suffixes, both taking a `preparation::Parameters` so the driver can hold them in one table. `_small` inverts the sampling problem to reach small targets: it grows a random witness decision tree of `k` internal nodes, reads `g` (and, for series 1, the helper `f`; for series 2, the subset `X` with `g` random off it) off that tree, then solves exactly and stores what the solver says. The witness bounds the target from above and is never the label -- there is no search for a pair hitting `k` on the nose, so a draw that partly collapses keeps its smaller size and one draw plus one solve is the whole cost of an entry. `docs/offline_data.md` documents the construction, why the entry count should be a multiple of the small-size range's width, and the per-bitness ceiling
- `cpp/server/` holds the daemon, socket protocol, shared-memory publication, and `main`
- `cpp/test/` holds the Google Test suite (fetched via CMake `FetchContent`), exercising `cpp/generator` and `cpp/producer`
- `cpp/validation/` holds the `validation` executable that checks training results: it reconstructs a scheme for a decision tree and verifies it against the exact solvers. It links `common` only -- no `gen::` names, no `generator/` includes -- and is a subdirectory of the `cpp/` CMake project, so `scripts/build.sh` builds it; `cpp/validation/build.sh` builds that one target
- `data/circuits/` holds the benchmark circuits (`*.aig`/`*.bench`); `data/dimensions.txt` records their sizes
- `cpp/generator/case.{h,cpp}` owns the `Case` base class: per-case deterministic randomness keyed by `(bitness, case_id)`, the fair-coin bit stream (`GenerateBool`), and the sampling entry points (`SampleValues`/`SampleRestrictions`/`SampledValueString`); `TableCase`/`TreeCase` supply the virtual `Evaluate(std::vector<bool>)`
- `cpp/tools/` is its own `tools` library in namespace `tools`, and it must not depend on `cpp/generator/`: no `gen::` names, no `generator/` includes, no link edge. The dependency runs one way -- `gen` links `tools` and `generator.h` includes `sample.h`. Keep it that way when adding to `cpp/tools/`
- `cpp/tools/sample.{h,cpp}` owns `tools::InputShape` and the whole input-point walk: the bit primitives (`BitsFromChars`/`SplitBitsInGroups`/`NextSequence`), `GenerateSequence` (draws one base sequence off a `tools::BitSource` fair-coin callback), `ExpandInputs` (block-and-random expansion of the per-batch bases), and `SampleInputs`, the composition `Case::Sample` calls with its own `GenerateBool` stream
- `cpp/tools/expand_inputs.cpp` is a thin stdin/stdout CLI over `tools::ExpandInputs` so Python reuses the exact C++ input walk instead of mirroring it (client helper: `expand_inputs` in `src/generator.py`, used by `scripts/plot_parity.py`)
- `cpp/generator/tree.{h,cpp}` owns `TreeCase`, `Div`, `Node`, tree evaluation/building, and exact small-bitness solving
- `cpp/common/tools/solver.{h,cpp}` owns the exact truth-table solvers `tools::SolveForDepth`/`tools::SolveForSize` (the 3^bitness dynamic programs behind a solvable table's targets), their `SolveFor...Given` variants scoring a tree that may also query a second function once per path (the paper's `S_1[g | f]`), their `SolveFor...Restricted` variants scoring a tree that only has to be right on a subset (`S_2[g | X]`), and `tools::kMaxSolvableBitness`, which `gen::kSolvableTableBitness` aliases; included as `"tools/solver.h"` and called from `table.cpp`. `cpp/test/test_solver.cpp` covers it, cross-checking against a brute-force reference at small bitness
- `cpp/generator/generator.h` declares the public synchronous API; it includes `tools/sample.h` and re-exports `gen::InputShape` as an alias of `tools::InputShape`
- `cpp/generator/aig.cpp` locates `data/circuits` relative to its own source path (falls back to walking up from the cwd)
- `cpp/generator/utils.{h,cpp}` owns deterministic case-ID sampling (`SampleCaseIds`/`DomainSeed`/`TaskSeed`) and the case-shaped helpers `FullBitId`/`SizeScore`; its random stream and SplitMix64 primitives come from `cpp/common/tools/random.{h,cpp}`
- `cpp/generator/generator.cpp` owns the compact bit-packed `BitMatrix` (`gen::Values`/`gen::Restrictions`) and the circuit forwarding; `tree.cpp`/`table.cpp` own their `*SampleCaseIds`/`*ForCases` batch entry points
- `cpp/producer/task_queue.{h,cpp}` owns the wire-level task/tensor types and the task queue; coordinates are produced strictly sequentially, chunking each coordinate's case ids across the thread pool and merging the chunks with `gen::Values::Concat`/`gen::Restrictions::Concat`
- `cpp/producer/thread_pool.{h,cpp}` owns the FIFO worker pool used to generate a coordinate's case-id chunks in parallel
- `cpp/server/daemon.{h,cpp}` owns the socket protocol, command loop, and shared-memory publication; `cpp/server/server.cpp` owns `main` and wires the daemon to a `TaskQueue`
- `cpp/validation/main.cpp` keeps its asserts under Release: its CMake entry undefines `NDEBUG` for that file, so the checks it runs are the point of the binary
- `cpp/preparation/main.cpp` does the same: an unopenable output path or an out-of-range bitness must abort, not pass silently, so its CMake entry undefines `NDEBUG` too
- `scripts/preparation/prepare_offline_train_data.sh` makes `data/s{1,2}_{08..12}_{rand,small}.bin` exist: it regenerates only the bitnesses whose set of four is incomplete, staging through the work dir and moving the results into `data/`. Its `work_dir`, `seed`, shared entry count, bitness range, and `small_size_from`/`small_size_to` all come from `conf/preparation.conf`; a key missing there is fatal rather than defaulted. `data/*` is gitignored apart from `data/circuits/`
- `scripts/test.sh` builds `cpp/test` under AddressSanitizer/UndefinedBehaviorSanitizer (in `cpp/build-asan`, separate from the Release `cpp/build`) and runs it with leak detection on
- Value-tensor APIs accept an `InputShape` (`batches` x `batch_size`); the block-and-random input scheme is a C++ implementation detail, so do not expose an input policy or restore Python-generated packed inputs
- `src/generator.py` owns the generator daemon client: server spawning, the task protocol, and shared-memory task views
- Value and restriction tensor inputs are generated in C++; do not add Python input-bit generation
- Generator values stay bit-packed all the way to the GPU: the daemon publishes tensors packed (targets stay float32), `src/generator.py` keeps them as packed uint8 rows (`restriction_chunks()` yields packed row chunks, logical shape in `restrictions_shape`), and expansion to ±1 float32 (little-endian, bit `b` -> `2*b - 1`) happens on-device in `src/model.py` `unpack_bits` -- `DeepSetPredictor.forward` unpacks uint8 input itself, float32 input passes through
- `src/sampler.py` owns the thin generator wrapper and pipelined dataset orchestration; do not restore Python multiprocessing or case routing
- Python chooses table/tree batch counts from bitness, while C++ samples case IDs and generates each typed batch
- `src/model.py` owns `DeepSetPredictor`: input `(batch, batches * points_in_batch, point_dim)` is split into `batches` groups, each processed by its own dedicated 2-layer `phi` MLP (an `nn.ModuleList`, no weight sharing), the outputs concatenated into a `batches * phi_out` vector and fed to a single `rho` head
- `src/train.py` owns the bitness training loop, model construction/loading/saving, and per-epoch optimization
- `src/config.py` owns config parsing plus snapshot/state/resume details; training should use config methods instead of reading snapshot internals
- `src/experiment_*.py` should contain experiment logic only; do not put ctypes or shared-library details there
- `scripts/*.py` should stay thin entrypoints over experiment/generator helpers


# Building

- `scripts/build.sh` builds the C++ generator into `cpp/build`

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
- `cpp/generator/generator.h` is the public synchronous generator API; keep it the single entry point for C++ callers.
- Prefer straightforward implementations over new abstractions unless they remove real duplication.
- Keep case-keyed common functionality (such as RNG preparation or seed derivation) in `cpp/generator/utils.{h,cpp}`; generator-independent bit helpers belong in `cpp/tools/sample.{h,cpp}`.
- When changing behavior, update nearby C++ and Python entry points together if they expose the same generator concept.
- Check builds through the local `cpp/CMakeLists.txt` path when edits touch compiled code.
