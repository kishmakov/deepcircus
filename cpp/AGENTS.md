# C++ generation architecture

Generation, task production, and connection handling have separate ownership
boundaries: `cpp/generator/`, `cpp/producer/`, and `cpp/server/`.

## Generator

`cpp/generator/` is synchronous and contains no worker pool. Each tensor call
runs to completion on its calling thread and returns an opaque ready handle.
Independent calls are safe from different producer threads. Values and recursive
restrictions remain bit-packed until a caller materializes them into a provided
float buffer; exact depth targets remain float vectors.

Case-ID sampling and value inputs are deterministic from their existing seed,
bitness, and case-ID domains. Recursive table handles own all generated
restriction chunks.

## Producer

`cpp/producer/` owns the FIFO thread pool (`ThreadPool`), the iteration/bitness
task queue (`TaskQueue`), and the wire-level task/tensor types (`TrainingShape`,
`Task`, `TaskData`, `TaskResult`, `TensorKind`) that the daemon later publishes.
Workers generate complete coordinates concurrently into compact handles;
`TaskQueue::Take()` exposes them in iteration-major, bitness-major order
regardless of completion order. It has no socket or shared-memory dependency,
so it is built as its own library and exercised directly by `cpp/test`.

A task's validation set depends only on its bitness, so it is generated once
per bitness (iteration 0) ahead of every training task for that bitness,
instead of being recomputed per iteration. Iteration-0 tasks carry only a
validation tensor with exact targets. Training tasks (iteration >= 1) carry
only a train tensor with exact targets (tables up to the solvable bitness,
trees above it) plus, above the solvable bitness, a recursive-table tensor
with restrictions instead of targets.

## Server

`cpp/server/` owns the socket protocol, command loop, and POSIX shared-memory
publication; only the current coordinate is expanded to float32 shared memory.
`server.cpp` owns `main`, constructing a `TaskQueue` and handing it to the
daemon as the task source. Both `scripts/bench.py` and the training client
(`src/generator.py`) speak this protocol.

## Tests

`cpp/test/` is a Google Test suite (fetched via CMake `FetchContent`, same
pattern as the `aiger` dependency) linked against the `producer` library.
