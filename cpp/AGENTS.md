# C++ generation architecture

Generation, task production, and connection handling have separate ownership
boundaries: `cpp/generator/`, `cpp/producer/`, and `cpp/server/`.

## Generator

`cpp/generator/` is synchronous and contains no worker pool. Each call runs to
completion on its calling thread. Independent calls are safe from different
producer threads. `Values` and `Restrictions` are dense, bit-packed matrices
whose rows are cases; exact depth targets remain separate float vectors.

Case-ID sampling is split from generation: `TreeSampleCaseIds`/
`TableSampleCaseIds` deterministically sample a bitness x cases pair once, and
`TreeValuesForCases`/`TableValuesForCases`/`TableRestrictionsForCases`
synchronously generate a batch for an explicit, pre-sampled chunk of those case
ids. Because each case's computation is deterministic from `(bitness, case_id)`
alone, generating disjoint chunks of the same sampled list on different threads
and merging their rows with `Values::Concat` reproduces exactly what one
non-chunked call over the full list would have produced.

## Producer

`cpp/producer/` owns the FIFO thread pool (`ThreadPool`), the iteration/bitness
task queue (`TaskQueue`), and the task types (`TrainingShape`, `Task`,
`TaskResult`) that the daemon later publishes. Wire-level tensor kinds belong to
the daemon.

Coordinates (`iteration` x `bitness` pairs) are produced strictly sequentially,
on demand, from `TaskQueue::Take()` — there is no cross-coordinate prefetch
pipeline. Parallelism lives inside a single coordinate instead: `Take()` samples
that tensor's case ids once, splits them evenly across `ThreadPool::WorkerCount()`
into contiguous chunks, and fans them out via the synchronous generator calls.
Exact-target value rows are merged back into one `gen::Values`; for recursive
tables, the worker chunks are likewise merged (`Values::Concat` plus
`Restrictions::Concat`) into a single `gen::GeneratedRestrictions` pairing
unlabeled table values with their restriction matrix, so published tensors do
not depend on worker count. `TaskQueue` has no socket or
shared-memory dependency, so it is built as its own library and exercised
directly by `cpp/test`.

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
