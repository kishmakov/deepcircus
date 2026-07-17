# C++ generation architecture

Generation, task production, and connection handling have separate ownership
boundaries: `cpp/generator/`, `cpp/producer/`, and `cpp/server/`.

## Generator

`cpp/generator/` is synchronous and contains no worker pool. Each call runs to
completion on its calling thread. Independent calls are safe from different
producer threads. `Values` and `Restrictions` are dense, bit-packed matrices
whose rows are cases; exact targets remain separate float vectors holding
`gen::kTargetsPerCase` interleaved values per case: the depth score
`bitness - depth` and the size score `log2(2^bitness - size)`.

A `Case` base class (`case.{h,cpp}`) owns each case's deterministic randomness,
keyed by `(bitness, case_id)`, and the block-and-random `InputShape`
sampling (`SampleValues`/`SampleRestrictions`, using the Gray-code
`NextSequence` walk); the concrete `TableCase`/`TreeCase` only supply the
virtual `Evaluate(std::vector<bool>)`. An `InputShape` is `batches`
independent samplings, each expanded into `batch_size` (power-of-two) points.
The expansion of per-batch base sequences into points is the public
`gen::ExpandInputs`, which `Case::Sample` composes with the case's own
sequence draws; the `expand_inputs` CLI (`cpp/tools/`) exposes it over
stdin/stdout so Python callers reuse the same walk.

Case-ID sampling is split from generation: `TreeSampleCaseIds`/
`TableSampleCaseIds` deterministically sample a bitness x cases pair once, and
`TreeValuesForCases`/`TableValuesForCases`/`TableRestrictionsForCases`
synchronously generate a batch for an explicit, pre-sampled chunk of those case
ids under a given `InputShape`. Because each case's computation is deterministic
from `(bitness, case_id)` alone, generating disjoint chunks of the same sampled
list on different threads and merging their rows with `Values::Concat`
reproduces exactly what one non-chunked call over the full list would have
produced.

## Producer

`cpp/producer/` owns the FIFO thread pool (`ThreadPool`), the iteration/bitness
task queue (`TaskQueue`), and the task types (`TrainingShape`, `Task`,
`TaskResult`) that the daemon later publishes. Wire-level tensor kinds belong to
the daemon.

Coordinates (`iteration` x `bitness` pairs) are produced strictly sequentially
by a dedicated producer thread inside `TaskQueue`, which prefetches finished
results into a bounded buffer (`TaskQueue::kPrefetchDepth`, currently 8) ahead
of consumption; `Take()` blocks until the next result in task order is ready,
pops it, and thereby unblocks the producer to top the buffer back up. Within a
coordinate, the producer samples that tensor's case ids once, splits them
evenly across `ThreadPool::WorkerCount()` into contiguous chunks, and fans
them out via the synchronous generator calls.
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
publication; only the current coordinate is published, and its value and
restriction tensors stay bit-packed (rows padded to whole bytes, little-endian
bit order, bit `b` standing for the float `2*b - 1`) while exact targets are
written as float32. The Python client unpacks the bits on its side, which
keeps a task's segment ~32x smaller than a float32 expansion.
`server.cpp` owns `main`, constructing a `TaskQueue` and handing it to the
daemon as the task source. Both `scripts/bench.py` and the training client
(`src/generator.py`) speak this protocol.

## Tests

`cpp/test/` is a Google Test suite (fetched via CMake `FetchContent`, same
pattern as the `aiger` dependency) linked against the `producer` library.
