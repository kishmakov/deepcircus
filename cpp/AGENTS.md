# C++ generation architecture

Generation and concurrency have separate ownership boundaries.

## Generator

`cpp/generator/` is synchronous and contains no worker pool. Each tensor call
runs to completion on its calling thread and returns an opaque ready handle.
Independent calls are safe from different server threads. Values and recursive
restrictions remain bit-packed until a caller materializes them into a provided
float buffer; exact depth targets remain float vectors.

Case-ID sampling and value inputs are deterministic from their existing seed,
bitness, and case-ID domains. Recursive table handles own all generated
restriction chunks.

## Server

`cpp/server/` owns the FIFO thread pool, iteration/bitness task queue, ordered
result publication, socket protocol, and POSIX shared memory. Workers generate
complete coordinates concurrently into compact handles. Results are exposed in
iteration-major, bitness-major order, with only the current coordinate expanded
to float32 shared memory.

A task's validation set depends only on its bitness, so it is generated once
per bitness (iteration 0) ahead of every training task for that bitness,
instead of being recomputed per iteration. Iteration-0 tasks carry only a
validation tensor with exact targets. Training tasks (iteration >= 1) carry
only a train tensor with exact targets (tables up to the solvable bitness,
trees above it) plus, above the solvable bitness, a recursive-table tensor
with restrictions instead of targets.

Both `scripts/bench.py` and the training client (`src/generator.py`) speak this
protocol.
