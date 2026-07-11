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

`scripts/bench.py` exercises this protocol. Migration of the main Python
training client to the daemon task protocol remains separate work.
