# cpp/producer

Parallel production of generated training and validation data. This library
contains the worker pool, task descriptions, and ordered task queue used by the
generator server.

Socket and shared-memory concerns remain in
[`../server/`](../server/README.md).
