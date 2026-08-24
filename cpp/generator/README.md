# cpp/generator

The synchronous generation library (`gen`). It defines generated cases,
decision trees, circuit evaluation, sampling, and bit-packed result tensors.

`generator.h` is the public entry point. Generation is deterministic from
`(bitness, seed)`; parallel scheduling belongs to
[`../producer/`](../producer/README.md).
