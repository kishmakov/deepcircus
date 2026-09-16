# cpp/train

[`main.cpp`](main.cpp) builds `graph_vis`, which writes the restriction graph
as layered text to stdout:

```sh
scripts/train/build_visualizer.sh
execs/graph_vis 3 239 45 > graph.txt
```

Arguments are `bitness` (1–255), an unsigned 64-bit `seed`, and an optional
`cells_number`, defaulting to `4 * (bitness + 1) + 1`. The budget counts all
vertices and must include the root and the `2 * (bitness + 1)` first-layer
vertices. Construction finishes its last random path, possibly exceeding the
budget; see [`docs/graph.md`](../../docs/graph.md).

Node labels use `0` and `1` for fixed coordinates and `*` for unqueried ones,
with the helper coordinate last. Each layer starts with 60 `#` characters and
a newline. Each vertex has a `label:` line, followed by children as
`  -axis-> label`, with zero-based axes. Shared nodes print once per layer;
leaves print only their label and may still contain `*`.
