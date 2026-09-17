# cpp/train

[`main.cpp`](main.cpp) builds `graph_vis`, which writes the graph as text for
reading; [`dump.cpp`](dump.cpp) builds `graph_dump`, which writes the same graph
packed for [`src/example.py`](../../src/example.py).

## graph_vis

It writes the restriction graph as layered text to stdout:

```sh
scripts/train/build_visualizer.sh
execs/graph_vis 3 239 45 > graph.txt
```

Arguments are `bitness` (1–255), an unsigned 64-bit `seed`, and `cells_number`.
`cells_number` is the minimum number of nodes. Generation repeats complete
axis-order passes until the minimum is met.
The root and all one- and two-axis cells are always included. See [`docs/graph.md`](../../docs/graph.md).

Node labels use `0` and `1` for fixed coordinates and `*` for unqueried ones,
with the helper coordinate last. Each layer starts with 60 `#` characters and
a newline. Each vertex has a `label:` line, followed by children as
`  -axis-> label`, with zero-based axes. Shared nodes print once per layer;
leaves print only their label and may still contain `*`.

## graph_dump

```sh
scripts/train/build_graph_dump.sh
execs/graph_dump 8 239 8192 256 > graph.bin
```

Arguments are `bitness` (1–31, so a cell's `bitness + 1` axes fit one word),
`seed`, `cells_number`, and the `points` handed to `SampleGraphInputs`.
Inputs are distinct, so `points` cannot exceed `2^bitness`.
It writes little-endian `uint32` words to stdout: the magic `DCG1`, `bitness`,
the cell, layer and point counts, then every cell's `fixed` and `values`, then
each [`DenseGraph`](../tools/dense_graph.h) layer as `nodes`, `edges`,
`offsets`, `zero`, `one`, `cell`, and last the sampled inputs. The helper axis
is bit `bitness` throughout.
