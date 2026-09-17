# cpp/tools

Utilities in namespace `tools`, shared by the C++ executables and tests.

[`sample.h`](sample.h) and [`sample.cpp`](sample.cpp) provide input sampling
and bit utilities.

[`graph.h`](graph.h) declares the shared M1 restriction graph and
`BuildGraph(bitness, seed, cells_number)`. [`graph.cpp`](graph.cpp)
generates complete axis-order passes until the minimum cell count is met,
using `common/tools/random.h`, then assembles layers and
compatible child pairs, as described in
[`docs/graph.md`](../../docs/graph.md).

`Graph` owns geometric restrictions and nodes with query axes.
`SampleGraphInputs(graph, seed, points)` samples primary inputs covering the
cells' primary projections, independently of construction.
[`graph_vis`](../train/README.md) formats the regular graph in its own `main.cpp`.

[`dense_graph.h`](dense_graph.h) and [`dense_graph.cpp`](dense_graph.cpp) provide
`DenseGraph(graph)`, which copies restrictions and compacts connectivity into arrays.

[`cli.h`](cli.h) declares shared unsigned argument parsing for the generator
and visualizer; [`cli.cpp`](cli.cpp) implements it with assertions enabled in
Release. `Parse16`, `Parse32`, and `Parse64` accept optional inclusive bounds,
defaulting to the full range of their unsigned return type.
