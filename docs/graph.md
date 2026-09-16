# Shared M1 graph

[`BuildGraph(n, seed, cells_number)`](../cpp/tools/graph.h) builds geometry
before sampling. There are `n + 1` query coordinates; the helper is last.
Construction is deterministic, with `1 <= n <= 255`, and has two phases.

1. For each shuffled axis order, generate a vector of patterns: the root,
   every one- and two-axis mask, and prefixes with up to two omitted axes.
   Assign all values to patterns with at most two defined axes; for larger
   patterns, try up to `4 * n²` random assignments. Retain both children along
   each path. Repeat until the budget is reached, finishing
   the current path. The initial layers contain `1 + 2 * (n + 1)²` cells;
   beyond that minimum, overshoot is fewer than `2 * (n + 1)` cells.
2. Create one node per cell in the layer given by its queried-bit count.
   For each free axis, add a question if both zero/one child cells exist,
   regardless of which generation paths produced them. Order questions by axis.

The budget includes the root and must be at least `1 + 2 * (n + 1)` and at
most `3^(n+1)`, subject to 32-bit indexing limits. Cells store fixed-coordinate
and value masks. Leaves may retain free coordinates.

[`DenseGraph(graph)`](../cpp/tools/dense_graph.h) copies restrictions and
compacts connectivity into `n + 2` layers. Node `0` is reserved; the root is
layer `0`, node `1`.
`offsets` bounds each node's questions; paired `zero`/`one` indices address the
next layer. `cell` indexes the copied restrictions (`UINT32_MAX` for node `0`).

`SampleGraphInputs(graph, seed, points)` covers the most restricted cells
first, completing their free primary bits randomly. Existing points cover
other compatible cells. It asserts if the requested count cannot hold this
covering sample, then fills remaining rows randomly; repetitions are allowed.
Rows occupy `ceil(n / 8)` bytes, low bit first, with zero padding.
Coverage applies to primary-input projections: both potential helper values
are represented geometrically, but actual presence still depends on `f(x)`.
Sampling leaves the graph unchanged. Training integration is pending.

[`graph_vis bitness seed [cells_number]`](../cpp/train/README.md) prints layers
separated by 60 `#` characters, with `label:` vertices and `-axis-> label`
children. Labels use `0`, `1`, and `*`; axes are zero-based.
