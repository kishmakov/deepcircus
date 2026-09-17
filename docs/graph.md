# Shared M1 graph

[`BuildGraph(n, seed, cells_number)`](../cpp/tools/graph.h) builds geometry
before sampling. There are `n + 1` query coordinates; the helper is last.
Construction is deterministic, with `1 <= n <= 255`, and has two phases.

1. Start with the root and all one- and two-axis cells: `1 + 2 * (n + 1)²`
   cells. For each shuffled order, generate prefix patterns with up to two omitted
   axes and draw one random assignment per pattern, retaining both children
   along each path. Deduplicate cells across all orders and repeat
   with further orders until at least `cells_number` unique cells exist.
2. Create one node per cell in the layer given by its queried-bit count.
   For each free axis, add a question if both zero/one child cells exist,
   regardless of which generation paths produced them. Order questions by axis.

`cells_number` is a minimum: completing whole orders may exceed it. Requests
must be at least `1 + 2 * (n + 1)` and at most `3^(n+1)`, subject to 32-bit
indexing limits. Leaves may retain free coordinates.

[`DenseGraph(graph)`](../cpp/tools/dense_graph.h) copies restrictions and
compacts connectivity into `n + 2` layers. Node `0` is reserved; the root is
layer `0`, node `1`.
`offsets` bounds each node's questions; paired `zero`/`one` indices address the
next layer. `cell` indexes the copied restrictions (`UINT32_MAX` for node `0`).

