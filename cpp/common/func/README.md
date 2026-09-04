# cpp/common/func

`func.h` holds `Func`, the interface a boolean function of `bitness` bits
implements: call it on one input, on a batch, or over every input at once as a
truth table. `table.h` is the truth-table-backed kind. `tree.h` is the kind
backed by a decision tree, a `tools::BinaryTree`
whose internal nodes hold the bit each one tests. `tt.h` composes the two into
the `(g, f)` pair the model estimates: `f` is a table, and `g` is a tree read
over the primary inputs together with `f`'s value, so a tree computing `g` from
that table comes with the case.

The code is in namespace `func` and compiled into `common`; function types do
not depend on the generator.
