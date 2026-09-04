# cpp/common/func

`func.h` holds `Func`, the interface a boolean function of `bitness` bits
implements: call it on one input, on a batch, or over every input at once as a
truth table. `table.h` is the truth-table-backed kind and its batch generation
helpers. `tree.h` is the kind backed by a decision tree, a `tools::BinaryTree`
whose internal nodes hold the bit each one tests. The code is in namespace
`func`; generator-dependent parts are compiled into `gen`, the rest into
`common`.
