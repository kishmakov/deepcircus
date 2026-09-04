# cpp/preparation

The `data_generator` executable. It creates the configured `M_1` and `M_2`
train/validation files, mixing solved witness pairs with unknown table pairs.
The exact solvers label solved entries through bitness 12.

Run parameters live in [`../../conf/`](../../conf/README.md), and the output
format is documented in
[`docs/data_m1.md`](../../docs/data_m1.md).
