# cpp/prep

The `data_generator` executable. It creates the configured `M_1` and `M_2`
train/validation files. Training mixes `tt` (tree over a matching table) and
`general` (independent tables) cases; validation uses `tt` only. Both kinds get
exact targets through bitness 12. Above 12, `tt` uses witness bounds and
`general` gets targets through bootstrapping at training startup.

Run parameters live in [`../../conf/`](../../conf/README.md), and the output
format in [`docs/data_m1.md`](../../docs/data_m1.md) and
[`docs/data_m2.md`](../../docs/data_m2.md).
