# Offline storage format for `M_2`

Each entry stores a pair `(g, X)` for `M_2[g | X]` and raw target fields or an
unknown-target marker. Target definitions, preparation, and training-time
processing are described in [`train.md`](train.md). One train/validation pair
exists per bitness:

```
data/m2_<bitness>.train
data/m2_<bitness>.val
```

Bitness is zero-padded and ranges from 8 through 255. The train and validation
files have the same layout.

## Shared envelope

`M_2` uses exactly the binary envelope specified by
[`data_m1.md`](data_m1.md): the same header, entry offsets, function kinds,
target fields, unknown marker, byte order, and size formula.

Within that envelope, the first function slot holds `g`. The second slot,
called `f` in the shared format, holds the indicator of `X`. Its value is one
exactly on inputs that belong to the subset.

## Target fields and function kinds

Known entries store raw decision-tree depth and size (internal-node count).
Unknown entries use the shared `(UINT8_MAX, UINT16_MAX)` marker; both fields
must be maximal together. No model transform is applied in storage.

All three shared function kinds (`table`, `tree`, and `tree-over-table`) are
valid in the envelope; there is no model-specific function-kind schema.
