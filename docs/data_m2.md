# Offline training data for `M_2`

The pairs `(g, X)` that train `M_2[g | X]`, with either exact targets or markers
requesting bootstrapped targets as described in "Targets for `M_2`" in
[`paper.tex`](paper.tex). One pair of files per bitness,

```
data/m2_<bitness>.train
data/m2_<bitness>.val
```

with zero-padded bitness. Supported bitnesses are 8 through 255. The two files
share the layout below exactly; which of them an entry lands in is the whole
difference between them.

The envelope is byte for byte the one [`data_m1.md`](data_m1.md) specifies --
same header, same entry framing, same function kinds, one reader and one writer
serving both -- so a change to it has to land in both files. What follows
repeats it and says what `M_2` means by it: the second function is the
indicator of `X` rather than a second argument of `g`, and the target counts a
different tree.

All integers are unsigned and little-endian. The file is packed without
alignment padding.

## The subset

`X ⊆ {0,1}^n` travels as its indicator: `x ∈ X` exactly when the second
function evaluates to 1 at `x`. Nothing distinguishes that function from any
other -- it is stored, serialized and evaluated like `g` is -- so a uniformly
drawn `table` stands for a uniformly drawn subset, about half the cube.

## Header

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 4 | `uint32_t` | number of entries |
| 4 | 1 | `uint8_t` | bitness `n` |
| 5 | `8 * entries` | `uint64_t[]` | byte offset of each entry |

The header has no magic number or version field.

Entries are variable-length, so the offset table is what keeps reading entry
`k` a seek instead of a scan through the `k - 1` before it. The entry count is
known before the first entry is written, so a writer reserves the table up
front and fills it in as it goes.

## Entry

Two functions of the same `n` bits, followed by their target fields. Each
function is written as its kind, the length of its payload, and the payload.
The second slot is the one `M_1` fills with `f`.

| Size | Type | Field |
| ---: | --- | --- |
| 1 | `uint8_t` | kind of `g` |
| 4 | `uint32_t` | payload length of `g` |
| `len(g)` | bytes | payload of `g` |
| 1 | `uint8_t` | kind of the indicator of `X` |
| 4 | `uint32_t` | payload length of the indicator |
| `len(X)` | bytes | payload of the indicator |
| 1 | `uint8_t` | decision-tree depth, or `UINT8_MAX` if unknown |
| 2 | `uint16_t` | decision-tree size, or `UINT16_MAX` if unknown |

A decision tree here queries the primary inputs and nothing else -- there is no
second function to split on, which is the whole difference from `M_1` -- and
has to compute `g` on `X` alone. Assignments outside `X` are don't-cares, so a
branch reaching none of `X` costs nothing and the tree is free to answer
anything there.

Depth and size are raw values; no model transform is applied. Through bitness
12 they are exact minima, from `SolveForDepthRestricted` and
`SolveForSizeRestricted` in
[`tools/solver.h`](../cpp/common/tools/solver.h). At higher bitnesses a solved
entry stores the depth and size of its witness tree, which are upper bounds on
the minima -- and loose ones here, since the witness computes `g` on the whole
cube while the target only asks about `X`.

The pair `(UINT8_MAX, UINT16_MAX)`, numerically `(255, 65535)`, marks an unknown
target. Such an entry supplies `(g, X)` for computing the target through the
lower-arity `M_2` alone: fixing a primary input drops the arity by one and
carries `X` down with it, and there is no reduction out of `M_2` into another
model. `M_1` is the model that reduces into this one, by splitting on its `f`.
The marker values are not training targets. The two fields form one marker:
either both are maximal or neither is. A mixed pair is malformed.

The preparation split follows the function kinds. Through bitness 12 a solved
entry stores `g` and the indicator as independent `table` functions and the
exact restricted targets. Above that a solved entry stores `g` as a `tree` --
the witness -- with the indicator still a `table`, and an unsolved entry stores
two `table` functions and the unknown marker. Validation contains only solved
entries.

## Function kinds

| Kind | Name | The function is backed by |
| ---: | --- | --- |
| 0 | `table` | a seeded table function -- [`func::TableFunc`](../cpp/common/func/table.h) |
| 1 | `tree` | a decision tree -- [`func::TreeFunc`](../cpp/common/func/tree.h) |
| 2 | `tree-over-table` | a table followed by a tree -- [`func::TTFunc`](../cpp/common/func/tt.h) |

`M_2` writes the first two; the third belongs to `M_1`, whose `g` is built over
the very `f` it is paired with. Readers accept all three regardless, since the
kind byte is what the envelope carries.

The payload is opaque to the reader and the writer. A function serializes
itself to a `std::vector<uint8_t>` and is rebuilt from those same bytes, so the
offline code only ever moves the blob around and the kind byte only says who is
able to read it. What each kind puts inside is documented with that kind.

## File size

For entry `k`,

```
entry_bytes(k) = 13 + len(g_k) + len(X_k)
file_bytes     = 5 + 8 * entries + sum over k of entry_bytes(k)
```

Entry `k` starts at the byte its offset-table slot names, the first one at
`5 + 8 * entries`.
