# Offline training data for `M_1`

The pairs `(g, f)` that train `M_1[g | f]`, with either exact targets or markers
requesting bootstrapped targets as described in "Targets for `M_1`" in
[`paper.tex`](paper.tex). One pair of files per bitness,

```
data/m1_<bitness>.train
data/m1_<bitness>.val
```

with zero-padded bitness. Supported bitnesses are 8 through 255. The two files
share the layout below exactly; which of them an entry lands in is the whole
difference between them.

`M_2[g | X]` uses the same binary envelope in `m2_<bitness>.train` and
`m2_<bitness>.val`, with the indicator of `X` in the second function's place;
[`data_m2.md`](data_m2.md) says what it means by it.

All integers are unsigned and little-endian. The file is packed without
alignment padding.

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

| Size | Type | Field |
| ---: | --- | --- |
| 1 | `uint8_t` | kind of `g` |
| 4 | `uint32_t` | payload length of `g` |
| `len(g)` | bytes | payload of `g` |
| 1 | `uint8_t` | kind of `f` |
| 4 | `uint32_t` | payload length of `f` |
| `len(f)` | bytes | payload of `f` |
| 1 | `uint8_t` | decision-tree depth, or `UINT8_MAX` if unknown |
| 2 | `uint16_t` | decision-tree size, or `UINT16_MAX` if unknown |

A decision tree here computes `g` from the primary inputs together with `f`,
and may query `f` once per path. Depth and size are raw values; no model
transform is applied. Through bitness 12 they are exact minima from the dynamic
programming solvers. At higher bitnesses a solved entry stores the depth and
size of its witness tree, which are upper bounds on the minima.

The pair `(UINT8_MAX, UINT16_MAX)`, numerically `(255, 65535)`, marks an unknown
target. Such an entry supplies `(g, f)` for computing the target through the
lower-arity `M_1` and auxiliary `M_2` models according to the reductions in
[`paper.tex`](paper.tex); the marker values are not training targets. The two
fields form one marker: either both are maximal or neither is. A mixed pair is
malformed.

The preparation split follows the function kinds. A solved `M_1` entry stores
`g` as `tree-over-table` and its matching `f` as `table`. An unsolved `M_1`
entry stores independent `table` functions and the unknown marker. Validation
contains only solved entries. Through bitness 12, solved `M_2` entries use
`table`/`table` and exact restricted targets. Above that, solved `M_2` entries
use `tree`/`table` witnesses and unsolved ones use `table`/`table`.

## Function kinds

| Kind | Name | The function is backed by |
| ---: | --- | --- |
| 0 | `table` | a seeded table function -- [`func::TableFunc`](../cpp/common/func/table.h) |
| 1 | `tree` | a decision tree -- [`func::TreeFunc`](../cpp/common/func/tree.h) |
| 2 | `tree-over-table` | a table followed by a tree -- [`func::TTFunc`](../cpp/common/func/tt.h) |

The payload is opaque to the reader and the writer. A function serializes
itself to a `std::vector<uint8_t>` and is rebuilt from those same bytes, so the
offline code only ever moves the blob around and the kind byte only says who is
able to read it. What each kind puts inside is documented with that kind.

## File size

For entry `k`,

```
entry_bytes(k) = 13 + len(g_k) + len(f_k)
file_bytes     = 5 + 8 * entries + sum over k of entry_bytes(k)
```

Entry `k` starts at the byte its offset-table slot names, the first one at
`5 + 8 * entries`.
