# Offline training data for `M_1`

The pairs `(g, f)` and the exact targets that train `M_1[g | f]`, the model of
"Targets for `M_1`" in [`paper.tex`](paper.tex). One pair of files per bitness,

```
data/m1_<bitness>.train
data/m1_<bitness>.val
```

with zero-padded bitness. Supported bitnesses are 8 through 12. The two files
share the layout below exactly; which of them an entry lands in is the whole
difference between them.

`M_2[g | X]` takes a function and a subset rather than a pair of functions, so
its data is not this format.

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

Two functions of the same `n` bits, followed by the targets they were solved
for. Each function is written as its kind, the length of its payload, and the
payload.

| Size | Type | Field |
| ---: | --- | --- |
| 1 | `uint8_t` | kind of `g` |
| 4 | `uint32_t` | payload length of `g` |
| `len(g)` | bytes | payload of `g` |
| 1 | `uint8_t` | kind of `f` |
| 4 | `uint32_t` | payload length of `f` |
| `len(f)` | bytes | payload of `f` |
| 1 | `uint8_t` | minimum decision-tree depth |
| 2 | `uint16_t` | minimum decision-tree size |

A decision tree here computes `g` from the primary inputs together with `f`,
and may query `f` once per path. Depth and size are raw exact values; no model
transform is applied.

## Function kinds

| Kind | Name | The function is backed by |
| ---: | --- | --- |
| 0 | `table` | a truth table -- [`func::TableCase`](../cpp/common/func/table.h) |
| 1 | `tree` | a tree |
| 2 | `tree-over-table` | a table with a tree applied on top of it |

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
