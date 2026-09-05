# Offline storage format for `M_1`

Each entry stores a pair `(g, f)` for `M_1[g | f]` and raw target fields or an
unknown-target marker. Target definitions, preparation, and training-time
processing are described in [`train.md`](train.md). One pair of files per bitness,

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

Entries are variable-length. Each offset is an absolute byte position from the
start of the file, allowing direct access to an entry.

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
| 2 | `uint16_t` | decision-tree size (internal-node count), or `UINT16_MAX` if unknown |

Depth and size are raw values; no model transform is applied in storage.

The pair `(UINT8_MAX, UINT16_MAX)`, numerically `(255, 65535)`, marks an unknown
target. The two fields form one marker: either both are maximal or neither is.
A mixed pair is malformed. The marker values are not depth and size values.

## Function kinds

| Kind | Name | The function is backed by |
| ---: | --- | --- |
| 0 | `table` | a seeded table function -- [`func::TableFunc`](../cpp/common/func/table.h) |
| 1 | `tree` | a decision tree -- [`func::TreeFunc`](../cpp/common/func/tree.h) |
| 2 | `tree-over-table` | a table followed by a tree -- [`func::TTFunc`](../cpp/common/func/tt.h) |

The payload is an opaque byte sequence in the shared envelope. The kind byte
identifies its encoding, documented with the corresponding function class.

## File size

For entry `k`,

```
entry_bytes(k) = 13 + len(g_k) + len(f_k)
file_bytes     = 5 + 8 * entries + sum over k of entry_bytes(k)
```

Entry `k` starts at the byte its offset-table slot names, the first one at
`5 + 8 * entries`.
