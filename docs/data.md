# Offline training data format

Offline training files are named
`data/s<series>_<bitness>_<source>.bin`, with zero-padded bitness. Supported
bitnesses are 8 through 12.

- Series 1 stores truth tables for `g` and helper function `f`.
- Series 2 stores a truth table for `g` and the indicator of subset `X`.
- The source is `rand` or `small`; it describes the entry distribution and
  does not change the binary layout.

All integers are unsigned and little-endian. The file is packed without
alignment padding.

## Header

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | 4 | `uint32_t` | number of entries |
| 4 | 1 | `uint8_t` | bitness `n` |

The header has no magic number or version field.

## Entry

Entries begin at byte 5 and have a fixed size within a file. Define
`table_bytes = 2^(n-3)`.

| Offset | Size | Type | Field |
| ---: | ---: | --- | --- |
| 0 | `table_bytes` | bit array | truth table of `g` |
| `table_bytes` | `table_bytes` | bit array | truth table of `f` or indicator of `X` |
| `2 * table_bytes` | 1 | `uint8_t` | minimum decision-tree depth |
| `2 * table_bytes + 1` | 2 | `uint16_t` | minimum decision-tree size |

For series 1, a decision tree may query `f` once per path. For series 2, it
only needs to be correct on `X`. Depth and size are raw exact values; no model
transform is applied.

## Truth tables

Truth-table bits are packed least-significant bit first within each byte. For
an assignment `x = (x_1, ..., x_n)`, the table index is

```
i = x_1 * 2^0 + x_2 * 2^1 + ... + x_n * 2^(n-1)
```

The value is bit `i % 8` of byte `i / 8`.

## File size

```
entry_bytes = 2 * table_bytes + 3
file_bytes  = 5 + entries * entry_bytes
```

Entry `k` starts at byte `5 + k * entry_bytes`.
