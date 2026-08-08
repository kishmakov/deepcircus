# Offline training data files

The offline training data lives in `data/` as `s<series>_<bitness>_<source>.bin`,
with the bitness zero-padded to two digits (`s1_08_rand.bin` ...
`s2_12_small.bin`) so a plain listing stays in bitness order. The files are
written by `cpp/preparation/` (`offline_train_data_generator <output_dir>
<bitness> <seed> <entries> <small_size_from> <small_size_to>`, one bitness per
invocation, both series and both sources at once) and staged into `data/` by
`scripts/preparation/prepare_offline_train_data.sh`.

The two series are the training sets of the two models in `docs/paper.tex`:

- `s1_*` -- samples for `S1`, which takes a pair of boolean functions
  `g, f : {0,1}^n -> {0,1}`.
- `s2_*` -- samples for `S2`, which takes a boolean function `g` and a
  subset `X ⊆ {0,1}^n`.

Both series use the *same* byte layout: each sample carries two truth tables of
`2^n` bits. Only the reading of the second one differs -- in `s1` it is the
truth table of `f`, in `s2` it is the indicator function of `X` (bit set means
the point belongs to `X`). Nothing in the file says which; the file name does.

## Sources

The suffix is the *source* -- how the pair was drawn. Both sources fill the same
layout and carry equally exact labels; they differ only in where in the target
range they land, and every source is written for both series.

- `_rand` -- both truth tables drawn uniformly at random.
- `_small` -- each entry built around a witness decision tree, so its exact
  minimal size is a chosen small number.

A uniformly random pair is essentially always hard: at `n = 12` the exact size
sits in the hundreds, and no amount of redrawing brings it near single digits.
`_small` therefore inverts the problem -- draw the tree first, then read the
functions off it:

1. Grow a random shape with exactly `k` internal nodes, repeatedly replacing a
   uniformly chosen leaf with an internal node carrying two fresh leaves.
2. Label each internal node with a query, walking down from the root carrying
   the bits already fixed on this path. Series 1 also offers the helper query
   `f`, which the `S1` model lets a path make once, weighted heavily against the
   free input bits so that `f` is actually consulted. Series 2 queries input
   bits only.
3. Label the leaves at random, then force sibling leaves apart -- a node whose
   two leaves agree is removable, and leaving it in would put the exact answer
   under `k` for no reason but sloppy labelling.
4. Read the entry off the tree. Series 1 draws `f` at random and takes
   `g(x) = T(x, f(x))`. Series 2 draws `X` on a fair coin per point, takes `g`
   to agree with `T` on `X`, and draws `g` at random off `X` -- which is what
   makes the subset worth knowing, since computing `g` everywhere then costs
   what a random table costs.
5. Solve exactly. The tree only bounds the target from above, so the entry is
   kept when the solver confirms `k` and redrawn otherwise.

The witness is usually the optimum, so step 5 accepts the large majority of
draws and the rejection loop turns over once or twice. `min_depth` comes along
for whatever the tree happens to give; only the size is aimed at.

The target is `small_size_from + index % (small_size_to - small_size_from + 1)`,
so the entry index cycles through the configured range and an entry count that
is a multiple of the range's width carries every target equally often. The
construction has a ceiling that grows with the bitness -- a target too large for
`2^n` rows to keep a `k`-node tree's distinctions alive is rejected forever, and
the generator aborts on its attempt bound rather than emitting something else.
Measured, `n = 8` reaches about 24, `n = 10` about 48, and `n = 12` past 64;
the configured `1..16` is well inside all of them.

## Layout

All integers are little-endian, unsigned, and unaligned -- the file is a packed
byte stream with no padding anywhere, header included.

### Header (5 bytes)

| Offset | Size | Type       | Field     |
| ------ | ---- | ---------- | --------- |
| 0      | 4    | `uint32_t` | `entries` |
| 4      | 1    | `uint8_t`  | `bitness` |

`entries` is the number of entries that follow; `bitness` is the arity `n` of
every function in the file, and matches the one in the file name. There is no
magic number and no version field: a file is identified by its name, and a
truncated or foreign file is only detectable through the size check below.

### Entry

Entries follow the header back to back, starting at offset 5. With
`table_bytes = 2^n / 8 = 2^(n-3)`:

| Offset within entry | Size          | Type            | Field       |
| ------------------- | ------------- | --------------- | ----------- |
| 0                   | `table_bytes` | bit array       | `g`         |
| `table_bytes`       | `table_bytes` | bit array       | `f` / `X`   |
| `2 * table_bytes`   | 1             | `uint8_t`       | `min_depth` |
| `2 * table_bytes+1` | 2             | `uint16_t`      | `min_size`  |

- `g` -- truth table of the function to be computed.
- `f` / `X` -- truth table of `f` (series 1) or the indicator function of `X`
  (series 2).
- `min_depth` -- depth of the minimal-*depth* decision tree, in `[0, n]`. `n <=
  12` fits a byte with room to spare.
- `min_size` -- number of internal nodes of the minimal-*size* decision tree.
  The two trees need not be the same one. A decision tree over `n` inputs has at
  most `2^n - 1` internal nodes, so at `n = 12` the value is at most 4095 and
  `uint16_t` is enough through the whole supported range.

For series 1 the tree may query `f` once per path. For series 2 only the
assignments in `X` must be computed correctly.

The model-side scales (`s = log2(min_size + 1)` in the paper, and the score
transforms in `cpp/generator/utils.h`) are *not* applied here: the file stores
the raw exact values, and the transform is the trainer's business.

### Truth table encoding

A truth table is `2^n` bits packed into `2^(n-3)` bytes, LSB-first within each
byte -- the same little-endian bit order the generator's packed tensors use.
`2^n` is a multiple of 8 for every supported bitness (`n >= 8`), so tables are
never padded and never share a byte with anything else.

The assignment `x = (x_1, ..., x_n)` is at index

```
i = x_1 * 2^0 + x_2 * 2^1 + ... + x_n * 2^(n-1)
```

i.e. `x_1` is the least significant bit of the index, and the value of the
function at `x` is bit `i % 8` of byte `i / 8`.

## Sizes

Every entry has the same size, so the file size is fully determined by the
header:

```
table_bytes = 2^(n-3)
entry_bytes = 2 * table_bytes + 3
file_bytes  = 5 + entries * entry_bytes
```

| Bitness | `table_bytes` | `entry_bytes` |
| ------- | ------------- | ------------- |
| 8       | 32            | 67            |
| 9       | 64            | 131           |
| 10      | 128           | 259           |
| 11      | 256           | 515           |
| 12      | 512           | 1027          |

That also gives cheap random access: entry `k` starts at
`5 + k * entry_bytes`, so a reader can `seek` to a sample without walking the
ones before it, and can validate a file by checking its size against the
formula.

## Reading

```python
import numpy as np

def read_offline_data(path):
    raw = np.fromfile(path, dtype=np.uint8)
    entries = int(raw[:4].view(np.uint32)[0])
    bitness = int(raw[4])

    table_bytes = 1 << (bitness - 3)
    entry_bytes = 2 * table_bytes + 3
    assert raw.size == 5 + entries * entry_bytes, path

    body = raw[5:].reshape(entries, entry_bytes)
    g = body[:, :table_bytes]
    fx = body[:, table_bytes:2 * table_bytes]
    min_depth = body[:, 2 * table_bytes]
    min_size = body[:, 2 * table_bytes + 1:].copy().view(np.uint16).ravel()
    return bitness, g, fx, min_depth, min_size
```

The packed tables stay packed on the way to the GPU, exactly like the online
generator's tensors: unpacking to +-1 float32 happens on-device in
`src/model.py` (`unpack_bits`, bit `b` -> `2*b - 1`).

## Determinism

The generator takes a seed, an entry count, and the small-size range on the
command line. All of them come from `conf/preparation.conf`, with one entry count
shared by every bitness. Its content must be a pure function of `(seed, series,
bitness, source, entries, small-size range)` -- no clock, no global RNG, no
filesystem state -- so a file can be reproduced from the config alone. That also
binds the `_small` rejection loop: an attempt is keyed by its counter alongside
the entry's coordinates, so entry `k` stays a pure function of `k` rather than
of how many draws the entries before it happened to burn. The seed is *not*
stored in the file: neither the header nor the name carries it, so
`conf/preparation.conf` is the only record of what produced a given `data/*.bin`.
