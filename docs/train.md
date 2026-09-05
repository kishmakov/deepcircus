# Training data processing

The C++ generator prepares offline function pairs and raw targets. At training
startup, the daemon reads those files and Python reconstructs unknown targets
through prerequisite models. The daemon then samples the pairs into fresh
training cases for each epoch, and Python unpacks the cases on the training
device.

The on-disk layouts are specified separately in [`data_m1.md`](data_m1.md) and
[`data_m2.md`](data_m2.md). The target definitions and reductions follow
"Targets for `M_1`" and "Targets for `M_2`" in the [paper](paper.tex).

## Target definitions

An `M_1[g | f]` decision tree computes `g` from the primary inputs together
with the auxiliary function `f`, and may query `f` once per path.

An `M_2[g | X]` decision tree queries only primary inputs and must compute `g`
on `X`. Assignments outside `X` are don't-cares: a branch reaching no member of
`X` costs nothing and may return either value. The second function in an
`M_2` pair is the indicator of `X`. A uniformly seeded `table` therefore
represents a uniformly drawn subset, typically about half of the input cube.

Depth and size count the tree's depth and internal nodes. Through bitness 12,
solved entries have exact minimum depth and size from the dynamic programming
solvers. For `M_2`, these are `SolveForDepthRestricted` and
`SolveForSizeRestricted` in
[`tools/solver.h`](../cpp/common/tools/solver.h).

Above bitness 12, solved entries use the depth and size of a witness tree as
upper bounds on the minima. For `M_2`, the witness computes `g` on the whole
cube, so its bounds may be loose when only `X` is constrained.

## Offline preparation

Preparation is configured in [`conf/preparation.yaml`](../conf/preparation.yaml)
and implemented in [`cpp/prep/`](../cpp/prep/README.md). Generation is
deterministic from `(bitness, seed)`.

For `M_1`, a solved entry pairs a `tree-over-table` function `g` with its
matching `table` function `f`. An unsolved entry uses independent `table`
functions and the unknown-target marker.

For `M_2`:

- Through bitness 12, a solved entry uses independent `table` functions for
  `g` and the indicator of `X`, with exact restricted targets.
- Above bitness 12, a solved entry uses a `tree` witness for `g`, a `table`
  indicator, and the witness's raw depth and size.
- An unsolved entry uses independent `table` functions in both slots and the
  unknown-target marker.

`M_2` preparation produces only `table` and `tree` functions, although the
shared storage format also permits `tree-over-table`. Validation contains only
solved entries for both models.

## Training scores

The daemon converts known raw targets into the two scores defined in
[`tools/score.h`](../cpp/common/tools/score.h):

```
depth_score = n - depth
size_score  = log2(2^n - size)
```

Both scores lie in `[0, n]` and increase as the function gets simpler. The
unknown-target marker is never used as a training target; those entries need
scores reconstructed before sampling the first training epoch.

## Reconstructing unknown targets

At startup, the daemon streams sampled reductions of unknown entries in bounded
chunks. [`src/bootstrap.py`](../src/bootstrap.py) evaluates them with already
trained models, combines the branch predictions, and installs one pair of
scores per unknown entry in the daemon.

For `M_2` at bitness `n`, both restrictions of every primary input are evaluated
by `M_2` at bitness `n - 1`. Python combines the two branches of each split and
selects the best score for each target over all input splits.

For `M_1` at bitness `n`, primary-input restrictions are evaluated by `M_1`
at bitness `n - 1`. Splitting on the helper `f` also produces the subsets
`f^-1(0)` and `f^-1(1)`, evaluated by `M_2` at the original bitness `n`.
Python combines the helper branches and takes the elementwise maximum of the
primary-input and helper-split scores.

Prerequisite checkpoints must already exist in the configured `work_dir`:

| Model being trained | Required checkpoints for unknown targets |
| --- | --- |
| `m2` at bitness `n` | `m2_<n-1>.best.pt` |
| `m1` at bitness `n` | `m1_<n-1>.best.pt` and `m2_<n>.best.pt` |

Checkpoint bitnesses are zero-padded. Above bitness 12, train in increasing
bitness, with `m2 n` before `m1 n`. Every finished run keeps its best
checkpoint in `data/` under that same name, so a `work_dir` emptied between
runs is restored by copying the prerequisites back into it. Once
reconstructed scores are installed, unknown and solved entries are served
alike. Training reads the offline files without modifying them.

## Sampling and device input

The [`C++ daemon`](../cpp/server/README.md) samples each function pair at
`batches * points_in_batch` inputs. Each point contains:

```
[ x_1..x_n | g(x), g(x^e_1)..g(x^e_n) | h(x), h(x^e_1)..h(x^e_n) ]
```

Here `x^e_i` flips input bit `i`, and `h` is `f` for `M_1` or the indicator
of `X` for `M_2`. A point has `3n + 2` bits. The daemon packs these bits and
serves the cases and score targets through shared memory; the
[`Python client`](../src/daemon/README.md) copies each epoch before requesting
the next one.

Epoch 0 serves validation. Each positive epoch makes one pass over the training
file with freshly sampled inputs. The epoch id enters the seed, so repeating
an epoch id reproduces the same samples, independently of thread scheduling.

[`model.unpack_bits`](../src/model.py) expands the packed input on the training
device into float32 values in `{-1, +1}`, mapping each bit `b` to `2*b - 1`.
The training loop fits the model to the two score targets. Run commands are
listed in the [project README](../README.md#run-the-pipeline), and training and
sampling settings live in [`conf/train.yaml`](../conf/train.yaml).
