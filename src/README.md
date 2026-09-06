# src

The Python side: the config, the daemon client, the model, and the training
loop. Nothing here generates input bits or evaluates functions -- that is
[`cpp/`](../cpp/README.md)'s job, and [`daemon/`](daemon/) is the only place
that knows how to ask for the result.

| File | What it owns |
| --- | --- |
| `config.py` | `conf/train.yaml` for one `(model, bitness)` run |
| [`daemon/`](daemon/README.md) | everything about where the data comes from |
| `bootstrap.py` | reconstructing unknown targets through prerequisite models |
| `train.py` | the training loop of one run, and its weights and metrics |
| `model.py` | `DeepSetPredictor` and the on-device unpacking |
| `ordered.py` | the restriction recursion over a sampled, cubically bounded lattice |

## config.py

`load_train_config(model, bitness)` reads `conf/train.yaml`, which holds
everything a run needs beyond the two names it is called with. The result
carries the paths the run touches -- the two files it reads out of `data_dir`,
and the weights and metrics it writes, which never leave `work_dir` -- and the
assertions the daemon would otherwise fail on later.

## daemon/

The one place that knows the data is sampled by a C++ process over a socket.
`open_training_data(config)` hands back a `TrainingData`, which answers
`validation()` once and `epoch(i)` as often as it is asked; both return
`(values, targets)`. Values stay bit-packed as the daemon wrote them,
`3 * n + 2` bits per point, and are expanded on the device by
`model.unpack_bits`. Each epoch is copied out of shared memory before the next
one is asked for.

`client.py` holds the process, commands and shared-memory segment. `__init__.py`
is what the rest of `src/` imports, and it says nothing about either. Before
the first epoch, unknown training entries are streamed through `bootstrap.py`
in bounded chunks. The daemon samples every primary-input restriction; for an
`M_1` entry it also samples the two subsets reached by splitting on `f`.
Python evaluates those rows with already trained models and sends the combined
targets back, after which unknown and exact entries are served alike.

Each epoch resamples every case at inputs of its own, so no two epochs train on
the same points. `DatasetSizes.unknown_train` counts entries whose targets were
reconstructed during startup.

## train.py

`python -m src.train --model m1 --bitness 8`, or `--model m2` for the auxiliary
model of [`docs/data_m2.md`](../docs/data_m2.md); it is what
[`scripts/train/train_model.sh`](../scripts/train/train_model.sh) runs. The two
models differ only in the files the daemon opens -- one loop, one model class,
one set of targets. One epoch is one pass over the whole training file, freshly
sampled; `training.epochs` is the maximum duration, and the RMSE threshold
ends the run early once every score is below it on training and on validation.
`training.gradient_clip` caps the gradient norm, and zero leaves it alone: the
recursion is deep, and one spike in it undoes a run.

`seed` initializes PyTorch as well as daemon sampling. The plateau scheduler's
`min_lr` keeps optimization active after repeated reductions. Metrics include
the network and optimizer settings used for the run, and every epoch records
the error of each score on its own beside the RMSE over both.

Everything a run writes goes to `work_dir`, and nothing goes anywhere else:
`<tag>.pt` after every epoch, `<tag>.best.pt` whenever validation improves, and
`<tag>.metrics.json` behind them. So a killed run keeps what it reached, the
next run continues from `<tag>.pt`, and taking a finished model out of there is
done by hand. A continued run carries the earlier best over with it, so it
cannot overwrite `<tag>.best.pt` with something worse.

`data/` is read-only to training: the two files the daemon serves are all it
touches there.

For bitness above 12, models must be trained in dependency order. `M_2^n`
requires `<work_dir>/m2_<n-1>.best.pt`; `M_1^n` requires both
`m1_<n-1>.best.pt` and `m2_<n>.best.pt`. Missing prerequisite weights fail
immediately with the coordinate that must be trained first.

## model.py

`DeepSetPredictor` reads the same points down two channels and concatenates both
into `rho`, which returns the depth and size scores.

`phi` is the raw channel: input `(batch, batches * points_in_batch, point_dim)`
is split into `batches` groups, and one shared 2-layer MLP reads each group's
points flattened, with mean and max pooling over the groups behind it.

`psi` is the pooled channel. A point is three blocks -- the `n` inputs, the
`n + 1` values of `g`, the `n + 1` values of `h` -- and `statistics` averages
every product of two different blocks over the points: `x_i * h_j`, `x_i * g_j`,
`h_i * g_j`, in that order. Each block is against `h` or `g`, so all of them are
`n + 1` wide and they stack into one `statistics_shape(n) = (3n + 1, n + 1)`
matrix. `psi` is shared over its `3n + 1` rows, and mean and max pooling over
them follow it, so `rho` takes `2 * phi_out + 2 * psi_out` at its input.

Products within a block are not taken, so nothing here is `g` against itself:
the edge terms `g(x) * g(x^e_i)` that measure the influence of bit `i` are
absent, and so are the plain block averages.

The pooled channel is invariant to point order, and permuting the input bits
permutes each of its blocks along both axes. Pooling makes `phi` blind to the
order of the batches, but not to the order of the points inside one, so the
model as a whole has neither property.

`unpack_bits` expands packed rows into ±1 float32 (little-endian, bit `b` ->
`2*b - 1`) on the device. `forward` takes the packed uint8 rows the daemon
serves and calls `unpacked` on them itself; `statistics` takes the expanded
points, shaped `(batch, batches * points_in_batch, point_dim)`.

## ordered.py

`model.architecture: ordered` selects `OrderedRestrictionPredictor`, the
default. It runs the decision-tree recursion -- combine the maximum and sum of
two child states, minimum-pool over the choice of the next query, zero for a
constant or empty restriction, one learned head for both scores -- over a
restriction lattice built from the sampled points rather than from a truth
table.

Which restrictions it keeps is the whole idea. Fix `orders` random orderings of
the query coordinates; a restriction is retained when its fixed set is a prefix
of some ordering with at most a few coordinates postponed. One ordering admits
`O(d^3)` such sets for `d` queryable coordinates, and a set holds at most one
state per sampled cell, so the lattice is `O(orders * points * d^3)` states
instead of `3^d`. `orders` is a constant, independent of the bitness.

Topology is built from Python integer bitsets over the sampled cells: nothing
of size `2^d` or `3^d` is ever allocated. Constant and duplicate or
complemented sampled query columns describe the same query and are collapsed
first; singleton groups end the recursion; only the most recent geometry is
cached, so a run whose cases share one sampled input set builds it once. The
CPU builds the indices and PyTorch runs the recurrence and its gradients on the
GPU. The predictor reads the sampled `g(x)` and `f(x)` values only, not the
neighbour columns, and the sample need not cover the cube.

More orderings mean more coverage and more cost. **At bitness 8, 64 orderings
retain the complete `3^9` lattice**, so the accuracy measured there is the
exhaustive recursion's, reached through a construction that obeys a polynomial
bound rather than through a sparse one; four orderings retain 76%, and accuracy
falls away with them. [`docs/experiments_log.md`](../docs/experiments_log.md)
has the measurements.
