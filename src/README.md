# src

The Python side: the config, the daemon client, the model, and the training
loop. Nothing here generates input bits or evaluates functions -- that is
[`cpp/`](../cpp/README.md)'s job, and [`daemon/`](daemon/) is the only place
that knows how to ask for the result.

| File | What it owns |
| --- | --- |
| `config.py` | `conf/train.yaml` for one `(model, bitness)` run |
| `daemon/` | everything about where the data comes from |
| `train.py` | the training loop of one run, and its weights and metrics |
| `model.py` | `DeepSetPredictor` and the on-device unpacking |

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

`client.py` holds the process, the two commands and the segment. `__init__.py`
is what the rest of `src/` imports, and it says nothing about either.

Each epoch resamples every case at inputs of its own, so no two epochs train on
the same points. `DatasetSizes.skipped` counts the entries the daemon left
behind because their target is the unknown marker.

## train.py

`python -m src.train --model m1 --bitness 8`, which is what
[`scripts/train/train_model.sh`](../scripts/train/train_model.sh) runs. One
epoch is one pass over the whole training file, freshly sampled;
`training.epochs` is the only duration knob, with the RMSE threshold able to end
the run early.

Everything a run writes goes to `work_dir`, and nothing goes anywhere else:
`<tag>.pt` after every epoch, `<tag>.best.pt` whenever validation improves, and
`<tag>.metrics.json` behind them. So a killed run keeps what it reached, the
next run continues from `<tag>.pt`, and taking a finished model out of there is
done by hand. A continued run carries the earlier best over with it, so it
cannot overwrite `<tag>.best.pt` with something worse.

`data/` is read-only to training: the two files the daemon serves are all it
touches there.

## model.py

`DeepSetPredictor`: input `(batch, batches * points_in_batch, point_dim)` is
split into `batches` groups, each processed by its own dedicated 2-layer `phi`
MLP (an `nn.ModuleList`, no weight sharing), the outputs concatenated into a
`batches * phi_out` vector and fed to a single `rho` head.

`unpack_bits` expands packed rows into ±1 float32 (little-endian, bit `b` ->
`2*b - 1`) on the device. `forward` unpacks uint8 input itself; float32 input
passes through.
