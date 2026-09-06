# conf

Run configuration. What each parameter means is documented next to the
parameter itself, as a comment in the file that declares it.

| File | What it configures |
| --- | --- |
| `prep.yaml` | offline data generation; read by [`generate_train_data.sh`](../scripts/prep/generate_train_data.sh) |
| `train_sampled.yaml` | the earlier `DeepSetPredictor`, kept with its own work directory; pass with `--config` to `src.train` |
| `train.yaml` | one offline training run -- a model at a bitness; loaded by `load_train_config` in [`src/config.py`](../src/config.py), read by [`train_model.sh`](../scripts/train/train_model.sh) |

The default selects the `ordered` architecture, whose lattice is
`O(orders * points * bitness^3)` states. What it reaches on `m1 8`, and the
measurements behind it, are in
[`docs/experiments_log.md`](../docs/experiments_log.md).
