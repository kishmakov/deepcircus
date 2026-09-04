# conf

Run configuration. What each parameter means is documented next to the
parameter itself, as a comment in the file that declares it.

| File | What it configures |
| --- | --- |
| `preparation.yaml` | offline data generation; read by [`generate_train_data.sh`](../scripts/prep/generate_train_data.sh) |
| `train.yaml` | one offline training run -- a model at a bitness; loaded by `load_train_config` in [`src/config.py`](../src/config.py), read by [`train_model.sh`](../scripts/train/train_model.sh) |
