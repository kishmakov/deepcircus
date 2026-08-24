# conf

Run configuration. What each parameter means is documented next to the
parameter itself, as a comment in the file that declares it.

| File | What it configures |
| --- | --- |
| `preparation.yaml` | offline data generation; read by [`prepare_offline_train_data.sh`](../scripts/preparation/prepare_offline_train_data.sh) |
| `train.conf` | a training run; loaded by `load_train_config` in [`src/config.py`](../src/config.py) |
| `bench.conf` | a short benchmark run; `train.conf`'s schema without the `model` and `optimizer` sections |
