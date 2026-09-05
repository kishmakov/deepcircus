# DeepCircus

Research code for learning decision-tree depth and size from sampled Boolean
functions. The project trains two related predictors:

- `M_1[g | f]` scores trees that compute `g` from primary inputs and one
  auxiliary function `f`;
- `M_2[g | X]` scores trees that compute `g` on a subset `X` of the input cube.

The C++ side generates deterministic offline datasets, samples them into packed
cases, and serves one freshly sampled epoch at a time. Python unpacks those
cases on the training device and trains the shared DeepSet architecture. The
offline storage formats are specified in [`docs/data_m1.md`](docs/data_m1.md)
and [`docs/data_m2.md`](docs/data_m2.md). Data preparation, target construction,
and training-time processing are described in [`docs/train.md`](docs/train.md),
with the mathematical definitions in the [paper](docs/paper.tex).

## Run the pipeline

Create the project environment and verify the implementation:

```bash
uv sync
scripts/test.sh
```

Prepare every coordinate declared in `conf/prep.yaml`:

```bash
scripts/prep/build_generator.sh
scripts/prep/generate_train_data.sh
```

Build the training daemon and train one coordinate:

```bash
scripts/train/build_server.sh
scripts/train/train_model.sh m2 8
scripts/train/train_model.sh m1 8
```

Bitnesses through 12 have targets produced by the exact solvers. Above 12,
train coordinates in increasing bitness: `m2 n` needs `m2 n-1`, then `m1 n`
needs `m1 n-1` and `m2 n`. Training reads its offline data from `data/`;
checkpoints and metrics go to the `work_dir` configured in `conf/train.yaml`,
and each finished run keeps a copy of its best weights in `data/`.

## Layout

| Directory | Responsibility |
| --- | --- |
| [`conf/`](conf/README.md) | data-preparation and training configuration |
| [`cpp/`](cpp/README.md) | generation, exact solvers, serving, and validation |
| [`data/circuits/`](data/circuits/README.md) | benchmark circuits; generated offline datasets live beside this directory under `data/` |
| `docs/` | paper, [storage formats](docs/data_m1.md), and [training data processing](docs/train.md) |
| [`scripts/`](scripts/README.md) | build, test, preparation, training, and plotting entry points |
| [`src/`](src/README.md) | model, daemon client, bootstrap, and training loop |
