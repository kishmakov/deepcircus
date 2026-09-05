# scripts

Thin entrypoints. Everything here is a wrapper: what a script knows is which
binary or module to run and which config to read it from -- the work itself
lives in [`cpp/`](../cpp/README.md) and [`src/`](../src/README.md), and the
parameters in [`conf/`](../conf/README.md).

## Preparing the offline data

| Script | What it does |
| --- | --- |
| [`prep/build_generator.sh`](prep/build_generator.sh) | builds `data_generator` and publishes it as `execs/data_generator` |
| [`prep/generate_train_data.sh`](prep/generate_train_data.sh) | fills `data/` with the `m{1,2}_<bitness>.{train,val}` files `conf/preparation.yaml` asks for; complete pairs are left alone on a re-run |

## Training

| Script | What it does |
| --- | --- |
| [`train/build_server.sh`](train/build_server.sh) | builds `offline_server` and publishes it as `execs/offline_server` |
| [`train/train_model.sh`](train/train_model.sh) | `train_model.sh m1 8`, `train_model.sh m2 8` -- trains either model at one bitness on its offline data, per `conf/train.yaml`, leaving the weights and metrics in that config's `work_dir`; above bitness 12, train `m2 n-1` before `m2 n`, then `m1 n-1` and `m2 n` before `m1 n` |
| [`docs/plot_metrics.sh`](docs/plot_metrics.sh) | plots the train and validation curves of every run in that `work_dir`, one PNG beside each `*.metrics.json`; `scale=log` puts RMSE on a logarithmic axis |

## Building and testing

| Script | What it does |
| --- | --- |
| [`build.sh`](build.sh) | configures and builds the C++ tree in Release; optional arguments name only the targets to build |
| [`test.sh`](test.sh) | builds the whole tree with ASan/UBSan and every assert alive, runs the suite and executable smoke checks, then trains tiny M1 and M2 runs for one epoch; extra arguments go to GoogleTest |
| [`clean.sh`](clean.sh) | removes build trees, published symlinks and LaTeX artifacts |

## Circuits and paper

| Script | What it does |
| --- | --- |
| [`dimensions.py`](dimensions.py) | rewrites `data/circuits/dimensions.txt` from the tracked AIG headers |
| [`docs/compile_paper.sh`](docs/compile_paper.sh) | compiles `docs/paper.pdf` with its bibliography |
