from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = ROOT / "conf"
CONFIG_NAME = "train.yaml"

MODELS = ("m1", "m2")


@dataclass(frozen=True)
class SamplingConfig:
    batches: int
    points_in_batch: int

    @property
    def points(self) -> int:
        return self.batches * self.points_in_batch


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    rmse_threshold: float


@dataclass(frozen=True)
class ModelConfig:
    name: str
    phi_hidden: int
    phi_out: int
    rho_hidden: int
    dropout: float


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    scheduler_patience: int
    scheduler_factor: float


@dataclass(frozen=True)
class TrainConfig:
    """One run: the model named by `model` at `bitness`, and how to train it."""

    model_name: str
    bitness: int
    data_dir: Path  # read-only: the two files the run trains on
    work_dir: Path  # written: everything the run produces
    seed: int
    sampling: SamplingConfig
    training: TrainingConfig
    model: ModelConfig
    optimizer: OptimizerConfig

    @property
    def tag(self) -> str:
        return f"{self.model_name}_{self.bitness:02d}"

    @property
    def point_dim(self) -> int:
        # The daemon's serving::PointDim: input bits, then g's value and its
        # single-bit flips, then the same for the entry's second function --
        # M_1's f, or the indicator of M_2's subset.
        return 3 * self.bitness + 2

    def train_path(self) -> Path:
        return self.data_dir / f"{self.tag}.train"

    def validation_path(self) -> Path:
        return self.data_dir / f"{self.tag}.val"

    # Everything a run writes, and it writes nothing outside `work_dir`: the
    # epoch just finished, the best epoch so far, and the metrics of every epoch
    # behind them. Taking a finished model out of there is done by hand.
    def checkpoint_path(self) -> Path:
        return self.work_dir / f"{self.tag}.pt"

    def best_checkpoint_path(self) -> Path:
        return self.work_dir / f"{self.tag}.best.pt"

    def metrics_path(self) -> Path:
        return self.work_dir / f"{self.tag}.metrics.json"


def load_train_config(
        model_name: str,
        bitness: int,
        path: Path | None = None,
) -> TrainConfig:
    assert model_name in MODELS, model_name
    raw = OmegaConf.load(path or CONF_DIR / CONFIG_NAME)
    assert isinstance(raw, DictConfig), path
    container = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(container, dict), path

    config = TrainConfig(
        model_name=model_name,
        bitness=bitness,
        data_dir=_directory(str(container["data_dir"])),
        work_dir=_directory(str(container["work_dir"])),
        seed=int(container["seed"]),
        sampling=_sampling(container["sampling"]),
        training=_training(container["training"]),
        model=_model(container["model"]),
        optimizer=_optimizer(container["optimizer"]),
    )
    assert config.train_path().exists(), config.train_path()
    assert config.validation_path().exists(), config.validation_path()
    return config


def _directory(value: str) -> Path:
    # Relative to the repository, so a run reaches the same `data/` from
    # wherever it was started.
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sampling(raw: Any) -> SamplingConfig:
    sampling = SamplingConfig(
        batches=int(raw["batches"]),
        points_in_batch=int(raw["points_in_batch"]),
    )
    # The daemon's own assertions, checked here so a typo fails before a server
    # is spawned to fail on it.
    assert sampling.batches > 1, sampling
    assert sampling.points_in_batch & (sampling.points_in_batch - 1) == 0, sampling
    assert sampling.points_in_batch > 0, sampling
    return sampling


def _training(raw: Any) -> TrainingConfig:
    training = TrainingConfig(
        epochs=int(raw["epochs"]),
        batch_size=int(raw["batch_size"]),
        rmse_threshold=float(raw["rmse_threshold"]),
    )
    assert training.epochs > 0, training
    assert training.batch_size > 0, training
    assert training.rmse_threshold > 0.0, training
    return training


def _model(raw: Any) -> ModelConfig:
    assert raw["name"] == "deepset", raw["name"]
    return ModelConfig(
        name=str(raw["name"]),
        phi_hidden=int(raw["phi_hidden"]),
        phi_out=int(raw["phi_out"]),
        rho_hidden=int(raw["rho_hidden"]),
        dropout=float(raw["dropout"]),
    )


def _optimizer(raw: Any) -> OptimizerConfig:
    scheduler = raw["scheduler"]
    assert raw["name"] == "adam", raw["name"]
    assert scheduler["name"] == "reduce_lr_on_plateau", scheduler["name"]
    return OptimizerConfig(
        name=str(raw["name"]),
        lr=float(raw["lr"]),
        scheduler_patience=int(scheduler["patience"]),
        scheduler_factor=float(scheduler["factor"]),
    )
