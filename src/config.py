from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yaml import safe_load


ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = ROOT / "conf"
CONFIG_NAME = "train.yaml"

MODELS = ("m1", "m2")
MIN_BITNESS = 8
MAX_BITNESS = 255


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
    phi_hidden: int
    phi_out: int
    rho_hidden: int
    dropout: float


@dataclass(frozen=True)
class OptimizerConfig:
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
    assert MIN_BITNESS <= bitness <= MAX_BITNESS, bitness
    config_path = path or CONF_DIR / CONFIG_NAME
    with open(config_path, encoding="utf-8") as f:
        container = safe_load(f)
    assert isinstance(container, dict), config_path

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
    assert 0 <= config.seed <= 0xFFFFFFFFFFFFFFFF, config.seed
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
    # The daemon payload stores both dimensions as uint16_t.
    assert 1 < sampling.batches <= 0xFFFF, sampling
    assert 1 < sampling.points_in_batch <= 0xFFFF, sampling
    assert sampling.points_in_batch & (sampling.points_in_batch - 1) == 0, sampling
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
    model = ModelConfig(
        phi_hidden=int(raw["phi_hidden"]),
        phi_out=int(raw["phi_out"]),
        rho_hidden=int(raw["rho_hidden"]),
        dropout=float(raw["dropout"]),
    )
    assert model.phi_hidden > 0, model
    assert model.phi_out > 0, model
    assert model.rho_hidden > 0, model
    assert 0.0 <= model.dropout < 1.0, model
    return model


def _optimizer(raw: Any) -> OptimizerConfig:
    scheduler = raw["scheduler"]
    optimizer = OptimizerConfig(
        lr=float(raw["lr"]),
        scheduler_patience=int(scheduler["patience"]),
        scheduler_factor=float(scheduler["factor"]),
    )
    assert optimizer.lr > 0.0, optimizer
    assert optimizer.scheduler_patience >= 0, optimizer
    assert 0.0 < optimizer.scheduler_factor < 1.0, optimizer
    return optimizer
