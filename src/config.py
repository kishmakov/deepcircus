from __future__ import annotations

from dataclasses import dataclass
from json import dump, load
from os import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from omegaconf import OmegaConf


BITNESS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "bitness.conf"
SNAPSHOT_NAME = "bitness_snapshot.json"


@dataclass(frozen=True)
class BitnessModelConfig:
    name: str
    n_points: int
    phi_hidden: int
    phi_out: int
    rho_hidden: int
    dropout: float


@dataclass(frozen=True)
class BitnessTrainingConfig:
    iterations_from: int
    iterations_to: int
    epochs: int
    batch_size: int
    rmse_threshold: float
    samples_per_model: int
    bitness_from: int
    bitness_to: int
    input_dim: int
    lr: float
    seed: int
    model_dir: Path


@dataclass(frozen=True)
class BitnessExperimentConfig:
    raw: dict[str, Any]
    bitness_from: int
    bitness_to: int
    training: BitnessTrainingConfig
    model: BitnessModelConfig


def load_bitness_config() -> BitnessExperimentConfig:
    raw = OmegaConf.to_container(OmegaConf.load(BITNESS_CONFIG_PATH), resolve=True)
    assert isinstance(raw, dict), raw

    training = build_training_config(raw)
    model = build_model_config(raw)

    assert training.bitness_from <= training.bitness_to, (
        training.bitness_from,
        training.bitness_to,
    )
    assert training.iterations_from <= training.iterations_to, (
        training.iterations_from,
        training.iterations_to,
    )

    return BitnessExperimentConfig(
        raw=raw,
        bitness_from=training.bitness_from,
        bitness_to=training.bitness_to,
        training=training,
        model=model,
    )


def build_training_config(raw: dict[str, Any]) -> BitnessTrainingConfig:
    training = raw["training"]
    optimizer = raw["optimizer"]
    return BitnessTrainingConfig(
        iterations_from=int(training["iterations_from"]),
        iterations_to=int(training["iterations_to"]),
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        rmse_threshold=float(training["rmse_threshold"]),
        samples_per_model=int(training["samples_per_model"]),
        bitness_from=int(training["bitness_from"]),
        bitness_to=int(training["bitness_to"]),
        input_dim=int(training["input_dim"]),
        lr=float(optimizer["lr"]),
        seed=int(training["seed"]),
        model_dir=Path(str(training["model_dir"])),
    )


def build_model_config(raw: dict[str, Any]) -> BitnessModelConfig:
    model = raw["model"]
    return BitnessModelConfig(
        name=str(model["name"]),
        n_points=int(model["n_points"]),
        phi_hidden=int(model["phi_hidden"]),
        phi_out=int(model["phi_out"]),
        rho_hidden=int(model["rho_hidden"]),
        dropout=float(model["dropout"]),
    )


def load_or_create_bitness_snapshot(
        config: BitnessExperimentConfig,
) -> dict[str, Any]:
    config.training.model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = bitness_snapshot_path(config)
    if snapshot_path.exists():
        snapshot = load_bitness_snapshot(snapshot_path)
        assert snapshot["config"] == config.raw, (
            "Cannot resume bitness training with changed config",
            snapshot_path,
        )
        print(f"resuming bitness training from {snapshot_path}")
        return snapshot

    snapshot = {
        "config": config.raw,
        "progress": {
            "stage": "train",
            "global_step": 0,
        },
        "completed": {},
        "metrics": [],
    }
    save_bitness_snapshot(snapshot, snapshot_path)
    return snapshot


def load_bitness_snapshot(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return load(f)


def save_bitness_snapshot(snapshot: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=".tmp-",
        suffix=".json",
        delete=False,
    ) as f:
        dump(snapshot, f, indent=2)
        f.write("\n")
        tmp_path = f.name
    replace(tmp_path, path)


def is_model_trained(
        snapshot: dict[str, Any],
        bitness: int,
        iteration: int,
) -> bool:
    return model_key(bitness, iteration) in snapshot["completed"]


def record_model_trained(
        snapshot: dict[str, Any],
        config: BitnessExperimentConfig,
        bitness: int,
        iteration: int,
        epoch: int,
        rmse: float,
        weights_path: Path,
) -> None:
    global_step = int(snapshot["progress"]["global_step"]) + 1
    metric = {
        "global_step": global_step,
        "iteration": iteration,
        "bitness": bitness,
        "epoch": epoch,
        "rmse": float(rmse),
        "weights_path": str(weights_path),
    }
    snapshot["metrics"].append(metric)
    snapshot["completed"][model_key(bitness, iteration)] = metric
    snapshot["progress"] = {
        "stage": "done" if is_training_complete(snapshot, config) else "train",
        "global_step": global_step,
        "iteration": iteration,
        "bitness": bitness,
    }
    save_bitness_snapshot(snapshot, bitness_snapshot_path(config))


def is_training_complete(
        snapshot: dict[str, Any],
        config: BitnessExperimentConfig,
) -> bool:
    iterations = (
        config.training.iterations_to
        - config.training.iterations_from
        + 1
    )
    bitnesses = config.bitness_to - config.bitness_from + 1
    return len(snapshot["completed"]) >= iterations * bitnesses


def bitness_snapshot_path(config: BitnessExperimentConfig) -> Path:
    return config.training.model_dir / SNAPSHOT_NAME


def bitness_weights_path(
        config: BitnessExperimentConfig,
        bitness: int,
        iteration: int,
) -> Path:
    return config.training.model_dir / f"bitness_b{bitness:02d}_i{iteration:03d}.pt"


def model_key(bitness: int, iteration: int) -> str:
    return f"i{iteration:03d}_b{bitness:02d}"
