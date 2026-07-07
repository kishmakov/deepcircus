from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from json import dump, load
from os import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from omegaconf import OmegaConf


BITNESS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "bitness.conf"
SNAPSHOT_NAME = "bitness_snapshot.json"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    phi_hidden: int
    phi_out: int
    rho_hidden: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    iterations_from: int
    iterations_to: int
    epochs: int
    batch_size: int
    rmse_threshold: float
    train_samples: int
    validation_samples: int
    points_per_sample: int
    bitness_from: int
    bitness_to: int
    lr: float
    scheduler_patience: int
    scheduler_factor: float
    seed: int
    model_dir: Path


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    bitness_from: int
    bitness_to: int
    training: TrainingConfig
    model: ModelConfig
    _snapshot: dict[str, Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _pending_iteration_metrics: dict[int, list[dict[str, Any]]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def snapshot(self) -> dict[str, Any]:
        if self._snapshot is None:
            object.__setattr__(self, "_snapshot", load_or_create_snapshot(self))
        assert self._snapshot is not None, self
        return self._snapshot

    def is_done(self) -> bool:
        return self.last_completed_iteration() >= self.training.iterations_to

    def iterations_range(self) -> range:
        first = max(self.training.iterations_from, self.last_completed_iteration() + 1)
        return range(first, self.training.iterations_to + 1)

    def bitness_range(self) -> range:
        return range(self.bitness_from, self.bitness_to + 1)

    def epochs_range(self) -> range:
        return range(1, self.training.epochs + 1)

    def rmse_threshold(self) -> float:
        return self.training.rmse_threshold

    def has_completed_iteration(self) -> bool:
        return self.last_completed_iteration() >= self.training.iterations_from

    def last_completed_iteration(self) -> int:
        return int(self.snapshot()["progress"]["last_completed_iteration"])

    def record_model_trained(
            self,
            bitness: int,
            iteration: int,
            epoch: int,
            train_rmse: float,
            val_rmse: float,
    ) -> None:
        snapshot = self.snapshot()
        metrics = self._pending_iteration_metrics.setdefault(iteration, [])
        completed_steps = int(snapshot["progress"]["global_step"])
        pending_steps = sum(
            len(pending) for pending in self._pending_iteration_metrics.values()
        )
        metrics.append(
            {
                "global_step": completed_steps + pending_steps + 1,
                "iteration": iteration,
                "bitness": bitness,
                "epoch": epoch,
                "train_rmse": float(train_rmse),
                "val_rmse": float(val_rmse),
            }
        )
        snapshot["pending_iteration"] = {
            "iteration": iteration,
            "metrics": list(metrics),
        }
        save_bitness_snapshot(snapshot, snapshot_path(self))

    def record_iteration_trained(self, iteration: int) -> None:
        metrics = self._pending_iteration_metrics.pop(iteration)
        assert len(metrics) == len(self.bitness_range()), metrics
        snapshot = self.snapshot()
        assert iteration == int(snapshot["progress"]["last_completed_iteration"]) + 1, (
            iteration,
            snapshot["progress"]["last_completed_iteration"],
        )
        global_step = int(snapshot["progress"]["global_step"]) + len(metrics)
        snapshot["metrics"] = metrics
        snapshot["progress"] = {
            "stage": "done" if iteration >= self.training.iterations_to else "train",
            "global_step": global_step,
            "last_completed_iteration": iteration,
            "iteration": iteration,
        }
        snapshot.pop("pending_iteration", None)
        save_bitness_snapshot(snapshot, snapshot_path(self))
        prune_bitness_weights(self, iteration)
        notify_iteration_trained(iteration, metrics)

    def weights_path(self, bitness: int) -> Path:
        return self.training.model_dir / f"bitness_b{bitness:02d}.pt"


def load_bitness_config() -> Config:
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

    return Config(
        raw=raw,
        bitness_from=training.bitness_from,
        bitness_to=training.bitness_to,
        training=training,
        model=model,
    )


def build_training_config(raw: dict[str, Any]) -> TrainingConfig:
    training = raw["training"]
    optimizer = raw["optimizer"]
    scheduler = optimizer["scheduler"]
    assert scheduler["name"] == "reduce_lr_on_plateau", scheduler["name"]
    return TrainingConfig(
        iterations_from=int(training["iterations_from"]),
        iterations_to=int(training["iterations_to"]),
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        rmse_threshold=float(training["rmse_threshold"]),
        train_samples=int(training["train_samples"]),
        validation_samples=int(training["validation_samples"]),
        points_per_sample=int(training["points_per_sample"]),
        bitness_from=int(training["bitness_from"]),
        bitness_to=int(training["bitness_to"]),
        lr=float(optimizer["lr"]),
        scheduler_patience=int(scheduler["patience"]),
        scheduler_factor=float(scheduler["factor"]),
        seed=int(training["seed"]),
        model_dir=Path(str(training["model_dir"])),
    )


def build_model_config(raw: dict[str, Any]) -> ModelConfig:
    model = raw["model"]
    return ModelConfig(
        name=str(model["name"]),
        phi_hidden=int(model["phi_hidden"]),
        phi_out=int(model["phi_out"]),
        rho_hidden=int(model["rho_hidden"]),
        dropout=float(model["dropout"]),
    )


def load_or_create_snapshot(config: Config) -> dict[str, Any]:
    config.training.model_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(config)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            snapshot = normalize_bitness_snapshot(load(f), config)
        assert snapshot["config"] == config.raw, (
            "Cannot resume bitness training with changed config",
            path,
        )
        save_bitness_snapshot(snapshot, path)
        prune_bitness_weights(
            config,
            int(snapshot["progress"]["last_completed_iteration"]),
        )
        print(f"resuming bitness training from {path}")
        return snapshot

    snapshot = {
        "config": config.raw,
        "progress": {
            "stage": "train",
            "global_step": 0,
            "last_completed_iteration": initial_last_completed_iteration(config),
        },
        "metrics": [],
    }
    save_bitness_snapshot(snapshot, path)
    return snapshot


def normalize_bitness_snapshot(
        snapshot: dict[str, Any], config: Config
) -> dict[str, Any]:
    progress = snapshot["progress"]
    if "last_completed_iteration" in progress:
        last_completed_iteration = int(progress["last_completed_iteration"])
        completed_iteration = snapshot.get("completed_iteration") or {}
        metrics = list(snapshot.get("metrics") or completed_iteration.get("metrics", []))
    else:
        completed_iteration = snapshot.get("completed_iteration")
        if completed_iteration is None:
            last_completed_iteration = initial_last_completed_iteration(config)
            metrics = []
        else:
            last_completed_iteration = int(completed_iteration["iteration"])
            metrics = list(completed_iteration["metrics"])

    assert last_completed_iteration <= config.training.iterations_to, (
        last_completed_iteration,
        config.training.iterations_to,
    )
    assert last_completed_iteration >= initial_last_completed_iteration(config), (
        last_completed_iteration,
        initial_last_completed_iteration(config),
    )

    global_step = int(progress.get("global_step", 0))
    return {
        "config": snapshot["config"],
        "progress": {
            "stage": "done"
            if last_completed_iteration >= config.training.iterations_to
            else "train",
            "global_step": global_step,
            "last_completed_iteration": last_completed_iteration,
        },
        "metrics": metrics,
    }


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


def notify_iteration_trained(iteration: int, metrics: list[dict[str, Any]]) -> None:
    lines = [f"bitness iteration={iteration:03d} trained"]
    for metric in sorted(metrics, key=lambda item: int(item["bitness"])):
        lines.append(
            f"bitness={int(metric['bitness']):02d} "
            f"last epoch={int(metric['epoch']):03d} "
            f"rmse={float(metric['rmse']):.6f}"
        )

    url = f"http://{os.environ['GC_VM_IP']}:{os.environ['HEREYOUGOBOT_PORT']}/notify"
    request = urllib.request.Request(
        url,
        data="\n".join(lines).encode("utf-8"),
        headers={"Content-Type": "text/plain"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except OSError as error:
        print(f"telegram notification failed: {error}")


def initial_last_completed_iteration(config: Config) -> int:
    return config.training.iterations_from - 1


def prune_bitness_weights(config: Config, keep_iteration: int) -> None:
    if keep_iteration < config.training.iterations_from:
        return
    keep_suffix = f"_i{keep_iteration:03d}.pt"
    for weights_path in config.training.model_dir.glob("bitness_b*_i*.pt"):
        if weights_path.name.endswith(keep_suffix):
            continue
        weights_path.unlink()


def snapshot_path(config: Config) -> Path:
    return config.training.model_dir / SNAPSHOT_NAME


def model_key(bitness: int, iteration: int) -> str:
    return f"i{iteration:03d}_b{bitness:02d}"
