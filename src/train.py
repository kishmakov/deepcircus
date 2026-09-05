from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from os import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.daemon import DatasetSizes, TrainingData, open_training_data
from src.model import DEVICE, make_predictor
from src.config import MODELS, TrainConfig, load_train_config


# Every console line of this module is marked, so it stands apart from the
# daemon's own `[server]` lines it shares the terminal with. Green through
# ANSI SGR, like the daemon's prefix.
_CONSOLE_PREFIX = "\033[32m[train.py]\033[0m "


def report(message: str) -> None:
    print(f"{_CONSOLE_PREFIX}{message}")


@dataclass
class EpochMetrics:
    epoch: int
    train_rmse: float
    validation_rmse: float
    train_depth_rmse: float
    train_size_rmse: float
    validation_depth_rmse: float
    validation_size_rmse: float
    learning_rate: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trains one model at one bitness on its offline data.")
    parser.add_argument("--model", required=True, choices=MODELS, help="which model to train")
    parser.add_argument("--bitness", required=True, type=int, help="arity of the functions it is trained on")
    parser.add_argument("--config", type=Path, default=None, help="config file, default conf/train.yaml")
    arguments = parser.parse_args(argv)

    config = load_train_config(arguments.model, arguments.bitness, arguments.config)
    run_training(config)


def run_training(config: TrainConfig) -> None:
    config.work_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)

    with open_training_data(config) as data:
        report_dataset(config, data.sizes)
        validation_loader = loader(config, *data.validation(), shuffle=False)

        model = build_model(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=config.optimizer.scheduler_patience,
            factor=config.optimizer.scheduler_factor,
            min_lr=config.optimizer.scheduler_min_lr,
        )

        metrics: list[EpochMetrics] = []
        best_rmse = previous_best(config)
        progress = tqdm(
            range(1, config.training.epochs + 1), desc=f"{_CONSOLE_PREFIX}{config.tag}", unit="epoch"
        )
        for epoch in progress:
            # Every epoch is the whole training file, sampled at inputs of its
            # own -- `epochs` is the only thing that decides how long this runs.
            train_scores = train_epoch(model, optimizer, loader(config, *data.epoch(epoch), shuffle=True))
            validation_scores = evaluate(model, validation_loader)
            train_rmse = combined_rmse(train_scores)
            validation_rmse = combined_rmse(validation_scores)
            scheduler.step(validation_rmse)
            metrics.append(
                EpochMetrics(epoch, train_rmse, validation_rmse, *train_scores, *validation_scores,
                             float(optimizer.param_groups[0]["lr"]))
            )
            progress.set_postfix(train=f"{train_rmse:.4f}", val=f"{validation_rmse:.4f}")
            record_epoch(config, model, metrics, validation_rmse, best_rmse)
            best_rmse = min(best_rmse, validation_rmse)

            if train_rmse < config.training.rmse_threshold:
                report(f"train rmse {train_rmse:.6f} below threshold {config.training.rmse_threshold}")
                break

    report(f"best validation rmse {best_rmse:.6f}")
    report(f"weights: {config.best_checkpoint_path()}")
    report(f"metrics: {config.metrics_path()}")


def record_epoch(
        config: TrainConfig,
        model: nn.Module,
        metrics: list[EpochMetrics],
        validation_rmse: float,
        best_rmse: float,
) -> None:
    """Writes the epoch out, all of it inside `work_dir`.

    Every epoch is saved, so a run that is killed keeps what it reached; the
    weights the run is finally judged on are the ones that scored best against
    validation, not the ones the last epoch happened to end on.
    """
    torch.save(model.state_dict(), config.checkpoint_path())
    if validation_rmse < best_rmse:
        shutil.copyfile(config.checkpoint_path(), config.best_checkpoint_path())
    save_metrics(config, metrics, min(best_rmse, validation_rmse))


def loader(config: TrainConfig, values: np.ndarray, targets: np.ndarray, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(values), torch.from_numpy(targets)),
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=True,
    )


def report_dataset(config: TrainConfig, sizes: DatasetSizes) -> None:
    report(
        f"{config.tag}: {sizes.train_entries} train cases, {sizes.validation_entries} validation cases, "
        f"{config.sampling.points} points of {sizes.point_dim} bits each"
    )
    report(
        f"one epoch is {sizes.epoch_bytes(config.sampling.points):,} bytes of packed values and targets"
    )
    if sizes.unknown_train:
        report(f"reconstructed {sizes.unknown_train} targets through trained reduction models")


def build_model(config: TrainConfig) -> nn.Module:
    model = make_predictor(config)
    # Where the last run of this coordinate left off, if it left anything.
    checkpoint_path = config.checkpoint_path()
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
        report(f"continuing from {checkpoint_path}")
    model.to(DEVICE)
    return model


def previous_best(config: TrainConfig) -> float:
    """The best validation RMSE any earlier run of this coordinate reached.

    Carried over so a run that continues from those weights and never beats them
    leaves `<tag>.best.pt` alone, instead of overwriting it with its own worse
    best. The epoch metrics still describe this run alone.
    """
    if not config.metrics_path().exists():
        return float("inf")

    with open(config.metrics_path(), encoding="utf-8") as f:
        best = float(json.load(f)["best_validation_rmse"])
    # The weights that scored it are the ones this run has to beat, so they have
    # to still be there.
    assert config.best_checkpoint_path().exists(), config.best_checkpoint_path()
    report(f"previous best validation rmse {best:.6f}")
    return best


def train_epoch(
        model: nn.Module, optimizer: torch.optim.Optimizer, loader: DataLoader
) -> tuple[float, float]:
    model.train()
    squared_error_sum = torch.zeros(2, dtype=torch.float64, device=DEVICE)
    count = 0

    for xb, yb in loader:
        # xb stays packed uint8; the model unpacks it on-device.
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        prediction = model(xb)
        squared_errors = (prediction - yb).square()
        loss = squared_errors.mean()
        loss.backward()
        optimizer.step()

        squared_error_sum += squared_errors.detach().sum(dim=0, dtype=torch.float64)
        count += len(yb)

    assert count > 0, count
    return per_score_rmse(squared_error_sum, count)


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    squared_error_sum = torch.zeros(2, dtype=torch.float64, device=DEVICE)
    count = 0

    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            squared_error_sum += (model(xb) - yb).square().sum(dim=0, dtype=torch.float64)
            count += len(yb)

    assert count > 0, count
    return per_score_rmse(squared_error_sum, count)


def per_score_rmse(squared_error_sum: torch.Tensor, count: int) -> tuple[float, float]:
    depth, size = (squared_error_sum / count).sqrt().tolist()
    return depth, size


def combined_rmse(scores: tuple[float, float]) -> float:
    """What the two per-score errors are together -- the RMSE over both columns."""
    depth, size = scores
    return ((depth**2 + size**2) / 2) ** 0.5


def save_metrics(config: TrainConfig, metrics: list[EpochMetrics], best_rmse: float) -> None:
    record = {
        "model": config.model_name,
        "bitness": config.bitness,
        "seed": config.seed,
        "network": asdict(config.model),
        "optimizer": asdict(config.optimizer),
        "sampling": {
            "batches": config.sampling.batches,
            "points_in_batch": config.sampling.points_in_batch,
        },
        "training": {
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "rmse_threshold": config.training.rmse_threshold,
        },
        "best_validation_rmse": best_rmse,
        "epochs": [entry.as_dict() for entry in metrics],
    }
    path = config.metrics_path()
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=".tmp-", suffix=".json", delete=False
    ) as f:
        json.dump(record, f, indent=2)
        f.write("\n")
        temporary = f.name
    replace(temporary, path)


if __name__ == "__main__":
    main()
