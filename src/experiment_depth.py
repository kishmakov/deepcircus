from __future__ import annotations

from os.path import exists, join
from pathlib import Path
from typing import Any
import time

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import DEVICE, DeepSetPredictor, regression_metrics
from src.sampler import DepthSampler
from src.state_io import (
    DEFAULT_META_PATH,
    DEFAULT_MODEL_DIR,
    ensure_model_dir,
    load_experiment_meta,
    load_training_checkpoint,
    save_experiment_meta,
    save_training_checkpoint,
)


DEPTH_CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "depth.conf"


def run_experiment(generator) -> None:
    config = build_config()
    ensure_model_dir(config["model_dir"])
    meta = load_or_create_depth_meta(config)
    progress = meta["progress"]
    if progress["stage"] == "done":
        print(f"training already complete: {config['meta_path']}")
        return

    samples = config["sampler"]["samples"]
    states = {
        sample["key"]: build_training_state(
            build_sampler(generator, config, sample),
            config,
        )
        for sample in samples
    }

    train_epochs = int(config["optimiser"]["train_epochs"])
    for epoch in range(1, train_epochs + 1):
        for sample in samples:
            sample_key = sample["key"]
            loader_progress = meta["progress"]["loaders"][sample_key]
            if loader_progress["stage"] == "done" or loader_progress["epoch"] >= epoch:
                continue

            state = states[sample_key]
            train_metrics = train_regression_epoch(
                state["model"],
                state["train_loader"],
                state["optimizer"],
                state["scheduler"],
                desc=f"depth {sample_key} epoch {epoch}/{train_epochs}",
            )
            validation_metrics = evaluate_regression_loader(
                state["model"],
                state["validation_loader"],
            )
            parity_prediction = predict_single_depth(
                state["model"],
                state["parity_x"],
            )
            save_sample_metrics(
                meta,
                config,
                state["model"],
                sample,
                epoch,
                train_metrics,
                validation_metrics,
                parity_prediction,
            )

    if meta["progress"]["stage"] != "done":
        meta["progress"]["stage"] = overall_stage(meta)
        save_experiment_meta(meta, config["meta_path"])


def load_depth_config() -> dict[str, Any]:
    config = OmegaConf.load(DEPTH_CONFIG_PATH)
    return OmegaConf.to_container(config, resolve=True)


def build_config() -> dict[str, Any]:
    config = load_depth_config()
    config = {
        "model_dir": DEFAULT_MODEL_DIR,
        "meta_path": DEFAULT_META_PATH,
        **config,
    }
    config["checkpoint_paths"] = {
        sample["key"]: checkpoint_path_for(sample, config)
        for sample in config["sampler"]["samples"]
    }
    return config


def load_or_create_depth_meta(config: dict[str, Any]) -> dict[str, Any]:
    meta_path = config["meta_path"]
    if exists(meta_path):
        meta = load_experiment_meta(meta_path)
        assert_resume_config(meta["config"], config)
        ensure_depth_plot_series(meta, config)
        print(f"resuming from {meta_path}: progress={meta['progress']}")
        return meta

    meta = {
        "config": config,
        "progress": {
            "stage": "train",
            "global_step": 0,
            "loaders": initial_loader_progress(config),
        },
        "metrics": [],
        "series": build_plot_series(config),
    }
    save_experiment_meta(meta, meta_path)
    return meta


def ensure_depth_plot_series(meta: dict[str, Any], config: dict[str, Any]) -> None:
    plot_series = build_plot_series(config)
    if meta.get("series") == plot_series:
        return

    meta["series"] = plot_series
    meta.pop("plot_series", None)
    save_experiment_meta(meta, config["meta_path"])


def build_plot_series(config: dict[str, Any]) -> list[dict[str, Any]]:
    metric_lines = []
    parity_lines = []
    bitness = int(config["sampler"]["bitness"])
    for sample in config["sampler"]["samples"]:
        sample_key = sample["key"]
        label = sample["label"]
        metric_lines.extend(
            [
                {
                    "label": f"{label} train",
                    "where": {"loader": sample_key},
                    "x_key": "epoch",
                    "mae_key": "train_mae",
                    "rmse_key": "train_rmse",
                },
                {
                    "label": f"{label} validation",
                    "where": {"loader": sample_key},
                    "x_key": "epoch",
                    "mae_key": "validation_mae",
                    "rmse_key": "validation_rmse",
                },
            ]
        )
        parity_lines.extend(
            [
                {
                    "label": f"{label} prediction",
                    "where": {"loader": sample_key},
                    "x_key": "epoch",
                    "y_key": "parity_prediction",
                },
                {
                    "label": f"{label} target",
                    "where": {"loader": sample_key},
                    "x_key": "epoch",
                    "y_key": "parity_target",
                    "linestyle": "--",
                    "linewidth": 1,
                },
            ]
        )
    return [
        {
            "name": "depth_metrics",
            "title": "Depth prediction metrics",
            "x_label": "Epoch",
            "lines": metric_lines,
        },
        {
            "kind": "value",
            "name": "parity_depth",
            "title": f"Parity depth prediction, target depth {bitness}",
            "x_label": "Epoch",
            "y_label": "Predicted depth",
            "lines": parity_lines,
        },
    ]


def build_sampler(
    generator,
    config: dict[str, Any],
    sample: dict[str, Any],
) -> DepthSampler:
    sampler = config["sampler"]
    return DepthSampler(
        generator,
        int(sampler["bitness"]),
        int(sampler["seed"]),
        int(sampler["train_samples"]),
        int(sampler["validation_samples"]),
        sample["key"],
        sample["method"],
        int(sample["reps"]),
        batch_size=int(config["optimiser"]["batch_size"]),
    )


def build_training_state(
    sampler: DepthSampler,
    config: dict[str, Any],
) -> dict[str, Any]:
    sample_key = sampler.name
    train_loader, validation_loader = sampler.training_inputs()
    parity_x = sampler.parity_inputs()

    model_config = dict(config["model"])
    model_name = model_config.pop("name")
    model_config.pop("predict_batch_size", None)
    assert model_name == "deepset", model_name
    model = DeepSetPredictor(**sampler.model_params(), **model_config)

    checkpoint_path = checkpoint_path_for_name(sample_key, config)
    if exists(checkpoint_path):
        checkpoint = load_training_checkpoint(checkpoint_path, DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)

    optimizer_config = config["optimiser"]["optimizer"]
    assert optimizer_config["name"] == "adam", optimizer_config["name"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
    )
    scheduler_config = config["optimiser"]["scheduler"]
    assert scheduler_config["name"] == "reduce_lr_on_plateau", scheduler_config["name"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=int(scheduler_config["patience"]),
        factor=float(scheduler_config["factor"]),
    )
    return {
        "model": model,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "parity_x": parity_x,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def save_sample_metrics(
    meta: dict[str, Any],
    config: dict[str, Any],
    model: nn.Module,
    sample: dict[str, Any],
    epoch: int,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    parity_prediction: float,
) -> None:
    sample_key = sample["key"]
    global_step = int(meta["progress"]["global_step"]) + 1
    train_epochs = int(config["optimiser"]["train_epochs"])
    parity_target = float(config["sampler"]["bitness"])
    progress = {
        "stage": "train" if epoch < train_epochs else "done",
        "epoch": epoch,
    }
    metric = {
        "global_step": global_step,
        "epoch": epoch,
        "bitness": int(config["sampler"]["bitness"]),
        "loader": sample_key,
        "method": sample["method"],
        "reps": sample["reps"],
        "train_loss": train_metrics["loss"],
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mad"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_mae": validation_metrics["mad"],
        "parity_prediction": float(parity_prediction),
        "parity_target": parity_target,
        "parity_error": float(parity_prediction - parity_target),
    }
    meta.setdefault("metrics", []).append(metric)
    meta["progress"]["global_step"] = global_step
    meta["progress"]["loaders"][sample_key] = progress
    meta["progress"]["stage"] = overall_stage(meta)
    save_training_checkpoint(
        model,
        config,
        progress,
        checkpoint_stem_for(sample, config),
    )
    save_experiment_meta(meta, config["meta_path"])
    print(
        f"depth loader={sample_key} epoch={epoch:>3}  "
        f"train_rmse={metric['train_rmse']:.4f}  "
        f"train_mae={metric['train_mae']:.4f}  "
        f"validation_rmse={metric['validation_rmse']:.4f}  "
        f"validation_mae={metric['validation_mae']:.4f}  "
        f"parity={metric['parity_prediction']:.4f}/{parity_target:.0f}"
    )


def initial_loader_progress(config: dict[str, Any]) -> dict[str, dict[str, int | str]]:
    return {
        sample["key"]: {
            "stage": "train",
            "epoch": 0,
        }
        for sample in config["sampler"]["samples"]
    }


def overall_stage(meta: dict[str, Any]) -> str:
    if all(
        progress["stage"] == "done"
        for progress in meta["progress"]["loaders"].values()
    ):
        return "done"
    return "train"


def checkpoint_stem_for(sample: dict[str, Any], config: dict[str, Any]) -> str:
    return checkpoint_stem_for_name(sample["key"], config)


def checkpoint_stem_for_name(sample_key: str, config: dict[str, Any]) -> str:
    bitness = int(config["sampler"]["bitness"])
    return join(config["model_dir"], f"depth_{sample_key}_b{bitness:02d}")


def checkpoint_path_for(sample: dict[str, Any], config: dict[str, Any]) -> str:
    return checkpoint_stem_for(sample, config) + ".pt"


def checkpoint_path_for_name(sample_key: str, config: dict[str, Any]) -> str:
    return checkpoint_stem_for_name(sample_key, config) + ".pt"


def evaluate_regression_loader(
    model: nn.Module,
    loader: DataLoader,
) -> dict[str, float]:
    model.to(DEVICE)
    model.eval()

    predictions = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = torch.as_tensor(xb, dtype=torch.float32, device=DEVICE)
            yb = torch.as_tensor(yb, dtype=torch.float32, device=DEVICE).reshape(-1)
            predictions.append(model(xb).detach().cpu().numpy().ravel())
            targets.append(yb.cpu().numpy().ravel())

    return regression_metrics(
        np.concatenate(predictions).astype(np.float32),
        np.concatenate(targets).astype(np.float32),
    )


def predict_single_depth(model: nn.Module, x: np.ndarray) -> float:
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        xb = torch.as_tensor(
            x[None, :, :],
            dtype=torch.float32,
            device=DEVICE,
        )
        return float(model(xb).detach().cpu().numpy().ravel()[0])


def train_regression_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    *,
    desc: str = "depth",
) -> dict[str, float]:
    criterion = nn.MSELoss()

    model.train()
    epoch_start = time.perf_counter()
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    count = 0

    for xb, yb in tqdm(loader, total=len(loader), desc=desc):
        xb = torch.as_tensor(xb, dtype=torch.float32, device=DEVICE)
        yb = torch.as_tensor(yb, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
        optimizer.zero_grad()
        prediction = model(xb)
        loss = criterion(prediction, yb)
        loss.backward()
        optimizer.step()

        errors = (prediction.detach() - yb).reshape(-1)
        squared_error_sum += torch.sum(errors.square()).item()
        absolute_error_sum += torch.sum(torch.abs(errors)).item()
        count += len(xb)

    final_loss = float(squared_error_sum / count)
    scheduler.step(final_loss)
    epoch_elapsed = time.perf_counter() - epoch_start
    print(
        f"{desc}  "
        f"loss={final_loss:.4f}  "
        f"elapsed={epoch_elapsed:.2f}s  "
        f"device={DEVICE}"
    )
    return {
        "loss": final_loss,
        "rmse": float(np.sqrt(final_loss)),
        "mad": float(absolute_error_sum / count),
    }


def assert_resume_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    missing = object()
    for key, value in current.items():
        saved_value = saved.get(key, missing)
        saved_display = "<missing>" if saved_value is missing else repr(saved_value)
        assert saved_value == value, (
            f"Cannot resume depth training with changed {key}: "
            f"saved={saved_display}, current={value!r}"
        )
