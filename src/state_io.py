"""Disk persistence for DeepCircus experiments."""

from __future__ import annotations

from collections.abc import Callable
from json import dump, load
from os import makedirs, replace
from os.path import dirname, exists, join
from tempfile import NamedTemporaryFile
from typing import Any

import torch


DEFAULT_MODEL_DIR = "/tmp/circus"
DEFAULT_META_PATH = join(DEFAULT_MODEL_DIR, "experiment.json")


def ensure_model_dir(model_dir: str) -> None:
    makedirs(model_dir, exist_ok=True)


def load_experiment_meta(meta_path: str) -> dict[str, Any]:
    with open(meta_path, encoding="utf-8") as f:
        return load(f)


def save_experiment_meta(meta: dict[str, Any], meta_path: str) -> None:
    _atomic_json_dump(meta, meta_path)


def load_or_create_experiment_meta(
    meta_path: str,
    config: dict[str, Any],
    initial_progress: dict[str, int | str],
    create_validation_config: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    if exists(meta_path):
        meta = load_experiment_meta(meta_path)
        assert_resume_config(meta["config"], config)
        print(f"resuming from {meta_path}: progress={meta['progress']}")
        return meta

    meta = {
        "config": config,
        "progress": initial_progress,
        "validation": create_validation_config(),
        "metrics": [],
    }
    save_experiment_meta(meta, meta_path)
    return meta


def load_training_checkpoint(model_path: str, device: str) -> dict[str, Any]:
    return torch.load(model_path, map_location=device, weights_only=False)


def load_model_checkpoint_if_exists(
    model_dir: str,
    bitness: int,
    device: str,
) -> dict[str, Any] | None:
    checkpoint_path = checkpoint_path_for(model_dir, bitness)
    if not exists(checkpoint_path):
        return None
    return load_training_checkpoint(checkpoint_path, device)


def save_training_checkpoint(
    model: Any,
    config: dict[str, Any],
    progress: dict[str, int | str],
    model_stem: str,
) -> None:
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": config,
        "progress": progress,
    }

    was_training = model.training
    model.eval()
    try:
        torch.save(checkpoint, model_stem + ".pt")
    finally:
        model.train(was_training)


def save_bitness_checkpoint(
    model: Any,
    config: dict[str, Any],
    progress: dict[str, int | str],
    model_dir: str,
    bitness: int,
) -> None:
    save_training_checkpoint(
        model,
        config,
        progress,
        model_stem_for(model_dir, bitness),
    )


def save_completed_target_state(
    meta: dict[str, Any],
    meta_path: str,
    model: Any,
    config: dict[str, Any],
    model_dir: str,
    bitness: int,
    iteration: int,
    train_loss: float,
    validation: dict[str, float],
) -> dict[str, int | str]:
    global_step = int(meta["progress"]["global_step"]) + 1
    progress = next_progress(config, iteration, bitness, global_step)
    metric = {
        "global_step": progress["global_step"],
        "iteration": iteration,
        "bitness": bitness,
        "train_loss": float(train_loss),
        "rmse": float(validation["rmse"]),
        "mae": float(validation["mae"]),
    }
    save_bitness_checkpoint(
        model,
        config,
        progress,
        model_dir,
        bitness,
    )
    append_validation_metric(meta, metric, progress, meta_path)
    return progress


def save_experiment_complete_state(
    meta: dict[str, Any],
    meta_path: str,
    config: dict[str, Any],
) -> None:
    progress = meta["progress"]
    if progress.get("stage") != "done":
        progress = {
            "stage": "done",
            "iteration": int(config["train_iterations"]),
            "bitness": int(config["min_bitness"]),
            "global_step": int(progress["global_step"]),
        }
        meta["progress"] = progress
    save_experiment_meta(meta, meta_path)


def append_validation_metric(
    meta: dict[str, Any],
    metric: dict[str, Any],
    progress: dict[str, int | str],
    meta_path: str,
) -> None:
    meta.setdefault("metrics", []).append(metric)
    meta["progress"] = progress
    save_experiment_meta(meta, meta_path)


def initial_progress(config: dict[str, Any]) -> dict[str, int | str]:
    return {
        "stage": "train",
        "iteration": 0,
        "bitness": int(config["min_bitness"]),
        "global_step": 0,
    }


def next_progress(
    config: dict[str, Any],
    iteration: int,
    bitness: int,
    global_step: int,
) -> dict[str, int | str]:
    max_bitness = int(config["max_bitness"])
    min_bitness = int(config["min_bitness"])
    train_iterations = int(config["train_iterations"])

    if bitness < max_bitness:
        return {
            "stage": "train",
            "iteration": iteration,
            "bitness": bitness + 1,
            "global_step": global_step,
        }
    if iteration + 1 < train_iterations:
        return {
            "stage": "train",
            "iteration": iteration + 1,
            "bitness": min_bitness,
            "global_step": global_step,
        }
    return {
        "stage": "done",
        "iteration": train_iterations,
        "bitness": min_bitness,
        "global_step": global_step,
    }


def model_stem_for(model_dir: str, bitness: int) -> str:
    return join(model_dir, f"pooling_b{bitness:02d}")


def checkpoint_path_for(model_dir: str, bitness: int) -> str:
    return model_stem_for(model_dir, bitness) + ".pt"


def assert_resume_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    keys = (
        "train_samples",
        "validation_samples",
        "min_bitness",
        "max_bitness",
        "train_iterations",
        "train_epochs",
        "batch_size",
        "lr",
        "reps",
        "threshold",
        "model",
        "seed_offset",
        "validation_seed_offset",
        "predict_batch_size",
        "target_case_batch_size",
    )
    for key in keys:
        assert saved[key] == current[key], (
            f"Cannot resume training with changed {key}: "
            f"saved={saved[key]!r}, current={current[key]!r}"
        )


def _atomic_json_dump(value: dict[str, Any], path: str) -> None:
    makedirs(dirname(path), exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=dirname(path),
        prefix=".tmp-",
        suffix=".json",
        delete=False,
    ) as f:
        dump(value, f, indent=2)
        f.write("\n")
        tmp_path = f.name
    replace(tmp_path, path)
