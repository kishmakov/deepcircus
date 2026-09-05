"""Reconstruct unknown offline targets through already trained models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import TrainConfig
from src.daemon.client import Client
from src.model import DEVICE, make_predictor, predict_values


_PARENTS_PER_CHUNK = 64


def reconstruct_unknown_targets(config: TrainConfig, client: Client) -> np.ndarray:
    """Predicts the two scores of every unknown training entry in file order."""
    unknown = client.sizes.unknown_train
    assert unknown > 0, unknown
    assert config.bitness > 8, config.bitness

    targets = _primary_targets(config, client, unknown)
    if config.model_name == "m1":
        targets = np.maximum(targets, _helper_targets(config, client, unknown))
    assert targets.shape == (unknown, 2), targets.shape
    assert np.isfinite(targets).all(), targets
    assert ((targets >= 0.0) & (targets <= config.bitness)).all(), targets
    return targets.astype(np.float32)


def combine_primary_predictions(
        predictions: np.ndarray,
        parents: int,
        candidates: int,
        child_bitness: int,
) -> np.ndarray:
    """Combines both fixed-value branches, then selects the best input split."""
    expected = (parents * candidates * 2, 2)
    assert predictions.shape == expected, (predictions.shape, expected)
    children = np.clip(predictions.astype(np.float64), 0.0, float(child_bitness))
    children = children.reshape(parents, candidates, 2, 2)

    # A parent query has depth 1 + max(d0, d1) and size 1 + s0 + s1.
    # In score space those become min(depth scores) and
    # log2(2^size_score_0 + 2^size_score_1 - 1).
    depth = children[..., 0].min(axis=2).max(axis=1)
    size_budget = np.exp2(children[..., 1]).sum(axis=2) - 1.0
    size = np.log2(size_budget).max(axis=1)
    return np.column_stack((depth, size)).astype(np.float32)


def combine_helper_predictions(predictions: np.ndarray, parents: int, bitness: int) -> np.ndarray:
    """Combines the two M_2 branches reached by querying M_1's helper."""
    expected = (parents * 2, 2)
    assert predictions.shape == expected, (predictions.shape, expected)
    children = np.clip(predictions.astype(np.float64), 0.0, float(bitness)).reshape(parents, 2, 2)

    # These children still have bitness n rather than n - 1, so accounting for
    # the new f-query subtracts one depth point and one full 2^n size budget.
    depth = np.maximum(0.0, children[..., 0].min(axis=1) - 1.0)
    size_budget = np.exp2(children[..., 1]).sum(axis=1) - np.exp2(float(bitness)) - 1.0
    size = np.log2(np.maximum(1.0, size_budget))
    return np.column_stack((depth, size)).astype(np.float32)


def _primary_targets(config: TrainConfig, client: Client, unknown: int) -> np.ndarray:
    child_bitness = config.bitness - 1
    model = _load_model(config, config.model_name, child_bitness)
    targets = np.empty((unknown, 2), dtype=np.float32)
    with tqdm(total=unknown, desc=f"bootstrap {config.model_name}_{config.bitness:02d} through inputs") as progress:
        for first in range(0, unknown, _PARENTS_PER_CHUNK):
            count = min(_PARENTS_PER_CHUNK, unknown - first)
            reductions = client.primary_reductions(first, count)
            assert reductions.columns == config.sampling.points * (3 * child_bitness + 2), reductions.columns
            predictions = predict_values(model, reductions.values, config.training.batch_size)
            targets[first : first + count] = combine_primary_predictions(
                predictions,
                count,
                config.bitness,
                child_bitness,
            )
            progress.update(count)
    del model
    _release_device_cache()
    return targets


def _helper_targets(config: TrainConfig, client: Client, unknown: int) -> np.ndarray:
    model = _load_model(config, "m2", config.bitness)
    targets = np.empty((unknown, 2), dtype=np.float32)
    with tqdm(total=unknown, desc=f"bootstrap {config.tag} through f") as progress:
        for first in range(0, unknown, _PARENTS_PER_CHUNK):
            count = min(_PARENTS_PER_CHUNK, unknown - first)
            reductions = client.helper_reductions(first, count)
            assert reductions.columns == config.sampling.points * config.point_dim, reductions.columns
            predictions = predict_values(model, reductions.values, config.training.batch_size)
            targets[first : first + count] = combine_helper_predictions(predictions, count, config.bitness)
            progress.update(count)
    del model
    _release_device_cache()
    return targets


def _load_model(config: TrainConfig, model_name: str, bitness: int) -> torch.nn.Module:
    path = config.work_dir / f"{model_name}_{bitness:02d}.best.pt"
    assert path.exists(), _missing_model_message(path, model_name, bitness)
    model = make_predictor(config, bitness)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


def _missing_model_message(path: Path, model_name: str, bitness: int) -> str:
    return f"{path} is required for bootstrapping; train {model_name}_{bitness:02d} first"


def _release_device_cache() -> None:
    torch.cuda.empty_cache()
