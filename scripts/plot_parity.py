#!/usr/bin/env python3
"""One-time check: feed parity-function samples to each saved bitness model
and plot the two predicted targets (depth score, size score) against bitness."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

DEEPCIRCUS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEPCIRCUS_DIR))

MODEL_DIR = Path("/tmp/circus")
SAMPLES = 256
SEED = 239


def parity_inputs(bitness: int, points_per_sample: int, rng: np.random.Generator) -> np.ndarray:
    # Mirrors InputGenerator (cpp/generator/utils.cpp): per sample, the first
    # half of the points are block modifications of one base input (rep's bits
    # choose which blocks to flip), the second half are fully random.
    block_reps = points_per_sample // 2
    blocks = (block_reps - 1).bit_length()
    flip_masks = np.zeros((block_reps, bitness), dtype=np.int64)
    block_start = 0
    for block_id in range(blocks):
        block_size = bitness // blocks + (1 if block_id < bitness % blocks else 0)
        rep_bit = (np.arange(block_reps) >> block_id) & 1
        flip_masks[:, block_start : block_start + block_size] = rep_bit[:, None]
        block_start += block_size
    assert block_start == bitness, (block_start, bitness)

    base = rng.integers(0, 2, size=(SAMPLES, 1, bitness))
    block_half = base ^ flip_masks[None, :, :]
    random_half = rng.integers(0, 2, size=(SAMPLES, points_per_sample - block_reps, bitness))
    return np.concatenate([block_half, random_half], axis=1)


def parity_samples(bitness: int, points_per_sample: int, rng: np.random.Generator) -> np.ndarray:
    # Point layout (see FillSample in cpp/generator/utils.cpp):
    # input bits, f(input), then f(input with bit j flipped) for each j,
    # every coordinate encoded as -1/+1 (PutBit for float writes -1.0f/1.0f).
    # Parity flips under any single-bit flip, so the flipped block is 1 - parity.
    bits = parity_inputs(bitness, points_per_sample, rng)
    parity = bits.sum(axis=-1, keepdims=True) % 2
    flipped = np.broadcast_to(1 - parity, bits.shape)
    sample = np.concatenate([bits, parity, flipped], axis=-1)
    return (2 * sample - 1).astype(np.float32)


def main() -> None:
    from src.config import load_train_config
    from src.model import DeepSetPredictor, predict_values

    config = load_train_config("train.conf")
    model_config = config.model
    assert model_config is not None, config

    bitnesses = sorted(
        int(match.group(1))
        for path in MODEL_DIR.glob("bitness_b*.pt")
        if (match := re.fullmatch(r"bitness_b(\d+)\.pt", path.name))
    )
    assert bitnesses, MODEL_DIR

    rng = np.random.default_rng(SEED)
    predictions = {}
    for bitness in bitnesses:
        model = DeepSetPredictor(
            point_dim=2 * bitness + 1,
            phi_hidden=model_config.phi_hidden,
            phi_out=model_config.phi_out,
            rho_hidden=model_config.rho_hidden,
            dropout=model_config.dropout,
        )
        import torch
        model.load_state_dict(torch.load(
            config.weights_path(bitness),
            map_location="cpu",
            weights_only=True,
        ))
        x = parity_samples(bitness, config.training.points_per_sample, rng)
        predictions[bitness] = predict_values(model, x, config.training.batch_size)
        print(f"bitness={bitness:02d} mean prediction: {predictions[bitness].mean(axis=0)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150, sharex=True)
    titles = ("depth score (bitness - depth)", "size score (log2(2^bitness - size))")
    for target, (axis, title) in enumerate(zip(axes, titles)):
        for bitness in bitnesses:
            values = predictions[bitness][:, target]
            axis.scatter([bitness] * len(values), values, s=8, alpha=0.15, color="#0072B2")
        means = [float(predictions[bitness][:, target].mean()) for bitness in bitnesses]
        axis.plot(bitnesses, means, color="#D55E00", linewidth=2, marker="o", label="mean")
        axis.set_title(title)
        axis.set_xlabel("bitness")
        axis.set_ylabel("prediction")
        axis.set_xticks(bitnesses)
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("model predictions on parity function")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = MODEL_DIR / "parity.png"
    fig.savefig(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
