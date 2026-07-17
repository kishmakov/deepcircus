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

from src.config import TrainingConfig
from src.generator import expand_inputs

MODEL_DIR = Path("/tmp/circus")
SAMPLES = 256
RANDOM_SAMPLES = 256
SEED = 239


def sample_inputs(
    bitness: int, training: TrainingConfig, samples: int, rng: np.random.Generator
) -> np.ndarray:
    # The per-batch base sequences are the only randomness; the block-and-random
    # expansion into points is the shared C++ walk (gen::ExpandInputs, the same
    # code Case::Sample uses during training-data generation).
    sequences = rng.integers(0, 2, size=(samples, training.sample_batches, bitness))
    return expand_inputs(sequences, training.sample_points_in_batch)


def parity_samples(bitness: int, training: TrainingConfig, rng: np.random.Generator) -> np.ndarray:
    # Point layout (see Case::SampleValues/ComputeAt in cpp/generator/case.cpp):
    # input bits, f(input), then f(input with bit j flipped) for each j,
    # every coordinate encoded as -1/+1.
    # Parity flips under any single-bit flip, so the flipped block is 1 - parity.
    bits = sample_inputs(bitness, training, SAMPLES, rng)
    parity = bits.sum(axis=-1, keepdims=True) % 2
    flipped = np.broadcast_to(1 - parity, bits.shape)
    sample = np.concatenate([bits, parity, flipped], axis=-1)
    return (2 * sample - 1).astype(np.float32)


def random_samples(bitness: int, training: TrainingConfig, rng: np.random.Generator) -> np.ndarray:
    # Baseline: inputs follow the same block-alteration scheme as parity, but
    # f(input) and the flipped-bit block are random bits with no consistency,
    # showing the model's average output level on structureless values.
    bits = sample_inputs(bitness, training, RANDOM_SAMPLES, rng)
    points = training.sample_batches * training.sample_points_in_batch
    values = rng.integers(0, 2, size=(RANDOM_SAMPLES, points, bitness + 1))
    sample = np.concatenate([bits, values], axis=-1)
    return (2 * sample - 1).astype(np.float32)


def main() -> None:
    from src.config import load_train_config
    from src.generator import sample_point_dim
    from src.model import DeepSetPredictor, predict_values

    config = load_train_config("train.conf")
    model_config = config.model
    assert model_config is not None, config
    training = config.training

    bitnesses = sorted(
        int(match.group(1))
        for path in MODEL_DIR.glob("bitness_b*.pt")
        if (match := re.fullmatch(r"bitness_b(\d+)\.pt", path.name))
    )
    assert bitnesses, MODEL_DIR

    rng = np.random.default_rng(SEED)
    parity_predictions = {}
    random_predictions = {}
    for bitness in bitnesses:
        model = DeepSetPredictor(
            batches=training.sample_batches,
            points_in_batch=training.sample_points_in_batch,
            point_dim=sample_point_dim(bitness),
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
        x = parity_samples(bitness, training, rng)
        parity_predictions[bitness] = predict_values(model, x, training.batch_size)
        r = random_samples(bitness, training, rng)
        random_predictions[bitness] = predict_values(model, r, training.batch_size)
        print(
            f"bitness={bitness:02d} mean parity: {parity_predictions[bitness].mean(axis=0)}"
            f" random: {random_predictions[bitness].mean(axis=0)}"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150, sharex=True)
    titles = ("depth score (bitness - depth)", "size score (log2(2^bitness - size))")
    for target, (axis, title) in enumerate(zip(axes, titles)):
        for bitness in bitnesses:
            values = parity_predictions[bitness][:, target]
            axis.scatter([bitness] * len(values), values, s=8, alpha=0.15, color="#0072B2")
        parity_means = [float(parity_predictions[bitness][:, target].mean()) for bitness in bitnesses]
        axis.plot(bitnesses, parity_means, color="#D55E00", linewidth=2, marker="o", label="parity mean")
        boxes = axis.boxplot(
            [random_predictions[bitness][:, target] for bitness in bitnesses],
            positions=bitnesses,
            widths=0.5,
            manage_ticks=False,
            boxprops={"color": "#009E73"},
            whiskerprops={"color": "#009E73"},
            capprops={"color": "#009E73"},
            medianprops={"color": "#009E73", "linewidth": 2},
            flierprops={"marker": ".", "markeredgecolor": "#009E73"},
        )
        boxes["boxes"][0].set_label("random distribution")
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
