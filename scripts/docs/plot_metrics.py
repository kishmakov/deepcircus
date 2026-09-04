"""Draws one run's train and validation curves; the entry point is
`plot_metrics.sh` beside it, which supplies the virtualenv's interpreter.

Reads `work_dir` out of the training config and plots every
`<model>_<bitness>.metrics.json` it finds there, writing each PNG next to the
JSON it came from.
"""

from __future__ import annotations

from argparse import ArgumentParser
from json import load
from os import environ, makedirs
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
makedirs(environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

# Nothing here shows a window; every figure leaves as a file.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "conf" / "train.yaml"

# Colourblind-safe, and the two curves stay apart in greyscale as well.
TRAIN_COLOR = "#0072B2"
VALIDATION_COLOR = "#E69F00"


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("lin", "log"), default="lin", help="the Y axis")
    parser.add_argument("--config", type=Path, default=CONFIG, help="the training config to read work_dir from")
    arguments = parser.parse_args()

    directory = work_dir(arguments.config)
    assert directory.is_dir(), f"{directory} does not exist; nothing has been trained yet"
    metrics_paths = sorted(directory.glob("*.metrics.json"))
    assert metrics_paths, f"no *.metrics.json in {directory}"

    for metrics_path in metrics_paths:
        print(plot(metrics_path, arguments.scale))


def work_dir(config_path: Path) -> Path:
    config = OmegaConf.load(config_path)
    # Relative to the repository, the same rule `src.config` reads it by.
    path = Path(str(config["work_dir"]))
    return path if path.is_absolute() else ROOT / path


def plot(metrics_path: Path, scale: str) -> Path:
    with open(metrics_path, encoding="utf-8") as f:
        metrics = load(f)

    epochs: list[dict[str, Any]] = metrics["epochs"]
    assert epochs, f"no epochs in {metrics_path}"
    numbers = [int(entry["epoch"]) for entry in epochs]
    train = [float(entry["train_rmse"]) for entry in epochs]
    validation = [float(entry["validation_rmse"]) for entry in epochs]
    if scale == "log":
        assert all(value > 0 for value in train + validation), metrics_path

    figure, axis = plt.subplots(figsize=(9, 5.5), dpi=150)
    axis.plot(numbers, train, label="train", color=TRAIN_COLOR, linewidth=2)
    axis.plot(numbers, validation, label="validation", color=VALIDATION_COLOR, linewidth=2)

    axis.set_title(
        f"{metrics['model']} at {int(metrics['bitness']):02d}b"
        f" -- {len(epochs)} epochs, best validation {metrics['best_validation_rmse']:.4f}"
    )
    axis.set_xlabel("epoch")
    axis.set_ylabel("RMSE")
    if scale == "log":
        axis.set_yscale("log")
        # An RMSE curve rarely crosses a decade, and the default log formatter
        # would label two ticks and leave the rest of the axis blank.
        for formatter in (axis.yaxis.set_major_formatter, axis.yaxis.set_minor_formatter):
            formatter(ScalarFormatter())
    axis.grid(alpha=0.2, which="both")
    axis.legend()

    figure.tight_layout()
    output_path = metrics_path.with_suffix(".png")
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    main()
