#!/usr/bin/env python3
"""Plot per-bitness train/validation RMSE curves from the bitness snapshot."""

from __future__ import annotations

from argparse import ArgumentParser
from json import load
from os import environ, makedirs
from os.path import dirname, join
from pathlib import Path
import sys
from typing import Any
from math import log

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]

environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
makedirs(environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt


DEFAULT_META = "/tmp/circus/bitness_snapshot.json"
GROUP_SIZE = 5
COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("scale", nargs="?", choices=("lin", "log"), default="lin")
    parser.add_argument("--json", default=DEFAULT_META)
    args = parser.parse_args()

    with open(args.json, encoding="utf-8") as f:
        meta = load(f)

    metrics = list(meta["metrics"])
    metrics += meta.get("pending_iteration", {}).get("metrics", [])
    assert metrics, f"no metrics in {args.json}"

    output_dir = dirname(args.json) or "."
    makedirs(output_dir, exist_ok=True)

    for group_start, bitnesses in bitness_groups(metrics).items():
        group_end = group_start + GROUP_SIZE - 1
        out_path = join(output_dir, f"rmse_b{group_start:02d}_b{group_end:02d}.png")
        plot_group(metrics, bitnesses, out_path, args.scale)
        print(out_path)


def bitness_groups(metrics: list[dict[str, Any]]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for bitness in sorted({int(metric["bitness"]) for metric in metrics}):
        groups.setdefault(bitness // GROUP_SIZE * GROUP_SIZE, []).append(bitness)
    return groups


def plot_group(
    metrics: list[dict[str, Any]],
    bitnesses: list[int],
    out_path: str,
    scale: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150, sharex=True)

    for bitness in bitnesses:
        points = bitness_points(metrics, bitness)
        x_values = [point["iteration"] for point in points]
        color = COLORS[bitness % GROUP_SIZE]
        for axis, rmse_key in zip(axes, ("train_rmse", "val_rmse")):
            axis.plot(
                x_values,
                transform_y_values([point[rmse_key] for point in points], scale),
                label=f"bitness {bitness}",
                color=color,
                linewidth=2,
                marker="o",
                markersize=4,
            )

    for axis, title in zip(axes, ("train", "validation")):
        axis.set_title(title)
        axis.set_xlabel("iteration")
        axis.set_ylabel("RMSE" if scale == "lin" else "log(RMSE)")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle(f"bitness {bitnesses[0]}–{bitnesses[-1]}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


def bitness_points(
    metrics: list[dict[str, Any]],
    bitness: int,
) -> list[dict[str, float]]:
    points = [
        {
            "iteration": float(metric["iteration"]),
            "train_rmse": float(metric["train_rmse"]),
            "val_rmse": float(metric["val_rmse"]),
        }
        for metric in metrics
        if int(metric["bitness"]) == bitness
    ]
    points.sort(key=lambda point: point["iteration"])
    return points


def transform_y_values(values: list[float], scale: str) -> list[float]:
    if scale == "lin":
        return values
    assert scale == "log", scale
    assert all(value > 0 for value in values), values
    return [log(value) for value in values]


if __name__ == "__main__":
    main()
