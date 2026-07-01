import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.model import (
    DEVICE,
    DeepSetPredictor,
    evaluate_regression,
    predict_values,
)
from src.sampler import (
    generate_ids,
    generate_restriction_tensors,
    generate_sample_tensors,
    generate_samples,
)
from src.state_io import (
    DEFAULT_META_PATH,
    DEFAULT_MODEL_DIR,
    ensure_model_dir,
    initial_progress,
    load_model_checkpoint_if_exists,
    load_or_create_experiment_meta,
    save_completed_target_state,
    save_experiment_complete_state,
    save_experiment_meta,
)

TRAIN_SAMPLES = 1 << 12
VALIDATION_SAMPLES = 128

MIN_BITNESS = 4
MAX_BITNESS = 10

TRAIN_ITERATIONS = 100
TRAIN_EPOCHS = 500
BATCH_SIZE = 256
LR = 1e-3

REPS = 128
MODELS = {}
THRESHOLD = 0.025

MODEL = {
    "name": "deepset",
    "phi_hidden": 512,
    "phi_out": 256,
    "rho_hidden": 512,
    "dropout": 0.3,
    "predict_batch_size": 1024,
}

TARGET_CASE_BATCH_SIZE = 128
SEED_OFFSET = 239
VALIDATION_SEED_OFFSET = 1_000_239

GROUPS = (
    (4, 10),
    (11, 18),
    (19, 25),
    (26, 32),
)


def approximate_targets(
    generator,
    config: dict,
    bitness: int,
    case_ids: list[int],
) -> np.ndarray:
    previous_model = MODELS.get(bitness - 1)
    assert previous_model, f"Missing model for bitness {bitness - 1}"
    predict_batch_size = int(config["model"]["predict_batch_size"])

    target_parts = []
    ranges = range(0, len(case_ids), TARGET_CASE_BATCH_SIZE)
    total_batches = (len(case_ids) + TARGET_CASE_BATCH_SIZE - 1) // TARGET_CASE_BATCH_SIZE

    print(f"Generating reduced samples for {len(case_ids)} targets at bitness {bitness}")
    for start in tqdm(ranges, total=total_batches, desc=f"targets b={bitness}"):
        batch_ids = case_ids[start : start + TARGET_CASE_BATCH_SIZE]
        x_restricted = generate_restriction_tensors(
            generator,
            bitness,
            batch_ids,
            REPS,
        )
        predictions = predict_values(previous_model, x_restricted, predict_batch_size)
        predictions = predictions.reshape(len(batch_ids), bitness, 2)
        predictions = np.clip(
            predictions,
            0.0,
            np.log1p((1 << bitness) - 1),
        )
        branch_sizes = np.maximum(np.expm1(predictions), 0.0)
        split_sizes = 1.0 + branch_sizes.sum(axis=2)
        target_parts.append(np.log1p(split_sizes.min(axis=1)))

    return np.concatenate(target_parts).astype(np.float32)


def build_dataset(generator, config: dict, bitness: int, seed: int):
    ids = generate_ids(generator, bitness, TRAIN_SAMPLES, seed)
    if bitness <= 4:
        x_train, y_train = generate_samples(
            generator, bitness, ids, REPS
        )
        y_train = np.log1p(y_train)
        return x_train, y_train

    x_train = generate_sample_tensors(generator, bitness, ids, REPS)
    y_train = approximate_targets(generator, config, bitness, ids)
    return x_train, y_train


def evaluate_validation(
    model: nn.Module,
    config: dict,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> dict[str, float]:
    metrics = evaluate_regression(
        model,
        x_validation,
        y_validation,
        int(config["model"]["predict_batch_size"]),
    )
    return {
        "rmse": metrics["rmse"],
        "mae": metrics["mad"],
    }


def train_regression_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    threshold: float | None = None,
) -> float:
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor=0.5,
    )

    x = torch.tensor(x_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    final_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(x)
        scheduler.step(epoch_loss)
        epoch_elapsed = time.perf_counter() - epoch_start
        final_loss = float(epoch_loss)

        should_log = threshold is not None and epoch_loss < threshold
        if should_log or epoch % 10 == 0:
            print(
                f"Epoch {epoch:>3}/{epochs}  "
                f"loss={epoch_loss:.4f}  "
                f"elapsed={epoch_elapsed:.2f}s  "
                f"device={DEVICE}"
            )

        if threshold is not None and epoch_loss < threshold:
            break

    return final_loss


def train(
    bitness: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = TRAIN_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
) -> float:
    model = MODELS.get(bitness)
    return train_regression_model(
        model,
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        threshold=THRESHOLD,
    )


def run_experiment(generator):
    ensure_model_dir(DEFAULT_MODEL_DIR)
    config = build_config()
    meta = load_or_create_experiment_meta(
        DEFAULT_META_PATH,
        config,
        initial_progress(config),
        create_validation_config,
    )
    ensure_plot_series(meta)
    progress = meta["progress"]
    if progress["stage"] == "done":
        print(f"training already complete: {DEFAULT_META_PATH}")
        return

    validation = {}
    for bitness in range(MIN_BITNESS, MAX_BITNESS + 1):
        point_dim = 2 * bitness + 1
        MODELS[bitness] = build_model(point_dim, config)
        checkpoint = load_model_checkpoint_if_exists(DEFAULT_MODEL_DIR, bitness, DEVICE)
        if checkpoint is not None:
            MODELS[bitness].load_state_dict(checkpoint["state_dict"])

    for iteration in range(int(progress["iteration"]), TRAIN_ITERATIONS):
        for bitness in range(MIN_BITNESS, MAX_BITNESS + 1):
            progress = meta["progress"]
            if iteration == progress["iteration"] and bitness < progress["bitness"]:
                continue

            seed = iteration + SEED_OFFSET
            x_train, y_train = build_dataset(generator, config, bitness, seed)
            loss = train(
                bitness,
                x_train,
                y_train,
            )
            validation[bitness] = validation.get(bitness) or load_validation_dataset(
                generator,
                meta,
                bitness,
            )
            validation_result = evaluate_validation(
                MODELS[bitness],
                config,
                validation[bitness]["x"],
                validation[bitness]["y"],
            )
            save_completed_target_state(
                meta,
                DEFAULT_META_PATH,
                MODELS[bitness],
                config,
                DEFAULT_MODEL_DIR,
                bitness,
                iteration,
                loss,
                validation_result,
            )

    save_experiment_complete_state(meta, DEFAULT_META_PATH, config)


def build_config() -> dict:
    return {
        "DEFAULT_MODEL_DIR": DEFAULT_MODEL_DIR,
        "train_samples": TRAIN_SAMPLES,
        "validation_samples": VALIDATION_SAMPLES,
        "min_bitness": MIN_BITNESS,
        "max_bitness": MAX_BITNESS,
        "train_iterations": TRAIN_ITERATIONS,
        "train_epochs": TRAIN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "reps": REPS,
        "threshold": THRESHOLD,
        "model": dict(MODEL),
        "seed_offset": SEED_OFFSET,
        "validation_seed_offset": VALIDATION_SEED_OFFSET,
        "predict_batch_size": MODEL["predict_batch_size"],
        "target_case_batch_size": TARGET_CASE_BATCH_SIZE,
    }


def build_model(point_dim: int, config: dict) -> DeepSetPredictor:
    model_config = dict(config["model"])
    model_name = model_config.pop("name")
    model_config.pop("predict_batch_size", None)
    assert model_name == "deepset", model_name
    return DeepSetPredictor(point_dim=point_dim, **model_config)


def create_validation_config() -> dict[str, dict[str, list[int]]]:
    return {}


def ensure_plot_series(meta: dict) -> None:
    plot_series = build_plot_series()
    if meta.get("series") == plot_series:
        return

    meta["series"] = plot_series
    meta.pop("plot_series", None)
    save_experiment_meta(meta, DEFAULT_META_PATH)


def build_plot_series() -> list[dict]:
    series = []
    for start, end in GROUPS:
        start = max(start, MIN_BITNESS)
        end = min(end, MAX_BITNESS)
        if start > end:
            continue
        series.append(
            {
                "name": f"pooling_b{start:02d}_b{end:02d}",
                "title": f"Pooling validation metrics, bitness {start}-{end}",
                "x_label": "Iteration",
                "legend_title": "Bitness",
                "lines": [
                    {
                        "label": str(bitness),
                        "where": {"bitness": bitness},
                        "x_key": "iteration",
                        "mae_key": "mae",
                        "rmse_key": "rmse",
                    }
                    for bitness in range(start, end + 1)
                ],
            }
        )
    return series


def load_validation_dataset(generator, meta: dict, bitness: int) -> dict[str, np.ndarray]:
    ensure_validation_entry(generator, meta, bitness)
    validation = meta["validation"][str(bitness)]
    ids = [int(case_id) for case_id in validation["ids"]]
    x_validation = generate_sample_tensors(generator, bitness, ids, REPS)
    y_validation = np.asarray(
        [np.log1p(generator.case_nodes(bitness, case_id)) for case_id in ids],
        dtype=np.float32,
    )
    return {
        "x": x_validation,
        "y": y_validation,
    }


def ensure_validation_entry(generator, meta: dict, bitness: int) -> None:
    key = str(bitness)
    if key in meta["validation"]:
        return

    ids = generate_ids(
        generator,
        bitness,
        VALIDATION_SAMPLES,
        VALIDATION_SEED_OFFSET + bitness,
    )
    meta["validation"][key] = {
        "ids": ids,
    }
    save_experiment_meta(meta, DEFAULT_META_PATH)
