from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    BitnessExperimentConfig,
    bitness_weights_path,
    is_model_trained,
    load_bitness_config,
    load_or_create_bitness_snapshot,
    record_model_trained,
)
from src.dataloader import BitnessTrainingSampler
from src.generator_proxy import GeneratorProxy
from src.model import DeepSetPredictor


def run_training(
        generator: GeneratorProxy,
) -> None:
    config = load_bitness_config()
    snapshot = load_or_create_bitness_snapshot(config)
    if snapshot["progress"]["stage"] == "done":
        print("bitness training already complete")
        return

    training_config = config.training
    sampler = BitnessTrainingSampler(config, generator)
    models = build_models(config)
    optimizers = {
        bitness: torch.optim.Adam(model.parameters(), lr=training_config.lr)
        for bitness, model in models.items()
    }

    for iteration in range(
        training_config.iterations_from,
        training_config.iterations_to + 1,
    ):
        previous_model = None
        for bitness in range(config.bitness_from, config.bitness_to + 1):
            if is_model_trained(snapshot, bitness, iteration):
                load_saved_model(models[bitness], config, bitness, iteration)
                previous_model = models[bitness]
                continue

            model = models[bitness]
            optimizer = optimizers[bitness]

            last_rmse = float("inf")
            trained_epochs = 0
            for epoch in range(1, training_config.epochs + 1):
                loader = sampler.train_loader(
                    bitness,
                    iteration,
                    previous_model,
                    epoch,
                )
                last_rmse = train_epoch(model, optimizer, loader)
                trained_epochs = epoch
                print(
                    f"bitness iteration={iteration:03d} "
                    f"bitness={bitness:02d} "
                    f"epoch={epoch:03d} "
                    f"rmse={last_rmse:.6f}"
                )
                if last_rmse < training_config.rmse_threshold:
                    break

            weights_path = save_model_weights(
                model,
                config,
                bitness,
                iteration,
            )
            record_model_trained(
                snapshot,
                config,
                bitness,
                iteration,
                trained_epochs,
                last_rmse,
                weights_path,
            )
            print(
                f"saved bitness weights: iteration={iteration:03d} "
                f"bitness={bitness:02d} rmse={last_rmse:.6f} "
                f"path={weights_path}"
            )
            previous_model = model


def build_models(config: BitnessExperimentConfig) -> dict[int, nn.Module]:
    model_config = config.model
    assert model_config.name == "deepset", model_config.name
    models = {}
    bitnesses = range(config.bitness_from, config.bitness_to + 1)
    for bitness in tqdm(bitnesses, desc="models"):
        models[bitness] = DeepSetPredictor(
            point_dim=config.training.input_dim,
            n_points=model_config.n_points,
            phi_hidden=model_config.phi_hidden,
            phi_out=model_config.phi_out,
            rho_hidden=model_config.rho_hidden,
            dropout=model_config.dropout,
        )
    return models


def train_epoch(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
) -> float:
    criterion = nn.MSELoss()
    model.train()
    squared_error_sum = 0.0
    count = 0

    for xb, yb in loader:
        optimizer.zero_grad()
        prediction = model(xb)
        loss = criterion(prediction, yb)
        loss.backward()
        optimizer.step()

        errors = prediction.detach() - yb
        squared_error_sum += float(torch.sum(errors.square()).item())
        count += len(xb)

    return float((squared_error_sum / count) ** 0.5)


def save_model_weights(
        model: nn.Module,
        config: BitnessExperimentConfig,
        bitness: int,
        iteration: int,
) -> Path:
    weights_path = bitness_weights_path(config, bitness, iteration)
    torch.save(model.state_dict(), weights_path)
    return weights_path


def load_saved_model(
        model: nn.Module,
        config: BitnessExperimentConfig,
        bitness: int,
        iteration: int,
) -> None:
    weights_path = bitness_weights_path(config, bitness, iteration)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
