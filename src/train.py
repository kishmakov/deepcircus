from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.generator import sample_point_dim
from src.config import TrainConfig, load_train_config
from src.model import DEVICE, DeepSetPredictor
from src.sampler import GeneratorProxy, Sampler


@dataclass
class IterationProgress:
    iteration: int
    bitness: int
    epoch: int = 0
    last_train_rmse: float = float("inf")
    last_val_rmse: float = float("inf")

    def record_epoch(self, epoch: int, train_rmse: float, val_rmse: float) -> None:
        self.epoch = epoch
        self.last_train_rmse = train_rmse
        self.last_val_rmse = val_rmse

    def checkpoint_model(
            self,
            model: nn.Module,
            config: TrainConfig,
    ) -> Path:
        assert self.epoch > 0, self
        torch.save(model.state_dict(), config.weights_path(self.bitness))
        config.record_model_trained(
            self.bitness,
            self.iteration,
            self.epoch,
            self.last_train_rmse,
            self.last_val_rmse,
        )
        print(
            f"saved: iteration={self.iteration:03d} "
            f"bitness={self.bitness:02d} rmse={self.last_train_rmse:.6f}"
        )


def run_training(generator: GeneratorProxy) -> None:
    config = load_train_config("train.conf")
    sampler = Sampler(config, generator)
    models: dict[int, nn.Module] = {}
    previous_model = None
    previous_iteration = None

    for iteration, bitness in sampler.coordinates():
        if iteration != previous_iteration:
            previous_model = None
        train_dataset = sampler.train_dataset(iteration, bitness, previous_model)
        progress = IterationProgress(iteration, bitness)
        model = get_or_create_model(models, config, bitness)
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)
        epoch_progress = tqdm(
            config.epochs_range(),
            desc=f"iteration={iteration:03d} bitness={bitness:02d}",
            unit="epoch",
        )
        validation_loader = sampler.validation_loader(bitness)

        for epoch in epoch_progress:
            train_loader = sampler.train_loader(iteration, bitness, train_dataset, epoch)
            train_rmse = train_epoch(model, optimizer, scheduler, train_loader)
            val_rmse = evaluate_epoch(model, validation_loader)
            progress.record_epoch(epoch, train_rmse, val_rmse)
            epoch_progress.set_postfix(rmse=f"{progress.last_train_rmse:.6f}")

            if progress.last_train_rmse < config.rmse_threshold():
                break

        progress.checkpoint_model(model, config)
        previous_model = model
        previous_iteration = iteration

        if bitness == config.bitness_to:
            config.record_iteration_trained(iteration)


def get_or_create_model(
        models: dict[int, nn.Module],
        config: TrainConfig,
        bitness: int,
) -> nn.Module:
    if bitness not in models:
        model_config = config.model
        assert model_config is not None, config
        assert model_config.name == "deepset", model_config.name
        model = DeepSetPredictor(
            point_dim=sample_point_dim(bitness),
            batches=config.training.sample_batches,
            points_in_batch=config.training.sample_points_in_batch,
            phi_hidden=model_config.phi_hidden,
            phi_out=model_config.phi_out,
            rho_hidden=model_config.rho_hidden,
            dropout=model_config.dropout,
        )
        load_saved_model(model, config, bitness)
        model.to(DEVICE)
        models[bitness] = model
    return models[bitness]


def create_optimizer(
        model: nn.Module,
        config: TrainConfig,
) -> torch.optim.Optimizer:
    optimizer_config = config.optimizer
    assert optimizer_config is not None, config
    assert optimizer_config.name == "adam", optimizer_config.name
    return torch.optim.Adam(
        model.parameters(),
        lr=optimizer_config.lr,
    )


def create_scheduler(
        optimizer: torch.optim.Optimizer,
        config: TrainConfig,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    optimizer_config = config.optimizer
    assert optimizer_config is not None, config
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=optimizer_config.scheduler_patience,
        factor=optimizer_config.scheduler_factor,
    )


def train_epoch(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        loader: DataLoader,
) -> float:
    criterion = nn.MSELoss()
    model.train()
    squared_error_sum = 0.0
    count = 0

    for xb, yb in loader:
        # xb stays packed uint8; the model unpacks it on-device
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        optimizer.zero_grad()
        prediction = model(xb)
        loss = criterion(prediction, yb)
        loss.backward()
        optimizer.step()

        errors = prediction.detach() - yb
        squared_error_sum += float(torch.sum(errors.square()).item())
        count += yb.numel()

    rmse = float((squared_error_sum / count) ** 0.5)
    scheduler.step(rmse)
    return rmse

def evaluate_epoch(
        model: nn.Module,
        loader: DataLoader,
) -> float:
    model.eval()
    squared_error_sum = 0.0
    count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            prediction = model(xb)
            errors = prediction - yb
            squared_error_sum += float(torch.sum(errors.square()).item())
            count += yb.numel()

    return float((squared_error_sum / count) ** 0.5)


def load_saved_model(
        model: nn.Module,
        config: TrainConfig,
        bitness: int,
) -> None:
    weights_path = config.weights_path(bitness)
    if not weights_path.exists():
        return
    model.load_state_dict(torch.load(
        weights_path,
        map_location=DEVICE,
        weights_only=True,
    ))
