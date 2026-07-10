from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config, load_bitness_config
from src.model import DEVICE, DeepSetPredictor
from src.sampler import GeneratorProxy, Sampler


BOOL_BENCH_DIR = Path(__file__).resolve().parents[1] / "bool-bench"
if str(BOOL_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BOOL_BENCH_DIR))

from bool_bench import (
    sample_point_dim,
)


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
            config: Config,
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
    config = load_bitness_config()
    sampler = Sampler(config, generator)
    models: dict[int, nn.Module] = {}
    coordinates = [
        (iteration, bitness)
        for iteration in config.iterations_range()
        for bitness in config.bitness_range()
    ]
    if not coordinates:
        return

    current_stage = sampler.request_stage(*coordinates[0])
    previous_model = None
    previous_iteration = None

    for coordinate_id, (iteration, bitness) in enumerate(coordinates):
        if iteration != previous_iteration:
            previous_model = None
        stage = current_stage
        assert (stage.iteration, stage.bitness) == (iteration, bitness)
        sampler.acquire_stage(stage)

        next_stage = None
        if coordinate_id + 1 < len(coordinates):
            next_stage = sampler.request_stage(*coordinates[coordinate_id + 1])

        sampler.prepare_stage(stage, previous_model)
        progress = IterationProgress(iteration, bitness)
        model = get_or_create_model(models, config, bitness)
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)
        epoch_progress = tqdm(
            config.epochs_range(),
            desc=f"iteration={iteration:03d} bitness={bitness:02d}",
            unit="epoch",
        )
        validation_loader = sampler.validation_loader(stage)

        for epoch in epoch_progress:
            train_loader = sampler.train_loader(stage, epoch)
            train_rmse = train_epoch(model, optimizer, scheduler, train_loader)
            val_rmse = evaluate_epoch(model, validation_loader)
            progress.record_epoch(epoch, train_rmse, val_rmse)
            epoch_progress.set_postfix(rmse=f"{progress.last_train_rmse:.6f}")

            if progress.last_train_rmse < config.rmse_threshold():
                break

        progress.checkpoint_model(model, config)
        del train_loader
        del validation_loader
        sampler.release_stage(stage)
        current_stage = next_stage
        previous_model = model
        previous_iteration = iteration

        if bitness == config.bitness_to:
            config.record_iteration_trained(iteration)


def get_or_create_model(
        models: dict[int, nn.Module],
        config: Config,
        bitness: int,
) -> nn.Module:
    if bitness not in models:
        model_config = config.model
        assert model_config.name == "deepset", model_config.name
        model = DeepSetPredictor(
            point_dim=sample_point_dim(bitness),
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
        config: Config,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=config.training.lr,
    )


def create_scheduler(
        optimizer: torch.optim.Optimizer,
        config: Config,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.training.scheduler_patience,
        factor=config.training.scheduler_factor,
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
        xb = torch.as_tensor(xb, dtype=torch.float32, device=DEVICE)
        yb = torch.as_tensor(yb, dtype=torch.float32, device=DEVICE)
        optimizer.zero_grad()
        prediction = model(xb)
        loss = criterion(prediction, yb)
        loss.backward()
        optimizer.step()

        errors = prediction.detach() - yb
        squared_error_sum += float(torch.sum(errors.square()).item())
        count += len(xb)

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
            xb = torch.as_tensor(xb, dtype=torch.float32, device=DEVICE)
            yb = torch.as_tensor(yb, dtype=torch.float32, device=DEVICE)
            prediction = model(xb)
            errors = prediction - yb
            squared_error_sum += float(torch.sum(errors.square()).item())
            count += len(xb)

    return float((squared_error_sum / count) ** 0.5)


def load_saved_model(
        model: nn.Module,
        config: Config,
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
