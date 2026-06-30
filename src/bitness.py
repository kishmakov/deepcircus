from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config, load_bitness_config
from src.dataloader import Sampler
from src.generator_proxy import GeneratorProxy
from src.model import DeepSetPredictor


@dataclass
class IterationProgress:
    iteration: int
    bitness: int
    epoch: int = 0
    last_rmse: float = float("inf")

    def record_epoch(self, epoch: int, rmse: float) -> None:
        self.epoch = epoch
        self.last_rmse = rmse

    def print_epoch(self) -> None:
        assert self.epoch > 0, self
        print(
            f"bitness iteration={self.iteration:03d} "
            f"bitness={self.bitness:02d} "
            f"epoch={self.epoch:03d} "
            f"rmse={self.last_rmse:.6f}"
        )

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
            self.last_rmse,
        )
        print(
            f"saved: iteration={self.iteration:03d} "
            f"bitness={self.bitness:02d} rmse={self.last_rmse:.6f}"
        )


def run_training(generator: GeneratorProxy) -> None:
    config = load_bitness_config()
    sampler = Sampler(config, generator)
    models: dict[int, nn.Module] = {}
    optimizers: dict[int, torch.optim.Optimizer] = {}

    for iteration in config.iterations_range():
        previous_model = None
        for bitness in config.bitness_range():
            progress = IterationProgress(iteration, bitness)
            model, optimizer = get_or_create_model(
                models,
                optimizers,
                config,
                bitness,
            )

            for epoch in config.epochs_range():
                loader = sampler.train_loader(
                    bitness,
                    iteration,
                    previous_model,
                    epoch,
                )
                progress.record_epoch(epoch, train_epoch(model, optimizer, loader))

                if progress.last_rmse < config.rmse_threshold():
                    progress.print_epoch()
                    break

                if epoch % 10 == 0:
                    progress.print_epoch()

            progress.checkpoint_model(model, config)
            previous_model = model

        config.record_iteration_trained(iteration)


def get_or_create_model(
        models: dict[int, nn.Module],
        optimizers: dict[int, torch.optim.Optimizer],
        config: Config,
        bitness: int,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    if bitness not in models:
        model_config = config.model
        assert model_config.name == "deepset", model_config.name
        model = DeepSetPredictor(
            point_dim=config.training.input_dim,
            n_points=model_config.n_points,
            phi_hidden=model_config.phi_hidden,
            phi_out=model_config.phi_out,
            rho_hidden=model_config.rho_hidden,
            dropout=model_config.dropout,
        )
        load_saved_model(model, config, bitness)
        models[bitness] = model
        optimizers[bitness] = torch.optim.Adam(
            model.parameters(),
            lr=config.training.lr,
        )
    return models[bitness], optimizers[bitness]


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


def load_saved_model(
        model: nn.Module,
        config: Config,
        bitness: int,
) -> None:
    weights_path = config.weights_path(bitness)
    if not weights_path.exists():
        return
    model.load_state_dict(torch.load(weights_path, weights_only=True))
