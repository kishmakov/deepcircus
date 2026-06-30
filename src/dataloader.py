from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.generator_proxy import GeneratorProxy


class Sampler:
    def __init__(
            self,
            config: Config,
            generator: GeneratorProxy,
    ):
        self.config = config
        self.training = config.training
        self.generator = generator
        self._inputs: dict[tuple[int, int], torch.Tensor] = {}

    def train_loader(
            self,
            bitness: int,
            iteration: int,
            previous_model: nn.Module | None,
            epoch: int,
    ) -> DataLoader:
        inputs = self.inputs(bitness, iteration)
        targets = self.targets(inputs, bitness, iteration, previous_model)
        generator = torch.Generator()
        generator.manual_seed(
            self.training.seed + iteration * 10_000 + bitness * 100 + epoch,
        )
        return DataLoader(
            TensorDataset(inputs, targets),
            batch_size=self.training.batch_size,
            shuffle=True,
            generator=generator,
        )

    def inputs(
            self,
            bitness: int,
            iteration: int,
    ) -> torch.Tensor:
        key = (bitness, iteration)
        if key not in self._inputs:
            self._inputs[key] = self._sample_inputs(bitness, iteration)
        return self._inputs[key]

    def targets(
            self,
            inputs: torch.Tensor,
            bitness: int,
            iteration: int,
            previous_model: nn.Module | None,
    ) -> torch.Tensor:
        flat_inputs = inputs.reshape(inputs.shape[0], -1)
        mean_signal = flat_inputs.mean(dim=1)
        variance_signal = flat_inputs.square().mean(dim=1)
        target = bitness * mean_signal + iteration * variance_signal

        if previous_model is not None:
            was_training = previous_model.training
            previous_model.eval()
            with torch.no_grad():
                previous_prediction = previous_model(inputs).reshape(-1)
            previous_model.train(was_training)
            target = target + 0.1 * previous_prediction

        return target.reshape(-1, 1)

    def _sample_inputs(
            self,
            bitness: int,
            iteration: int,
    ) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(self.training.seed + iteration * 10_000 + bitness)
        return torch.randn(
            self.training.samples_per_model,
            self.config.model.n_points,
            self.training.input_dim,
            generator=generator,
            dtype=torch.float32,
        )
