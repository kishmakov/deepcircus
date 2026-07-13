from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.generator import GeneratedTask, Generator, load_generator
from src.config import TrainConfig
from src.model import predict_values


class GeneratorProxy:
    """Thin compatibility wrapper around the C++ generator daemon."""

    def __init__(self, workers: int):
        self._generator = load_generator(workers=workers)

    def __getattr__(self, name):
        return getattr(self._generator, name)

    def close(self) -> None:
        self._generator.close()

    @property
    def generator(self) -> Generator:
        return self._generator


@dataclass
class Stage:
    iteration: int
    bitness: int
    train_dataset: torch.utils.data.Dataset


class Sampler:
    def __init__(self, config: TrainConfig, generator: GeneratorProxy):
        self.config = config
        self.training = config.training
        self.generator = generator
        self._coordinates = [
            (iteration, bitness)
            for iteration in config.iterations_range()
            for bitness in config.bitness_range()
        ]
        self._validation_datasets: dict[int, torch.utils.data.Dataset] = {}
        if self._coordinates:
            iterations = config.iterations_range()
            self.generator.initialize(
                first_iteration=iterations.start,
                last_iteration=self.training.iterations,
                bitness_from=self.config.bitness_from,
                bitness_to=self.config.bitness_to,
                seed=self.training.seed,
                train_samples=self.training.train_samples,
                validation_samples=self.training.validation_samples,
                points_per_sample=self.training.points_per_sample,
            )
            self._take_validation_datasets()

    def coordinates(self) -> list[tuple[int, int]]:
        return list(self._coordinates)

    def _take_validation_datasets(self) -> None:
        # The daemon emits one validation-only task (iteration 0) per bitness
        # ahead of every training task, since validation depends only on
        # bitness; pull them all up front and keep them for the whole run.
        for bitness in self.config.bitness_range():
            task = self.generator.next_task()
            assert task is not None, bitness
            assert (task.iteration, task.bitness) == (0, bitness), task
            self._validation_datasets[bitness] = torch.utils.data.TensorDataset(
                torch.tensor(task.validation_values),
                torch.tensor(task.validation_targets).reshape(-1, 1),
            )
            task.release()

    def take_stage(
        self,
        iteration: int,
        bitness: int,
        previous_model: nn.Module | None,
    ) -> Stage:
        task = self.generator.next_task()
        assert task is not None, (iteration, bitness)
        assert (task.iteration, task.bitness) == (iteration, bitness), task

        values = [torch.tensor(task.train_values)]
        targets = [torch.tensor(task.train_targets)]
        if task.approx_values is not None:
            assert previous_model is not None, bitness
            values.append(torch.tensor(task.approx_values))
            targets.append(torch.from_numpy(self._approximate_targets(previous_model, task)))
        train_dataset = torch.utils.data.TensorDataset(
            torch.cat(values),
            torch.cat(targets).reshape(-1, 1),
        )
        task.release()
        return Stage(iteration, bitness, train_dataset)

    def train_loader(self, stage: Stage, epoch: int) -> torch.utils.data.DataLoader:
        seed = (
            self.training.seed
            + stage.bitness * 10_000
            + stage.iteration * 100
            + epoch
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.utils.data.DataLoader(
            stage.train_dataset,
            batch_size=self.training.batch_size,
            shuffle=True,
            generator=generator,
        )

    def validation_loader(self, bitness: int) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self._validation_datasets[bitness],
            batch_size=self.training.batch_size,
            shuffle=False,
        )

    def _approximate_targets(
        self,
        previous_model: nn.Module,
        task: GeneratedTask,
    ) -> np.ndarray:
        bitness = task.bitness
        assert task.approx_values is not None
        cases = len(task.approx_values)
        targets = np.empty(cases, dtype=np.float32)
        start = 0

        with tqdm(
            total=cases,
            desc=f"train:table_restrictions b={bitness}",
        ) as progress:
            for chunk in task.restrictions:
                chunk_cases, _, reps, point_dim = chunk.shape
                predictions = predict_values(
                    previous_model,
                    chunk.reshape(chunk_cases * bitness * 2, reps, point_dim),
                    self.training.batch_size,
                )
                predictions = predictions.reshape(chunk_cases, bitness, 2)
                predictions = np.clip(predictions, 0.0, float(bitness - 1))
                split_targets = predictions.min(axis=2)
                targets[start : start + chunk_cases] = split_targets.max(axis=1)
                start += chunk_cases
                progress.update(chunk_cases)

        assert start == cases, (start, cases)
        return targets
