from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import Config
from src.model import predict_values


BOOL_BENCH_DIR = Path(__file__).resolve().parents[1] / "bool-bench"
if str(BOOL_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BOOL_BENCH_DIR))

from bool_bench import GeneratedData, Generator, load_generator  # noqa: E402


class GeneratorProxy:
    """Thin compatibility wrapper around the C++ threaded generator."""

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
class GeneratedStage:
    iteration: int
    bitness: int
    train_data: list[GeneratedData]
    validation_data: list[GeneratedData]
    acquired: bool = False
    train_dataset: torch.utils.data.Dataset | None = None
    validation_dataset: torch.utils.data.Dataset | None = None
    python_targets: list[np.ndarray] = field(default_factory=list)


class Sampler:
    def __init__(self, config: Config, generator: GeneratorProxy):
        self.config = config
        self.training = config.training
        self.generator = generator

    def request_stage(self, iteration: int, bitness: int) -> GeneratedStage:
        return GeneratedStage(
            iteration=iteration,
            bitness=bitness,
            train_data=self._request_train(iteration, bitness),
            validation_data=self._request_validation(bitness),
        )

    def acquire_stage(self, stage: GeneratedStage) -> None:
        assert not stage.acquired, (stage.iteration, stage.bitness)
        for data in stage.train_data:
            data.acquire()
        for data in stage.validation_data:
            data.acquire()
        stage.acquired = True

    def prepare_stage(
        self,
        stage: GeneratedStage,
        previous_model: nn.Module | None,
    ) -> None:
        assert stage.acquired, (stage.iteration, stage.bitness)
        assert stage.train_dataset is None
        assert stage.validation_dataset is None

        train_parts = []
        for data in stage.train_data:
            targets = data.targets
            if targets is None:
                assert previous_model is not None, stage.bitness
                targets = self._approximate_recursive_targets(previous_model, data)
                stage.python_targets.append(targets)
            train_parts.append(self._tensor_dataset(data, targets))

        validation_parts = [
            self._tensor_dataset(data, data.targets)
            for data in stage.validation_data
        ]
        assert train_parts, stage.bitness
        assert validation_parts, stage.bitness
        stage.train_dataset = torch.utils.data.ConcatDataset(train_parts)
        stage.validation_dataset = torch.utils.data.ConcatDataset(validation_parts)

    def train_loader(
        self,
        stage: GeneratedStage,
        epoch: int,
    ) -> torch.utils.data.DataLoader:
        assert stage.train_dataset is not None
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

    def validation_loader(
        self,
        stage: GeneratedStage,
    ) -> torch.utils.data.DataLoader:
        assert stage.validation_dataset is not None
        return torch.utils.data.DataLoader(
            stage.validation_dataset,
            batch_size=self.training.batch_size,
            shuffle=False,
        )

    def release_stage(self, stage: GeneratedStage) -> None:
        assert stage.acquired, (stage.iteration, stage.bitness)
        stage.train_dataset = None
        stage.validation_dataset = None
        stage.python_targets.clear()
        for data in stage.train_data:
            data.release()
        for data in stage.validation_data:
            data.release()
        stage.train_data.clear()
        stage.validation_data.clear()
        stage.acquired = False

    def _request_train(self, iteration: int, bitness: int) -> list[GeneratedData]:
        needed = self.training.train_samples
        min_tree = self.generator.min_tree_bitness()
        solvable = self.generator.table_solvable_bitness()
        assert min_tree <= solvable, (min_tree, solvable)

        if bitness < min_tree:
            table_cases = needed
            tree_cases = 0
        else:
            assert needed % 2 == 0, needed
            table_cases = needed // 2
            tree_cases = needed // 2

        result = []
        if table_cases:
            recursive = bitness > solvable
            chunk_cases = self.training.batch_size if recursive else 0
            assert table_cases % max(chunk_cases, 1) == 0, (
                table_cases,
                chunk_cases,
            )
            result.append(self.generator.table_value_tensor(
                bitness,
                table_cases,
                self.training.points_per_sample,
                chunk_cases,
                self._seed(iteration, bitness, 0x1001),
            ))
        if tree_cases:
            result.append(self.generator.tree_value_tensor(
                bitness,
                tree_cases,
                self.training.points_per_sample,
                self._seed(iteration, bitness, 0x1002),
            ))
        return result

    def _request_validation(self, bitness: int) -> list[GeneratedData]:
        needed = self.training.validation_samples
        min_tree = self.generator.min_tree_bitness()
        solvable = self.generator.table_solvable_bitness()

        if bitness < min_tree:
            table_cases = needed
            tree_cases = 0
        elif bitness <= solvable:
            assert needed % 2 == 0, needed
            table_cases = needed // 2
            tree_cases = needed // 2
        else:
            table_cases = 0
            tree_cases = needed

        result = []
        if table_cases:
            result.append(self.generator.table_value_tensor(
                bitness,
                table_cases,
                self.training.points_per_sample,
                0,
                self._seed(0, bitness, 0x2001),
            ))
        if tree_cases:
            result.append(self.generator.tree_value_tensor(
                bitness,
                tree_cases,
                self.training.points_per_sample,
                self._seed(0, bitness, 0x2002),
            ))
        return result

    def _seed(self, iteration: int, bitness: int, domain: int) -> int:
        return (
            self.training.seed
            + bitness * 10_000
            + iteration * 100
            + domain
        ) & ((1 << 64) - 1)

    def _tensor_dataset(
        self,
        data: GeneratedData,
        targets: np.ndarray | None,
    ) -> torch.utils.data.TensorDataset:
        assert data.values is not None
        assert targets is not None
        assert data.cases is not None
        assert targets.shape == (data.cases,), targets.shape
        return torch.utils.data.TensorDataset(
            torch.from_numpy(data.values),
            torch.from_numpy(targets).reshape(-1, 1),
        )

    def _approximate_recursive_targets(
        self,
        previous_model: nn.Module,
        data: GeneratedData,
    ) -> np.ndarray:
        assert data.bitness is not None
        assert data.cases is not None
        assert data.reps is not None
        bitness = data.bitness
        cases = data.cases
        chunk_cases = self.training.batch_size
        assert cases % chunk_cases == 0, (cases, chunk_cases)

        targets = np.empty(cases, dtype=np.float32)
        starts = list(range(0, cases, chunk_cases))
        current = data.restrictions(starts[0], chunk_cases)

        with tqdm(
            total=cases,
            desc=f"train:table_restrictions b={bitness}",
        ) as progress:
            for chunk_id, start in enumerate(starts):
                restricted = current.acquire()
                next_tensor = None
                if chunk_id + 1 < len(starts):
                    next_tensor = data.restrictions(
                        starts[chunk_id + 1],
                        chunk_cases,
                    )

                predictions = predict_values(
                    previous_model,
                    restricted.reshape(
                        chunk_cases * bitness * 2,
                        data.reps,
                        2 * bitness - 1,
                    ),
                    self.training.batch_size,
                )
                predictions = predictions.reshape(chunk_cases, bitness, 2)
                predictions = np.clip(predictions, 0.0, float(bitness - 1))
                split_targets = predictions.min(axis=2)
                targets[start : start + chunk_cases] = split_targets.max(axis=1)

                del restricted
                current.release()
                progress.update(chunk_cases)
                if next_tensor is not None:
                    current = next_tensor

        return targets
