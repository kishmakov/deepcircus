from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import Config
from src.generator_proxy import GeneratorProxy
from src.model import predict_values

torch.multiprocessing.set_sharing_strategy("file_system")


class Sampler:
    def __init__(self, config: Config, generator: GeneratorProxy):
        self.config = config
        self.training = config.training
        self.generator = generator

        self._val_tree_ids: dict[int, list[int]] = {}
        self._val_table_ids: dict[int, list[int]] = {}
        self._val_x: dict[int, torch.Tensor] = {}
        self._val_y: dict[int, torch.Tensor] = {}

        self._train_tree_ids: dict[int, list[int]] = {}
        self._train_table_solvable_ids: dict[int, list[int]] = {}
        self._train_table_recursive_ids: dict[int, list[int]] = {}
        self._train_x: dict[int, torch.Tensor] = {}
        self._train_y: dict[int, torch.Tensor] = {}


    def val_loader(self, bitness: int) -> torch.utils.data.DataLoader:
        self._ensure_val(bitness)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self._val_x[bitness], self._val_y[bitness]),
            batch_size=self.training.batch_size,
            shuffle=False
        )


    def reset_train(
            self,
            bitness: int,
            iteration: int,
            previous_model: nn.Module | None,
    ) -> None:
        self._train_x.pop(bitness, None)
        self._train_y.pop(bitness, None)
        self._set_train(bitness, iteration, previous_model)


    def train_loader(
            self,
            bitness: int,
            iteration: int,
            epoch: int,
    ) -> torch.utils.data.DataLoader:
        seed = self.training.seed + bitness * 10_000 + iteration * 100 + epoch
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self._train_x[bitness], self._train_y[bitness]),
            batch_size=self.training.batch_size,
            shuffle=True,
            generator=generator,
        )


    def _ensure_val(self, bitness: int) -> None:
        if bitness in self._val_x:
            return

        rng = random.Random(self.training.seed + bitness * 10_000)
        loaders = self._ensure_val_ids(bitness, rng)

        x_parts = []
        y_parts = []
        for ids, value_tensors, depth_tensors in loaders:
            x_parts.append(
                torch.as_tensor(
                    value_tensors(
                        "val",
                        bitness,
                        ids,
                        self.training.points_per_sample,
                        rng.getrandbits(64),
                    ),
                    dtype=torch.float32,
                )
            )
            y_parts.append(
                torch.as_tensor(
                    [bitness - depth for depth in depth_tensors("val", bitness, ids)],
                    dtype=torch.float32,
                )
            )

        assert x_parts, bitness
        self._val_x[bitness] = torch.cat(x_parts)
        self._val_y[bitness] = torch.cat(y_parts).reshape(-1, 1)


    def _ensure_val_ids(self, bitness: int, rng: random.Random) -> list[
        tuple[
            list[int],
            Callable[[str, int, list[int], int, int], np.ndarray],
            Callable[[str, int, list[int]], np.ndarray],
        ]
    ]:
        self._val_tree_ids[bitness] = []
        self._val_table_ids[bitness] = []

        needed = self.training.validation_samples

        min_tree_bitness = self.generator.min_tree_bitness()
        table_solvable_bitness = self.generator.solvable_bitness()
        assert min_tree_bitness <= table_solvable_bitness, (
            min_tree_bitness,
            table_solvable_bitness,
        )

        if bitness < min_tree_bitness:
            table_needed = needed
            tree_needed = 0
        elif bitness <= table_solvable_bitness:
            assert needed % 2 == 0, needed
            table_needed = needed // 2
            tree_needed = needed // 2
        else:
            table_needed = 0
            tree_needed = needed

        loaders = []
        if table_needed > 0:
            provided = self.generator.table_cases_number(bitness)
            assert provided >= table_needed, (bitness, provided, table_needed)
            self._val_table_ids[bitness] = rng.sample(range(provided), table_needed)
            loaders.append(
                (
                    self._val_table_ids[bitness],
                    self.generator.table_value_tensors,
                    self.generator.table_depth_tensors,
                )
            )
        if tree_needed > 0:
            provided = self.generator.tree_cases_number(bitness)
            assert provided >= tree_needed, (bitness, provided, tree_needed)
            self._val_tree_ids[bitness] = rng.sample(range(provided), tree_needed)
            loaders.append(
                (
                    self._val_tree_ids[bitness],
                    self.generator.tree_value_tensors,
                    self.generator.tree_depth_tensors,
                )
            )
        return loaders


    def _set_train(self, bitness: int, iteration: int, previous_model: nn.Module | None) -> None:
        seed = self.training.seed + bitness * 10_000 + iteration * 100
        rng = random.Random(seed)
        self._set_train_ids(bitness, rng)

        x_parts = []
        y_parts = []

        table_solvable_ids = self._train_table_solvable_ids[bitness]
        if table_solvable_ids:
            x_parts.append(
                torch.as_tensor(
                    self.generator.table_value_tensors(
                        "train",
                        bitness,
                        table_solvable_ids,
                        self.training.points_per_sample,
                        rng.getrandbits(64),
                    ),
                    dtype=torch.float32,
                )
            )
            depths = self.generator.table_depth_tensors("train", bitness, table_solvable_ids)
            y_parts.append(torch.as_tensor(bitness - depths, dtype=torch.float32))

        table_recursive_ids = self._train_table_recursive_ids[bitness]
        if table_recursive_ids:
            assert previous_model is not None, bitness
            x_parts.append(
                torch.as_tensor(
                    self.generator.table_value_tensors(
                        "train",
                        bitness,
                        table_recursive_ids,
                        self.training.points_per_sample,
                        rng.getrandbits(64),
                    ),
                    dtype=torch.float32,
                )
            )
            y_parts.append(
                torch.as_tensor(
                    self._approximate_recursive_targets(
                        previous_model,
                        bitness,
                        table_recursive_ids,
                        self.training.points_per_sample,
                        rng,
                    ),
                    dtype=torch.float32,
                )
            )

        tree_ids = self._train_tree_ids[bitness]
        if tree_ids:
            x_parts.append(
                torch.as_tensor(
                    self.generator.tree_value_tensors(
                        "train",
                        bitness,
                        tree_ids,
                        self.training.points_per_sample,
                        rng.getrandbits(64),
                    ),
                    dtype=torch.float32,
                )
            )
            depths = self.generator.tree_depth_tensors("train", bitness, tree_ids)
            y_parts.append(torch.as_tensor(bitness - depths, dtype=torch.float32))

        assert x_parts, bitness
        self._train_x[bitness] = torch.cat(x_parts)
        self._train_y[bitness] = torch.cat(y_parts).reshape(-1, 1)


    def _set_train_ids(self, bitness: int, rng: random.Random) -> None:
        self._train_tree_ids[bitness] = []
        self._train_table_solvable_ids[bitness] = []
        self._train_table_recursive_ids[bitness] = []

        needed = self.training.train_samples

        min_tree_bitness = self.generator.min_tree_bitness()
        table_solvable_bitness = self.generator.solvable_bitness()
        assert min_tree_bitness <= table_solvable_bitness, (
            min_tree_bitness,
            table_solvable_bitness,
        )

        if bitness < min_tree_bitness:
            table_solvable = needed
            table_recursive = 0
            tree = 0
        elif bitness <= table_solvable_bitness:
            assert needed % 2 == 0, needed
            table_solvable = needed // 2
            table_recursive = 0
            tree = needed // 2
        else:
            assert needed % 2 == 0, needed
            table_solvable = 0
            table_recursive = needed // 2
            tree = needed // 2

        if table_solvable > 0:
            provided = self.generator.table_cases_number(bitness)
            assert provided >= table_solvable, (bitness, provided, table_solvable)
            self._train_table_solvable_ids[bitness] = rng.sample(range(provided), table_solvable)

        if table_recursive > 0:
            provided = self.generator.table_cases_number(bitness)
            assert provided >= table_recursive, (bitness, provided, table_recursive)
            self._train_table_recursive_ids[bitness] = rng.sample(range(provided), table_recursive)

        if tree > 0:
            provided = self.generator.tree_cases_number(bitness)
            assert provided >= tree, (bitness, provided, tree)
            self._train_tree_ids[bitness] = rng.sample(range(provided), tree)


    def _approximate_recursive_targets(
            self,
            previous_model: nn.Module,
            bitness: int,
            case_ids: list[int],
            reps: int,
            rng: random.Random,
    ) -> np.ndarray:
        target_parts = []
        batch_size = self.training.batch_size
        ranges = range(0, len(case_ids), batch_size)
        assert len(case_ids) % batch_size == 0, (len(case_ids), batch_size)

        with tqdm(
                total=len(case_ids),
                desc=f"train:table_restrictions b={bitness}",
        ) as restriction_progress:
            for start in ranges:
                batch_ids = case_ids[start : start + batch_size]
                x_restricted = self.generator.restrictions_tensors(
                    "table_restrictions",
                    bitness,
                    batch_ids,
                    reps,
                    rng.getrandbits(64),
                    restriction_progress,
                )
                predictions = predict_values(
                    previous_model,
                    x_restricted,
                    self.training.batch_size,
                )
                predictions = predictions.reshape(len(batch_ids), bitness, 2)
                predictions = np.clip(predictions, 0.0, float(bitness - 1))
                split_targets = predictions.min(axis=2)
                target_parts.append(split_targets.max(axis=1))

        return np.concatenate(target_parts).astype(np.float32)
