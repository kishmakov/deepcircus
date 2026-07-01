from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

BOOL_BENCH_DIR = Path(__file__).resolve().parents[1] / "bool-bench"
if str(BOOL_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BOOL_BENCH_DIR))

from bool_bench import (
    sample_point_dim,
)

from src.config import Config
from src.generator_proxy import GeneratorProxy
from src.model import predict_values

torch.multiprocessing.set_sharing_strategy("file_system")


_SAMPLE_TENSOR_CACHE = {}
_RESTRICTION_TENSOR_CACHE = {}
_SAMPLE_TARGET_CACHE = {}
TARGET_CASE_BATCH_SIZE = 512


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
        self._targets: dict[tuple[int, int], torch.Tensor] = {}
        self._case_ids: dict[tuple[int, int], list[int]] = {}
        self._random_case_mask: dict[tuple[int, int], list[bool]] = {}

    def train_loader(
            self,
            bitness: int,
            iteration: int,
            previous_model: nn.Module | None,
            epoch: int,
    ) -> torch.utils.data.DataLoader:
        inputs, targets = self.samples(bitness, iteration, previous_model)
        generator = torch.Generator()
        generator.manual_seed(
            self.training.seed + iteration * 10_000 + bitness * 100 + epoch,
        )
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(inputs, targets),
            batch_size=self.training.batch_size,
            shuffle=True,
            generator=generator,
        )

    def samples(
            self,
            bitness: int,
            iteration: int,
            previous_model: nn.Module | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (bitness, iteration)
        if key not in self._inputs:
            self._sample_cases(bitness, iteration, previous_model)
        assert key in self._targets, key
        return self._inputs[key], self._targets[key]

    def case_ids(
            self,
            bitness: int,
            iteration: int,
    ) -> list[int]:
        key = (bitness, iteration)
        if key not in self._case_ids:
            rng = random.Random(self.training.seed + iteration * 10_000 + bitness)
            cases_number = self.generator.cases_number(bitness)
            if bitness > self.generator.solvable_bitness():
                cases_number = min(
                    cases_number,
                    max(
                        self.training.samples_per_model + 1,
                        self.training.samples_per_model * 4,
                    ),
                )
            self._case_ids[key] = rng.sample(
                range(cases_number),
                self.training.samples_per_model,
            )
        return self._case_ids[key]

    def random_case_mask(
            self,
            bitness: int,
            iteration: int,
            previous_model: nn.Module | None,
    ) -> list[bool]:
        key = (bitness, iteration)
        if key in self._random_case_mask:
            return self._random_case_mask[key]

        case_ids = self.case_ids(bitness, iteration)
        if previous_model is None:
            assert bitness == self.training.bitness_from, bitness
            mask = [False] * len(case_ids)
        elif bitness <= self.generator.solvable_bitness():
            mask = [False] * len(case_ids)
        else:
            indices = list(range(len(case_ids)))
            rng = random.Random(
                self.training.seed + iteration * 10_000 + bitness * 1_000,
            )
            rng.shuffle(indices)
            random_indices = set(indices[len(indices) // 2 :])
            mask = [row_id in random_indices for row_id in range(len(case_ids))]

        self._random_case_mask[key] = mask
        return mask

    def _sample_cases(
            self,
            bitness: int,
            iteration: int,
            previous_model: nn.Module | None,
    ) -> None:
        key = (bitness, iteration)
        case_ids = self.case_ids(bitness, iteration)
        random_mask = self.random_case_mask(bitness, iteration, previous_model)
        input_bits = self._input_bits(bitness, iteration, case_ids)

        x = np.empty(
            (
                len(case_ids),
                self.training.samples_per_case,
                sample_point_dim(bitness),
            ),
            dtype=np.float32,
        )
        y = np.empty(len(case_ids), dtype=np.float32)

        solvable_rows = [
            row_id for row_id, is_random in enumerate(random_mask) if not is_random
        ]
        random_rows = [
            row_id for row_id, is_random in enumerate(random_mask) if is_random
        ]

        if solvable_rows:
            solvable_ids = [case_ids[row_id] for row_id in solvable_rows]
            solvable_inputs = [input_bits[row_id] for row_id in solvable_rows]
            x[solvable_rows] = self.generator.generate_value_tensors(
                bitness,
                solvable_ids,
                solvable_inputs,
            )
            y[solvable_rows] = self.generator.generate_depths_tensors(
                bitness,
                solvable_ids,
            )

        if random_rows:
            assert previous_model is not None, (bitness, iteration)
            random_ids = [case_ids[row_id] for row_id in random_rows]
            random_inputs = [input_bits[row_id] for row_id in random_rows]
            x[random_rows] = self.generator.generate_value_tensors_rnd(
                bitness,
                random_ids,
                random_inputs,
            )
            y[random_rows] = self._approximate_random_depths(
                previous_model,
                bitness,
                random_ids,
            )

        self._inputs[key] = torch.from_numpy(x)
        self._targets[key] = torch.from_numpy(y.reshape(-1, 1))

    def _input_bits(
            self,
            bitness: int,
            iteration: int,
            case_ids: list[int],
    ) -> list[list[str]]:
        assert self.training.samples_per_case % 2 == 0, self.training.samples_per_case
        return [
            self._mixed_input_bits(bitness, iteration, case_id)
            for case_id in case_ids
        ]

    def _mixed_input_bits(
            self,
            bitness: int,
            iteration: int,
            case_id: int,
    ) -> list[str]:
        half = self.training.samples_per_case // 2
        rng = random.Random(
            self.training.seed
            + iteration * 10_000
            + bitness * 1_000_000
            + case_id
        )
        block = block_inversion_input_bits(bitness, half, rng)
        random_bits = random_input_bits(bitness, half, rng)
        return block + random_bits

    def _approximate_random_depths(
            self,
            previous_model: nn.Module,
            bitness: int,
            case_ids: list[int],
    ) -> np.ndarray:
        target_parts = []
        ranges = range(0, len(case_ids), TARGET_CASE_BATCH_SIZE)
        total_batches = (
            len(case_ids) + TARGET_CASE_BATCH_SIZE - 1
        ) // TARGET_CASE_BATCH_SIZE
        for start in tqdm(ranges, total=total_batches, desc=f"targets_rnd b={bitness}"):
            batch_ids = case_ids[start : start + TARGET_CASE_BATCH_SIZE]
            x_restricted = self.generator.generate_restriction_tensors_rnd(
                bitness,
                batch_ids,
                self.training.samples_per_case,
            )
            predictions = predict_values(
                previous_model,
                x_restricted,
                self.training.batch_size,
            )
            predictions = predictions.reshape(len(batch_ids), bitness, 2)
            predictions = np.clip(predictions, 0.0, float(bitness - 1))
            split_depths = 1.0 + predictions.max(axis=2)
            target_parts.append(split_depths.min(axis=1))

        return np.concatenate(target_parts).astype(np.float32)


class DepthSampler:
    def __init__(
        self,
        generator: GeneratorProxy,
        bitness: int,
        seed: int,
        train_size: int,
        validation_size: int,
        name: str,
        method: str,
        reps: int,
        batch_size: int,
    ):
        self.generator = generator
        self.bitness = bitness
        self.seed = seed
        self.train_size = train_size
        self.validation_size = validation_size
        self.name = name
        self.method = method
        self.reps = reps
        self.batch_size = batch_size

        ids = generate_ids(generator, bitness, train_size + validation_size, seed)
        train_ids = ids[:train_size]
        validation_ids = ids[train_size:]
        self.train_ids = [int(case_id) for case_id in train_ids]
        self.validation_ids = [int(case_id) for case_id in validation_ids]
        assert len(self.train_ids) == train_size
        assert len(self.validation_ids) == validation_size
        self._train_samples = None
        self._validation_samples = None

    def training_inputs(
        self,
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        return self.train_loader(), self.validation_loader()

    def model_params(self) -> dict[str, int]:
        return {
            "point_dim": sample_point_dim(self.bitness),
        }

    def train_loader(self) -> torch.utils.data.DataLoader:
        if self._train_samples is None:
            self._train_samples = self._sample_cases(self.train_ids)
        return self._loader(*self._train_samples, True)

    def validation_loader(self) -> torch.utils.data.DataLoader:
        if self._validation_samples is None:
            self._validation_samples = self._sample_cases(self.validation_ids)
        return self._loader(*self._validation_samples, False)

    def parity_inputs(self) -> np.ndarray:
        input_bits = self._sample_input_bits(self.seed, self.reps)
        samples = np.empty(
            (self.reps, sample_point_dim(self.bitness)),
            dtype=np.float32,
        )
        for row_id, bits in enumerate(input_bits):
            samples[row_id] = self.generator.parity_value(self.bitness, bits)
        return samples

    def _sample_cases(
        self,
        case_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        input_bits = [
            self._sample_input_bits(case_id, self.reps)
            for case_id in tqdm(
                case_ids,
                desc=f"inputs {self.method}",
            )
        ]
        x = self.generator.generate_value_tensors(self.bitness, case_ids, input_bits)
        y = self.generator.generate_depths_tensors(self.bitness, case_ids)
        print(
            f"Generated {len(x)} depth samples; "
            f"sample_shape={tuple(x.shape[1:])}"
        )
        if len(x) > 0:
            print("First sample:")
            for rep in _sample_to_bit_strings(x[0]):
                print(rep)
        return x, y

    def _loader(
        self,
        x: np.ndarray,
        y: np.ndarray,
        shuffle: bool,
    ) -> torch.utils.data.DataLoader:
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y),
        )
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(shuffle))
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle,
        )

    def _sample_input_bits(
        self,
        case_id: int,
        reps: int,
    ) -> list[str]:
        rng = random.Random((self.bitness << 16) + case_id + self.bitness)
        if self.method == "random":
            return random_input_bits(self.bitness, reps, rng)
        if self.method == "block":
            return block_inversion_input_bits(self.bitness, reps, rng)
        if self.method == "mix":
            l1 = block_inversion_input_bits(self.bitness, reps // 2, rng)
            l2 = random_input_bits(self.bitness, reps // 2, rng)
            return l1 + l2
        assert False, f"Unknown sample mode: {self.method}"


def _case_ids_key(case_ids) -> tuple[int, ...]:
    return tuple(int(case_id) for case_id in case_ids)


def _sample_to_bit_strings(sample: np.ndarray) -> list[str]:
    bits = (sample > 0).astype(np.uint8).T
    return ["".join(str(bit) for bit in row) for row in bits]


def generate_ids(
    generator: GeneratorProxy,
    bitness: int,
    number: int,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    return rng.sample(range(generator.cases_number(bitness)), number)


def random_input_bits(bitness: int, reps: int, rng: random.Random) -> list[str]:
    return ["".join(rng.choice("01") for _ in range(bitness)) for _ in range(reps)]


def block_inversion_input_bits(bitness: int, reps: int, rng: random.Random) -> list[str]:
    assert reps > 0
    blocks = (reps - 1).bit_length()
    base_input = [rng.choice("01") for _ in range(bitness)]
    bit_blocks = _split_bit_blocks(bitness, blocks) if blocks > 0 else []
    samples = []

    for mask in range(reps):
        input_bits = base_input.copy()
        for block_id, bit_ids in enumerate(bit_blocks):
            if ((mask >> block_id) & 1) == 0:
                continue
            for bit_id in bit_ids:
                input_bits[bit_id] = "0" if input_bits[bit_id] == "1" else "1"
        samples.append("".join(input_bits))

    return samples


def _split_bit_blocks(bitness: int, blocks: int) -> list[range]:
    base_size = bitness // blocks
    remainder = bitness % blocks
    result = []
    start = 0
    for block_id in range(blocks):
        size = base_size + (1 if block_id < remainder else 0)
        result.append(range(start, start + size))
        start += size
    return result


def generate_sample_tensors(
    generator: GeneratorProxy,
    bitness: int,
    case_ids: list[int],
    reps: int,
) -> np.ndarray:
    case_ids = list(case_ids)
    cache_key = (bitness, _case_ids_key(case_ids), reps)
    if cache_key in _SAMPLE_TENSOR_CACHE:
        return _SAMPLE_TENSOR_CACHE[cache_key]

    print(f"Generating {len(case_ids)} sample tensors for bitness {bitness}")
    input_bits = [random_input_bits(bitness, case_id, reps) for case_id in case_ids]
    x = generator.generate_value_tensors(bitness, case_ids, input_bits)

    # _SAMPLE_TENSOR_CACHE[cache_key] = x
    return x


def generate_restriction_tensors(
    generator: GeneratorProxy,
    bitness: int,
    case_ids: list[int],
    reps: int,
) -> np.ndarray:
    case_ids = list(case_ids)
    cache_key = (bitness, _case_ids_key(case_ids), reps)
    if cache_key in _RESTRICTION_TENSOR_CACHE:
        return _RESTRICTION_TENSOR_CACHE[cache_key]

    x = generator.generate_restriction_tensors(bitness, case_ids, reps)
    # _RESTRICTION_TENSOR_CACHE[cache_key] = x
    return x


def generate_samples(
    generator: GeneratorProxy,
    bitness: int,
    case_ids: list[int],
    reps: int,
) -> tuple[np.ndarray, np.ndarray]:
    case_ids = list(case_ids)
    cache_key = (bitness, _case_ids_key(case_ids), reps)
    if cache_key in _SAMPLE_TARGET_CACHE:
        return _SAMPLE_TARGET_CACHE[cache_key]

    x = generate_sample_tensors(generator, bitness, case_ids, reps)
    y = generator.generate_node_tensors(bitness, case_ids).astype(np.int64)
    result = (x, y)
    _SAMPLE_TARGET_CACHE[cache_key] = result
    return result
