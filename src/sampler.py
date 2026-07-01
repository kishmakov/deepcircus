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

torch.multiprocessing.set_sharing_strategy("file_system")


_SAMPLE_TENSOR_CACHE = {}
_RESTRICTION_TENSOR_CACHE = {}
_SAMPLE_TARGET_CACHE = {}


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
    ) -> torch.utils.data.DataLoader:
        inputs = self.inputs(bitness, iteration)
        targets = self.targets(inputs, bitness, iteration, previous_model)
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
            "n_points": self.reps,
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
