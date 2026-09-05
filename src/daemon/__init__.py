"""Where the training data comes from.

The rest of `src/` asks a `TrainingData` for an epoch and gets arrays back. That
the arrays are sampled by a C++ process over a socket, and that each epoch draws
fresh inputs for the same pairs of functions, is this package's business and
nobody else's.

    with open_training_data(config) as data:
        values, targets = data.validation()
        for epoch in range(1, epochs + 1):
            values, targets = data.epoch(epoch)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np

from src.daemon.client import VALIDATION_EPOCH, Cases, Client, DatasetSizes
from src.config import TrainConfig


__all__ = ["DatasetSizes", "TrainingData", "open_training_data"]


class TrainingData:
    """Both splits of one `(model, bitness)` coordinate, an epoch at a time."""

    def __init__(self, config: TrainConfig):
        self._client = Client(
            model_name=config.model_name,
            bitness=config.bitness,
            data_dir=config.data_dir,
            seed=config.seed,
            batches=config.sampling.batches,
            points_in_batch=config.sampling.points_in_batch,
        )
        try:
            assert self.sizes.point_dim == config.point_dim, (self.sizes.point_dim, config.point_dim)
            assert self.sizes.validation_entries == self.sizes.validation_known, self.sizes
            if self.sizes.unknown_train:
                from src.bootstrap import reconstruct_unknown_targets

                self._client.set_unknown_targets(reconstruct_unknown_targets(config, self._client))
        except BaseException:
            self._client.close()
            raise

    @property
    def sizes(self) -> DatasetSizes:
        return self._client.sizes

    def validation(self) -> tuple[np.ndarray, np.ndarray]:
        """The validation cases, the same ones however often they are asked for."""
        return _arrays(self._client.fetch(VALIDATION_EPOCH))

    def epoch(self, epoch: int) -> tuple[np.ndarray, np.ndarray]:
        """The training cases, sampled at inputs this epoch draws for itself."""
        assert epoch > VALIDATION_EPOCH, epoch
        return _arrays(self._client.fetch(epoch))

    def close(self) -> None:
        self._client.close()


@contextmanager
def open_training_data(config: TrainConfig) -> Iterator[TrainingData]:
    data = TrainingData(config)
    try:
        yield data
    finally:
        data.close()


def _arrays(cases: Cases) -> tuple[np.ndarray, np.ndarray]:
    return cases.values, cases.targets
