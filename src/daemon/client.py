"""The wire: the daemon process, its commands, and its shared memory.

Nothing outside this package should have to know any of it exists. What leaves
here is arrays.
"""

from __future__ import annotations

import os
import socket
import struct
import subprocess
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np


SERVER_NAME = "offline_server"
MODELS = ("m1", "m2")
_CONSOLE_PREFIX = "\033[32m[client.py]\033[0m "

_HEADER = struct.Struct("<IQ")
_INITIALIZE_PAYLOAD = struct.Struct("<HHHHQ")
_DATASET = struct.Struct("<HHIIII")
_EPOCH_PAYLOAD = struct.Struct("<I")
_REDUCTIONS_PAYLOAD = struct.Struct("<II")
_CASES = struct.Struct("<IQQ")
_TARGETS_DESCRIPTOR = struct.Struct("<QQ")
_STRING_SIZE = struct.Struct("<I")

_INITIALIZE = 1
_EPOCH = 2
_PRIMARY_REDUCTIONS = 3
_HELPER_REDUCTIONS = 4
_SET_TARGETS = 5

# The daemon's own convention: epoch 0 is the validation file, and every epoch
# above it is the training one, sampled at inputs of that epoch's own.
VALIDATION_EPOCH = 0


def find_server() -> Path:
    override = os.environ.get("OFFLINE_SERVER")
    if override:
        return Path(override)
    root = Path(__file__).resolve().parents[2]
    published = root / "execs" / SERVER_NAME
    return published if published.exists() else root / "cpp" / "build" / SERVER_NAME


@dataclass(frozen=True)
class DatasetSizes:
    """What the daemon found in the two files it was pointed at."""

    bitness: int
    point_dim: int
    validation_known: int
    validation_entries: int
    train_known: int
    train_entries: int

    @property
    def unknown_train(self) -> int:
        """Training entries whose targets are reconstructed before epochs begin."""
        return self.train_entries - self.train_known


@dataclass(frozen=True)
class Cases:
    """One epoch of cases: packed rows, and two scores per row.

    Values stay bit-packed as the daemon wrote them -- `columns` bits per row,
    padded to whole bytes -- and are expanded to ±1 float32 on the device.
    """

    values: np.ndarray
    targets: np.ndarray
    columns: int


class Client:
    """One daemon, one dataset, one epoch fetched at a time."""

    def __init__(
            self,
            model_name: str,
            bitness: int,
            data_dir: Path,
            seed: int,
            batches: int,
            points_in_batch: int,
            server_path: Path | None = None,
    ):
        assert model_name in MODELS, model_name
        self.server_path = server_path or find_server()
        assert self.server_path.exists(), (
            f"{self.server_path} is not built; run scripts/train/build_server.sh"
        )
        self.bitness = bitness
        self._first_training_fetch = True
        self._process: subprocess.Popen[str] | None = None
        self._socket: socket.socket | None = None
        try:
            self._connect()
            self.sizes = self._initialize(model_name, data_dir, seed, batches, points_in_batch)
        except BaseException:
            self._close(check_return_code=False)
            raise

    def _connect(self) -> None:
        self._process = subprocess.Popen(
            [str(self.server_path), "--port", "0"],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert self._process.stdout is not None
        ready = self._process.stdout.readline().strip()
        assert ready.startswith("PORT "), ready
        self._socket = socket.create_connection(("127.0.0.1", int(ready.removeprefix("PORT "))))

    def _initialize(
            self,
            model_name: str,
            data_dir: Path,
            seed: int,
            batches: int,
            points_in_batch: int,
    ) -> DatasetSizes:
        directory = str(data_dir).encode("utf-8")
        payload = _INITIALIZE_PAYLOAD.pack(
            MODELS.index(model_name) + 1,
            self.bitness,
            batches,
            points_in_batch,
            seed,
        ) + _STRING_SIZE.pack(len(directory)) + directory

        response = self._request(_INITIALIZE, payload)
        assert len(response) == _DATASET.size, len(response)
        bitness, point_dim, *counts = _DATASET.unpack(response)
        assert bitness == self.bitness, (bitness, self.bitness)
        return DatasetSizes(bitness, point_dim, *counts)

    def fetch(self, epoch: int) -> Cases:
        """Samples `epoch` and copies it out of shared memory."""
        cases = self._fetch_cases(_EPOCH, _EPOCH_PAYLOAD.pack(epoch), with_targets=True)
        assert cases.columns % self.sizes.point_dim == 0, cases.columns
        if epoch > VALIDATION_EPOCH and self._first_training_fetch:
            size = cases.values.nbytes + cases.targets.nbytes
            print(
                f"{_CONSOLE_PREFIX}fetched training epoch {epoch}: {size:,} "
                "bytes, including packed values and targets",
                flush=True,
            )
            self._first_training_fetch = False
        return cases

    def primary_reductions(self, first: int, count: int) -> Cases:
        cases = self._fetch_cases(
            _PRIMARY_REDUCTIONS,
            _REDUCTIONS_PAYLOAD.pack(first, count),
            with_targets=False,
        )
        assert cases.values.shape[0] == count * 2 * self.bitness, cases.values.shape
        assert cases.columns % (3 * (self.bitness - 1) + 2) == 0, cases.columns
        return cases

    def helper_reductions(self, first: int, count: int) -> Cases:
        cases = self._fetch_cases(
            _HELPER_REDUCTIONS,
            _REDUCTIONS_PAYLOAD.pack(first, count),
            with_targets=False,
        )
        assert cases.values.shape[0] == count * 2, cases.values.shape
        assert cases.columns % self.sizes.point_dim == 0, cases.columns
        return cases

    def set_unknown_targets(self, targets: np.ndarray) -> None:
        assert targets.shape == (self.sizes.unknown_train, 2), targets.shape
        assert targets.dtype == np.float32, targets.dtype
        response = self._request(_SET_TARGETS, targets.tobytes(order="C"))
        assert not response, response

    def _fetch_cases(self, command: int, request: bytes, with_targets: bool) -> Cases:
        payload = self._request(command, request)

        offset = 0
        cases, columns, shared_size = _CASES.unpack_from(payload, offset)
        offset += _CASES.size
        name_size = _STRING_SIZE.unpack_from(payload, offset)[0]
        offset += _STRING_SIZE.size
        name = payload[offset : offset + name_size].decode("ascii")
        offset += name_size
        targets_offset, target_count = _TARGETS_DESCRIPTOR.unpack_from(payload, offset)
        offset += _TARGETS_DESCRIPTOR.size
        assert offset == len(payload), (offset, len(payload))
        assert target_count == (2 * cases if with_targets else 0), (target_count, cases)

        memory = shared_memory.SharedMemory(name=name)
        try:
            assert memory.size == shared_size, (memory.size, shared_size)
            row_bytes = (columns + 7) // 8
            # Copied out, so the epoch outlives the segment the daemon frees
            # when the next one is asked for.
            values = np.array(np.ndarray((cases, row_bytes), dtype=np.uint8, buffer=memory.buf))
            targets = (
                np.array(np.ndarray((cases, 2), dtype=np.float32, buffer=memory.buf, offset=targets_offset))
                if with_targets
                else np.empty((cases, 0), dtype=np.float32)
            )
        finally:
            # Unlinked from here as well as by the daemon; whichever call comes
            # second is the one that finds the name gone.
            memory.close()
            memory.unlink()
        return Cases(values=values, targets=targets, columns=columns)

    def close(self) -> None:
        # Hanging up is the goodbye: the daemon exits once the socket closes.
        self._close(check_return_code=True)

    def _close(self, check_return_code: bool) -> None:
        connected = self._socket is not None
        if connected:
            self._socket.close()
            self._socket = None
        if self._process is not None:
            # Before connection there is nobody to hang up; stop a daemon that
            # is still waiting in accept().
            if not connected and self._process.poll() is None:
                self._process.terminate()
            return_code = self._process.wait()
            self._process = None
            if check_return_code:
                assert return_code == 0, return_code

    def _request(self, command: int, payload: bytes = b"") -> bytearray:
        assert self._socket is not None, "closed client"
        self._socket.sendall(_HEADER.pack(command, len(payload)) + payload)
        status, response_size = _HEADER.unpack(self._receive_exact(_HEADER.size))
        assert status == 0, status
        return self._receive_exact(response_size)

    def _receive_exact(self, size: int) -> bytearray:
        assert self._socket is not None
        result = bytearray(size)
        view = memoryview(result)
        offset = 0
        while offset < size:
            count = self._socket.recv_into(view[offset:])
            assert count > 0, "daemon disconnected"
            offset += count
        return result
