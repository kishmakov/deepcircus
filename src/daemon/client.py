"""The wire: the daemon process, its two commands, and its shared memory.

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

_HEADER = struct.Struct("<IQ")
_INITIALIZE_PAYLOAD = struct.Struct("<HHHHQ")
_DATASET = struct.Struct("<HHIIII")
_EPOCH_PAYLOAD = struct.Struct("<I")
_CASES = struct.Struct("<IQQ")
_TARGETS = struct.Struct("<QQ")
_STRING_SIZE = struct.Struct("<I")

_INITIALIZE = 1
_EPOCH = 2

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
    validation_cases: int
    validation_entries: int
    train_cases: int
    train_entries: int

    @property
    def skipped(self) -> int:
        """Entries left behind because their target is the unknown marker."""
        return (self.train_entries - self.train_cases) + (
            self.validation_entries - self.validation_cases
        )


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
        self._process: subprocess.Popen[str] | None = None
        self._socket: socket.socket | None = None
        self._connect()
        self.sizes = self._initialize(model_name, data_dir, seed, batches, points_in_batch)

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
        payload = self._request(_EPOCH, _EPOCH_PAYLOAD.pack(epoch))

        offset = 0
        cases, columns, shared_size = _CASES.unpack_from(payload, offset)
        offset += _CASES.size
        name_size = _STRING_SIZE.unpack_from(payload, offset)[0]
        offset += _STRING_SIZE.size
        name = payload[offset : offset + name_size].decode("ascii")
        offset += name_size
        targets_offset, target_count = _TARGETS.unpack_from(payload, offset)
        offset += _TARGETS.size
        assert offset == len(payload), (offset, len(payload))
        assert columns == self.sizes.point_dim * (columns // self.sizes.point_dim), columns
        assert target_count == 2 * cases, (target_count, cases)

        memory = shared_memory.SharedMemory(name=name)
        try:
            assert memory.size == shared_size, (memory.size, shared_size)
            row_bytes = (columns + 7) // 8
            # Copied out, so the epoch outlives the segment the daemon frees
            # when the next one is asked for.
            values = np.array(np.ndarray((cases, row_bytes), dtype=np.uint8, buffer=memory.buf))
            targets = np.array(
                np.ndarray((cases, 2), dtype=np.float32, buffer=memory.buf, offset=targets_offset)
            )
        finally:
            # Unlinked from here as well as by the daemon; whichever call comes
            # second is the one that finds the name gone.
            memory.close()
            memory.unlink()
        return Cases(values=values, targets=targets, columns=columns)

    def close(self) -> None:
        # Hanging up is the goodbye: the daemon exits once the socket closes.
        if self._socket is None:
            return
        self._socket.close()
        self._socket = None
        if self._process is not None:
            return_code = self._process.wait()
            assert return_code == 0, return_code
            self._process = None

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
