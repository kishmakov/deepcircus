from __future__ import annotations

import ctypes
import os
import socket
import struct
import subprocess
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np


_SERVER_NAME = "generator_server"
_HEADER = struct.Struct("<IQ")
_INITIALIZATION = struct.Struct("<QQHHQQQQ")
_TASK_PREFIX = struct.Struct("<QHQQ")
_TENSOR_DESCRIPTOR = struct.Struct("<BHQQQQQQ")

_INITIALIZE = 1
_NEXT = 2
_RELEASE = 3

_KIND_RESTRICTIONS = 2
_NO_OFFSET = (1 << 64) - 1


def _find_server() -> Path:
    override = os.environ.get("GENERATOR_SERVER")
    if override:
        return Path(override)
    root = Path(__file__).resolve().parent.parent
    for build_dir in ("build", "cmake-build-debug"):
        candidate = root / "cpp" / build_dir / _SERVER_NAME
        if candidate.exists():
            return candidate
    return root / "cpp" / "build" / _SERVER_NAME


SERVER = _find_server()


def _load_library() -> ctypes.CDLL:
    library = ctypes.CDLL(str(SERVER.parent / "libgen.so"))
    library.gen_unpack_rows.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.gen_unpack_rows.restype = None
    return library


_LIBRARY = _load_library()


def sample_point_dim(bitness: int) -> int:
    return 2 * bitness + 1


def restriction_point_dim(bitness: int) -> int:
    return sample_point_dim(bitness - 1)


def _row_bytes(columns: int) -> int:
    return (columns + 7) // 8


def _unpack_rows(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Unpacks (rows, row_bytes) little-endian bits into float32 of `shape`.

    The daemon publishes tensors bit-packed, one padded row of bytes per
    case; bit b carries the value 2*b - 1. The expansion runs in C++
    (gen_unpack_rows from libgen.so).
    """
    rows, columns = shape[0], int(np.prod(shape[1:]))
    assert packed.shape == (rows, _row_bytes(columns)), (packed.shape, shape)
    assert packed.flags["C_CONTIGUOUS"], packed.shape
    values = np.empty(shape, dtype=np.float32)
    _LIBRARY.gen_unpack_rows(
        packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        rows,
        columns,
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return values


@dataclass
class GeneratedTask:
    """One (iteration, bitness) coordinate borrowed from the daemon.

    Validation depends only on bitness, so it arrives once per bitness in a
    task with iteration 0, ahead of every training task for that bitness;
    training tasks (iteration >= 1) carry only the train fields. The daemon
    publishes tensors bit-packed; value and target arrays are unpacked into
    owned float32 copies at parse time, so they stay valid after release().
    The restrictions matrix is too large to unpack whole: it stays a packed
    view into shared memory, streamed as float32 case chunks through
    restriction_chunks(), which must be consumed before release(). The
    training tensor comes with exact (cases, 2) target pairs (depth score,
    size score), while `approx_values` (present above the solvable bitness)
    comes with restrictions instead of targets.
    """

    generator: Generator
    iteration: int
    bitness: int
    seed: int
    memory: shared_memory.SharedMemory
    train_values: np.ndarray | None = None
    train_targets: np.ndarray | None = None
    approx_values: np.ndarray | None = None
    validation_values: np.ndarray | None = None
    validation_targets: np.ndarray | None = None
    _restrictions_packed: np.ndarray | None = None
    _restrictions_shape: tuple[int, ...] | None = None

    def restriction_chunks(self, chunk_cases: int = 256):
        assert self._restrictions_packed is not None, "restrictions missing or already released"
        cases = self._restrictions_shape[0]
        for start in range(0, cases, chunk_cases):
            stop = min(start + chunk_cases, cases)
            chunk_shape = (stop - start, *self._restrictions_shape[1:])
            yield _unpack_rows(self._restrictions_packed[start:stop], chunk_shape)

    def release(self) -> None:
        self._restrictions_packed = None
        self._restrictions_shape = None
        self.memory.close()
        self.memory.unlink()
        self.generator._release_task(self)


class Generator:
    """Client of the generator daemon's task protocol."""

    def __init__(self, server_path: Path = SERVER, workers: int = 1):
        assert workers > 0, workers
        self.server_path = str(server_path)
        self.workers = workers
        self._process: subprocess.Popen[str] | None = None
        self._socket: socket.socket | None = None
        self._initialized = False
        self._borrowed: GeneratedTask | None = None
        self._connect()

    def _connect(self) -> None:
        host = os.environ.get("GENERATOR_HOST")
        port_text = os.environ.get("GENERATOR_PORT")
        if host is not None or port_text is not None:
            assert host is not None and port_text is not None, (host, port_text)
            self._socket = socket.create_connection((host, int(port_text)))
            return

        self._process = subprocess.Popen(
            [self.server_path, "--port", "0", "--workers", str(self.workers)],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert self._process.stdout is not None
        ready = self._process.stdout.readline().strip()
        assert ready.startswith("PORT "), ready
        port = int(ready.removeprefix("PORT "))
        self._socket = socket.create_connection(("127.0.0.1", port))

    def initialize(
        self,
        first_iteration: int,
        last_iteration: int,
        bitness_from: int,
        bitness_to: int,
        seed: int,
        train_samples: int,
        validation_samples: int,
        points_per_sample: int,
    ) -> None:
        assert not self._initialized
        payload = _INITIALIZATION.pack(
            first_iteration,
            last_iteration,
            bitness_from,
            bitness_to,
            seed,
            train_samples,
            validation_samples,
            points_per_sample,
        )
        response = self._request(_INITIALIZE, payload)
        assert not response, response
        self._initialized = True

    def next_task(self) -> GeneratedTask | None:
        assert self._initialized, "generator not initialized"
        assert self._borrowed is None, "previous task not released"
        payload = self._request(_NEXT)
        assert payload, payload
        if payload[0] == 1:
            assert len(payload) == 1, len(payload)
            return None
        assert payload[0] == 0, payload[0]

        offset = 1
        iteration, bitness, seed, shared_size = _TASK_PREFIX.unpack_from(payload, offset)
        offset += _TASK_PREFIX.size
        name_size = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        name = payload[offset : offset + name_size].decode("ascii")
        offset += name_size
        tensors = struct.unpack_from("<I", payload, offset)[0]
        offset += 4

        memory = shared_memory.SharedMemory(name=name)
        assert memory.size == shared_size, (memory.size, shared_size)
        task = GeneratedTask(self, iteration, bitness, seed, memory)

        for _ in range(tensors):
            kind, tensor_bitness, cases, reps, values_offset, value_count, targets_offset, target_count = (
                _TENSOR_DESCRIPTOR.unpack_from(payload, offset)
            )
            offset += _TENSOR_DESCRIPTOR.size
            assert tensor_bitness == bitness, (tensor_bitness, bitness)
            if kind == _KIND_RESTRICTIONS:
                shape = (cases, 2 * bitness, reps, restriction_point_dim(bitness))
            else:
                shape = (cases, reps, sample_point_dim(bitness))
            assert int(np.prod(shape)) == value_count, (shape, value_count)
            packed = np.ndarray(
                (cases, _row_bytes(value_count // cases)),
                dtype=np.uint8,
                buffer=memory.buf,
                offset=values_offset,
            )
            targets = None
            if target_count:
                assert target_count == 2 * cases, (target_count, cases)
                targets = np.array(np.ndarray((cases, 2), dtype=np.float32, buffer=memory.buf, offset=targets_offset))
            else:
                assert targets_offset == _NO_OFFSET, targets_offset

            if kind == _KIND_RESTRICTIONS:
                assert task.approx_values is not None, kind
                assert task._restrictions_packed is None
                task._restrictions_packed = packed
                task._restrictions_shape = shape
            elif targets is None:
                assert task.approx_values is None
                task.approx_values = _unpack_rows(packed, shape)
            elif iteration == 0:
                assert task.validation_values is None
                task.validation_values = _unpack_rows(packed, shape)
                task.validation_targets = targets
            else:
                assert task.train_values is None
                task.train_values = _unpack_rows(packed, shape)
                task.train_targets = targets
        assert offset == len(payload), (offset, len(payload))
        is_validation = task.validation_values is not None
        assert is_validation != (task.train_values is not None), (iteration, bitness)
        assert is_validation == (iteration == 0), (iteration, bitness)

        self._borrowed = task
        return task

    def _release_task(self, task: GeneratedTask) -> None:
        assert self._borrowed is task
        response = self._request(_RELEASE)
        assert not response, response
        self._borrowed = None

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
        assert self._socket is not None, "closed generator"
        self._socket.sendall(_HEADER.pack(command, len(payload)) + payload)
        header = self._receive_exact(_HEADER.size)
        status, response_size = _HEADER.unpack(header)
        assert status == 0, status
        return self._receive_exact(response_size)

    def _receive_exact(self, size: int) -> bytearray:
        assert self._socket is not None
        result = bytearray(size)
        view = memoryview(result)
        offset = 0
        while offset < size:
            count = self._socket.recv_into(view[offset:])
            assert count > 0, "generator server disconnected"
            offset += count
        return result


def load_generator(workers: int = 1) -> Generator:
    return Generator(SERVER, workers)
