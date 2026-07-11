from __future__ import annotations

import os
import socket
import struct
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np


_SERVER_NAME = "generator_server"
_PROTOCOL_VERSION = 1
_HEADER = struct.Struct("<IQ")
_U64 = struct.Struct("<Q")


class _Command:
    INFO = 1
    CASES = 2
    TREE_REQUEST = 3
    TABLE_REQUEST = 4
    DATA_ACQUIRE = 5
    RESTRICTIONS_REQUEST = 6
    TENSOR_ACQUIRE = 7
    DATA_RELEASE = 8
    TREE_VALUE = 9
    TABLE_VALUE = 10
    CIRCUIT_SETS = 11
    CIRCUIT_CASES = 12
    CIRCUIT_INPUTS = 13
    CIRCUIT_OUTPUTS = 14
    CIRCUIT_VALUE = 15
    CLOSE = 16
    SHUTDOWN = 17


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


def sample_point_dim(bitness: int) -> int:
    return 2 * bitness + 1


def restriction_point_dim(bitness: int) -> int:
    return sample_point_dim(bitness - 1)


def circuit_sample_point_dim(inputs: int, outputs: int) -> int:
    return inputs + outputs * (inputs + 1)


def _string(value: str) -> bytes:
    encoded = value.encode("ascii")
    return struct.pack("<I", len(encoded)) + encoded


class GeneratedTensor:
    def __init__(
        self,
        generator: Generator,
        handle: int,
        bitness: int,
        cases: int,
        reps: int,
    ) -> None:
        self._generator = generator
        self._handle = handle
        self.bitness = bitness
        self.cases = cases
        self.reps = reps
        self._values: np.ndarray | None = None

    def acquire(self) -> np.ndarray:
        assert self._handle is not None, "released tensor"
        assert self._values is None, "tensor already acquired"
        shape = (
            self.cases,
            2 * self.bitness,
            self.reps,
            restriction_point_dim(self.bitness),
        )
        count = int(np.prod(shape))
        payload = _U64.pack(self._handle)
        buffer = self._generator._request(_Command.TENSOR_ACQUIRE, payload)
        assert len(buffer) == count * np.dtype(np.float32).itemsize, len(buffer)
        self._values = np.frombuffer(buffer, dtype=np.float32).reshape(shape)
        self._generator._forget(self)
        self._handle = None
        return self._values

    def release(self) -> None:
        assert self._values is not None, "tensor not acquired"
        self._values = None

    def _invalidate(self) -> None:
        self._handle = None


class GeneratedData:
    def __init__(
        self,
        generator: Generator,
        handle: int,
        recursive_table: bool,
    ) -> None:
        self._generator = generator
        self._handle = handle
        self.recursive_table = recursive_table
        self.bitness: int | None = None
        self.cases: int | None = None
        self.reps: int | None = None
        self.values: np.ndarray | None = None
        self.targets: np.ndarray | None = None
        self._acquired = False

    def acquire(self) -> GeneratedData:
        assert self._handle is not None, "released data"
        assert self.values is None, "data already acquired"
        buffer = self._generator._request(
            _Command.DATA_ACQUIRE,
            _U64.pack(self._handle),
        )
        metadata = struct.Struct("<HQQB")
        self.bitness, self.cases, self.reps, has_targets = metadata.unpack_from(buffer)
        value_count = self.cases * self.reps * sample_point_dim(self.bitness)
        values_offset = metadata.size
        targets_offset = values_offset + value_count * 4
        self.values = np.frombuffer(
            buffer,
            dtype=np.float32,
            count=value_count,
            offset=values_offset,
        ).reshape(self.cases, self.reps, sample_point_dim(self.bitness))
        if has_targets:
            self.targets = np.frombuffer(
                buffer,
                dtype=np.float32,
                count=self.cases,
                offset=targets_offset,
            )
        else:
            assert self.recursive_table
            self.targets = None
        expected = targets_offset + (self.cases * 4 if has_targets else 0)
        assert len(buffer) == expected, (len(buffer), expected)
        self._acquired = True

        if not self.recursive_table:
            self._generator._forget(self)
            self._handle = None
        return self

    def restrictions(self, first_case: int, cases: int) -> GeneratedTensor:
        assert self._handle is not None, "released data"
        assert self.values is not None, "data not acquired"
        assert self.recursive_table
        assert self.bitness is not None
        assert self.reps is not None
        assert self.cases is not None
        assert cases > 0, cases
        assert first_case + cases <= self.cases, (first_case, cases, self.cases)
        response = self._generator._request(
            _Command.RESTRICTIONS_REQUEST,
            struct.pack("<QQQ", self._handle, first_case, cases),
        )
        handle = _U64.unpack(response)[0]
        tensor = GeneratedTensor(
            self._generator,
            handle,
            self.bitness,
            cases,
            self.reps,
        )
        self._generator._remember(tensor)
        return tensor

    def release(self) -> None:
        assert self._acquired, "data not acquired"
        self.values = None
        self.targets = None
        if self._handle is not None:
            self._generator._request(_Command.DATA_RELEASE, _U64.pack(self._handle))
            self._generator._forget(self)
            self._handle = None
        self._acquired = False

    def _invalidate(self) -> None:
        self._handle = None


class Generator:
    def __init__(self, server_path: Path = SERVER, workers: int = 1):
        assert workers > 0, workers
        self.server_path = str(server_path)
        self.workers = workers
        self._process: subprocess.Popen[str] | None = None
        self._socket: socket.socket | None = None
        self._owned: set[GeneratedData | GeneratedTensor] = set()
        self._info: tuple[int, int] | None = None
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

    def close(self) -> None:
        if self._socket is None:
            return
        command = _Command.SHUTDOWN if self._process is not None else _Command.CLOSE
        self._request(command)
        self._socket.close()
        self._socket = None
        if self._process is not None:
            return_code = self._process.wait()
            assert return_code == 0, return_code
            self._process = None
        for owner in list(self._owned):
            owner._invalidate()
        self._owned.clear()

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

    def _remember(self, owner: GeneratedData | GeneratedTensor) -> None:
        assert owner not in self._owned
        self._owned.add(owner)

    def _forget(self, owner: GeneratedData | GeneratedTensor) -> None:
        assert owner in self._owned
        self._owned.remove(owner)

    def _get_info(self) -> tuple[int, int]:
        if self._info is None:
            version, min_tree, table_solvable = struct.unpack(
                "<IHH",
                self._request(_Command.INFO),
            )
            assert version == _PROTOCOL_VERSION, version
            self._info = (min_tree, table_solvable)
        return self._info

    def tree_cases_number(self, bitness: int) -> int:
        response = self._request(_Command.CASES, struct.pack("<BH", 0, bitness))
        return _U64.unpack(response)[0]

    def table_cases_number(self, bitness: int) -> int:
        response = self._request(_Command.CASES, struct.pack("<BH", 1, bitness))
        return _U64.unpack(response)[0]

    def min_tree_bitness(self) -> int:
        return self._get_info()[0]

    def table_solvable_bitness(self) -> int:
        return self._get_info()[1]

    def tree_value_tensor(
        self,
        bitness: int,
        cases: int,
        reps: int,
        seed: int,
    ) -> GeneratedData:
        response = self._request(
            _Command.TREE_REQUEST,
            struct.pack("<HQQQ", bitness, cases, reps, seed),
        )
        data = GeneratedData(self, _U64.unpack(response)[0], recursive_table=False)
        self._remember(data)
        return data

    def table_value_tensor(
        self,
        bitness: int,
        cases: int,
        reps: int,
        restriction_chunk_cases: int,
        seed: int,
    ) -> GeneratedData:
        recursive = bitness > self.table_solvable_bitness()
        assert (restriction_chunk_cases > 0) == recursive, (
            bitness,
            restriction_chunk_cases,
        )
        response = self._request(
            _Command.TABLE_REQUEST,
            struct.pack(
                "<HQQQQ",
                bitness,
                cases,
                reps,
                restriction_chunk_cases,
                seed,
            ),
        )
        data = GeneratedData(self, _U64.unpack(response)[0], recursive)
        self._remember(data)
        return data

    def tree_value(self, bitness: int, case_id: int, input_bits: str) -> np.ndarray:
        return self._value(_Command.TREE_VALUE, bitness, case_id, input_bits)

    def tree_values(
        self,
        bitness: int,
        case_id: int,
        input_bits: Sequence[str],
    ) -> np.ndarray:
        return self._values(_Command.TREE_VALUE, bitness, case_id, input_bits)

    def table_value(self, bitness: int, case_id: int, input_bits: str) -> np.ndarray:
        return self._value(_Command.TABLE_VALUE, bitness, case_id, input_bits)

    def table_values(
        self,
        bitness: int,
        case_id: int,
        input_bits: Sequence[str],
    ) -> np.ndarray:
        return self._values(_Command.TABLE_VALUE, bitness, case_id, input_bits)

    def _value(
        self,
        command: int,
        bitness: int,
        case_id: int,
        input_bits: str,
    ) -> np.ndarray:
        response = self._request(
            command,
            struct.pack("<HQ", bitness, case_id) + _string(input_bits),
        )
        return _ascii_bits_to_signed(response, sample_point_dim(bitness))

    def _values(
        self,
        command: int,
        bitness: int,
        case_id: int,
        input_bits: Sequence[str],
    ) -> np.ndarray:
        samples = np.empty(
            (len(input_bits), sample_point_dim(bitness)),
            dtype=np.float32,
        )
        for row_id, bits in enumerate(input_bits):
            samples[row_id] = self._value(command, bitness, case_id, bits)
        return samples

    def circuit_sets(self) -> list[str]:
        return _split_newlines(self._request(_Command.CIRCUIT_SETS))

    def circuit_cases(self, set_name: str) -> list[str]:
        return _split_newlines(
            self._request(_Command.CIRCUIT_CASES, _string(set_name)),
        )

    def circuit_inputs(self, set_name: str, case_name: str) -> int:
        response = self._request(
            _Command.CIRCUIT_INPUTS,
            _string(set_name) + _string(case_name),
        )
        return _U64.unpack(response)[0]

    def circuit_outputs(self, set_name: str, case_name: str) -> int:
        response = self._request(
            _Command.CIRCUIT_OUTPUTS,
            _string(set_name) + _string(case_name),
        )
        return _U64.unpack(response)[0]

    def circuit_value(
        self,
        set_name: str,
        case_name: str,
        input_state: str,
    ) -> np.ndarray:
        response = self._request(
            _Command.CIRCUIT_VALUE,
            _string(set_name) + _string(case_name) + _string(input_state),
        )
        point_dim = circuit_sample_point_dim(
            self.circuit_inputs(set_name, case_name),
            self.circuit_outputs(set_name, case_name),
        )
        return _ascii_bits_to_signed(response, point_dim)


def load_generator(workers: int = 1) -> Generator:
    return Generator(SERVER, workers)


def _ascii_bits_to_signed(value: bytes, expected_len: int) -> np.ndarray:
    assert len(value) == expected_len
    bits = np.frombuffer(value, dtype=np.uint8).astype(np.int8) - ord("0")
    return bits * 2 - 1


def _split_newlines(value: bytes) -> list[str]:
    text = value.decode("ascii")
    return [] if not text else text.split("\n")
