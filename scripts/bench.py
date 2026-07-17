#!/usr/bin/env python3

from __future__ import annotations

import json
import socket
import struct
import subprocess
import sys
from multiprocessing import shared_memory
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HEADER = struct.Struct("<IQ")
INITIALIZATION = struct.Struct("<QQHHQQQHH")
TASK_PREFIX = struct.Struct("<QHQQ")
TENSOR_DESCRIPTOR = struct.Struct("<BHQQQQQQ")
INITIALIZE = 1
NEXT = 2
RELEASE = 3
RESTRICTIONS = 2
NO_OFFSET = (1 << 64) - 1


def receive_exact(connection: socket.socket, size: int) -> bytes:
    result = bytearray(size)
    view = memoryview(result)
    offset = 0
    while offset < size:
        count = connection.recv_into(view[offset:])
        assert count > 0, "generator server disconnected"
        offset += count
    return bytes(result)


def request(connection: socket.socket, command: int, payload: bytes = b"") -> bytes:
    connection.sendall(HEADER.pack(command, len(payload)) + payload)
    status, response_size = HEADER.unpack(receive_exact(connection, HEADER.size))
    assert status == 0, status
    return receive_exact(connection, response_size)


def take_string(payload: bytes, offset: int) -> tuple[str, int]:
    size = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    return payload[offset : offset + size].decode("ascii"), offset + size


def consume_tasks(
    connection: socket.socket,
    coordinates: list[tuple[int, int]],
) -> tuple[int, int, int]:
    task_count = 0
    tensor_count = 0
    byte_count = 0
    while True:
        payload = request(connection, NEXT)
        assert payload, payload
        if payload[0] == 1:
            assert len(payload) == 1, len(payload)
            assert task_count == len(coordinates), (task_count, coordinates)
            return task_count, tensor_count, byte_count
        assert payload[0] == 0, payload[0]

        offset = 1
        iteration, bitness, task_seed, shared_size = TASK_PREFIX.unpack_from(
            payload,
            offset,
        )
        assert task_count < len(coordinates), (task_count, coordinates)
        assert (iteration, bitness) == coordinates[task_count], (
            (iteration, bitness),
            coordinates[task_count],
        )
        offset += TASK_PREFIX.size
        name, offset = take_string(payload, offset)
        tensors = struct.unpack_from("<I", payload, offset)[0]
        offset += 4

        memory = shared_memory.SharedMemory(name=name)
        assert memory.size == shared_size, (memory.size, shared_size)
        try:
            for _ in range(tensors):
                descriptor = TENSOR_DESCRIPTOR.unpack_from(payload, offset)
                offset += TENSOR_DESCRIPTOR.size
                (
                    kind,
                    tensor_bitness,
                    cases,
                    reps,
                    values_offset,
                    value_count,
                    targets_offset,
                    target_count,
                ) = descriptor
                assert tensor_bitness == bitness, (tensor_bitness, bitness)
                assert kind in (1, RESTRICTIONS), kind
                if kind == RESTRICTIONS:
                    shape = (cases, 2 * bitness, reps, 2 * bitness - 1)
                else:
                    shape = (cases, reps, 2 * bitness + 1)
                assert int(np.prod(shape)) == value_count, (shape, value_count)
                # Tensors are published bit-packed: one padded row of bytes
                # per case, little-endian bit order.
                row_bytes = (value_count // cases + 7) // 8
                values = np.ndarray(
                    (cases, row_bytes),
                    dtype=np.uint8,
                    buffer=memory.buf,
                    offset=values_offset,
                )
                if target_count:
                    assert targets_offset != NO_OFFSET, targets_offset
                    assert target_count == 2 * cases, (target_count, cases)
                    targets = np.ndarray(
                        (cases, 2),
                        dtype=np.float32,
                        buffer=memory.buf,
                        offset=targets_offset,
                    )
                    del targets
                else:
                    assert targets_offset == NO_OFFSET, targets_offset
                del values
            assert offset == len(payload), (offset, len(payload))
        finally:
            memory.close()
            memory.unlink()

        response = request(connection, RELEASE)
        assert not response, response
        task_count += 1
        tensor_count += tensors
        byte_count += shared_size


def main() -> None:
    from src.config import load_train_config
    from src.generator import SERVER

    config = load_train_config("bench.conf")
    first_iteration = 1
    last_iteration = config.training.iterations
    total_started = perf_counter()
    process = subprocess.Popen(
        [SERVER, "--port", "0"],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None

    connection = None
    try:
        ready = process.stdout.readline().strip()
        assert ready.startswith("PORT "), ready
        port = int(ready.removeprefix("PORT "))
        startup_seconds = perf_counter() - total_started

        connection = socket.create_connection(("127.0.0.1", port))
        payload = INITIALIZATION.pack(
            first_iteration,
            last_iteration,
            config.bitness_from,
            config.bitness_to,
            config.training.seed,
            config.training.train_samples,
            config.training.validation_samples,
            config.training.sample_batches,
            config.training.sample_points_in_batch,
        )
        initialization_started = perf_counter()
        response = request(connection, INITIALIZE, payload)
        assert not response, response
        initialization_seconds = perf_counter() - initialization_started

        generation_started = perf_counter()
        # Validation tensors depend only on bitness, so the daemon emits one
        # task per bitness (iteration 0) ahead of the training tasks.
        coordinates = [(0, bitness) for bitness in config.bitness_range()]
        coordinates += [
            (iteration, bitness)
            for iteration in range(first_iteration, last_iteration + 1)
            for bitness in config.bitness_range()
        ]
        task_count, tensor_count, byte_count = consume_tasks(
            connection,
            coordinates,
        )
        generation_seconds = perf_counter() - generation_started

        print(json.dumps({
            "first_iteration": first_iteration,
            "last_iteration": last_iteration,
            "bitness_from": config.bitness_from,
            "bitness_to": config.bitness_to,
            "seed": config.training.seed,
            "train_samples": config.training.train_samples,
            "validation_samples": config.training.validation_samples,
            "sample_batches": config.training.sample_batches,
            "sample_points_in_batch": config.training.sample_points_in_batch,
            "tasks": task_count,
            "tensors": tensor_count,
            "tensor_bytes": byte_count,
            "startup_seconds": startup_seconds,
            "initialization_seconds": initialization_seconds,
            "generation_seconds": generation_seconds,
            "tasks_per_second": task_count / generation_seconds,
            "total_seconds": perf_counter() - total_started,
        }))
    finally:
        if connection is not None:
            connection.close()
            return_code = process.wait()
            assert return_code == 0, return_code
        elif process.poll() is None:
            process.terminate()
            process.wait()


if __name__ == "__main__":
    main()
