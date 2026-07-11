#!/usr/bin/env python3

from __future__ import annotations

import json
import socket
import struct
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HEADER = struct.Struct("<IQ")
INITIALIZATION = struct.Struct("<QQHHQ")
INITIALIZE = 1
SHUTDOWN = 2


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
    initialized = False
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
        )
        initialization_started = perf_counter()
        response = request(connection, INITIALIZE, payload)
        assert not response, response
        initialization_seconds = perf_counter() - initialization_started
        initialized = True

        print(json.dumps({
            "first_iteration": first_iteration,
            "last_iteration": last_iteration,
            "bitness_from": config.bitness_from,
            "bitness_to": config.bitness_to,
            "seed": config.training.seed,
            "startup_seconds": startup_seconds,
            "initialization_seconds": initialization_seconds,
            "total_seconds": perf_counter() - total_started,
        }))
    finally:
        if connection is not None:
            if initialized:
                response = request(connection, SHUTDOWN)
                assert not response, response
            connection.close()
        if initialized:
            return_code = process.wait()
            assert return_code == 0, return_code
        elif process.poll() is None:
            process.terminate()
            process.wait()


if __name__ == "__main__":
    main()
