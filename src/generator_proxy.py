from __future__ import annotations

import atexit
import multiprocessing as mp
from collections.abc import Iterator, Sequence
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm


BOOL_BENCH_DIR = Path(__file__).resolve().parents[1] / "bool-bench"
if str(BOOL_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BOOL_BENCH_DIR))

from bool_bench import (  # noqa: E402
    Generator,
    load_generator,
    restriction_point_dim,
    sample_point_dim,
)


_WORKER_GENERATOR: Generator | None = None
_FLEET = None
_FLEET_KEY = None


def _route(bitness: int, case_id: int, processes: int) -> int:
    """Map a ``(bitness, case_id)`` pair to a fixed worker.

    Routing is deterministic so each pair is only ever processed by a single
    worker, keeping its libbb.so cache free of cross-process duplication.
    """
    return (bitness * 1_000_003 + case_id) % processes


def _init_worker(library_path: str) -> None:
    global _WORKER_GENERATOR
    _WORKER_GENERATOR = Generator(Path(library_path))
    _ = _WORKER_GENERATOR.library


def _get_fleet(generator: Generator, processes: int) -> list:
    # A persistent fleet of single-worker pools, one per route bucket. A given
    # (bitness, case_id) pair always runs on the same worker, keeping libbb.so
    # and its C++ tree caches warm for the whole run.
    global _FLEET, _FLEET_KEY
    key = (generator.library_path, processes)
    if _FLEET_KEY != key:
        close_fleet()
        context = mp.get_context("fork")
        _FLEET = [
            context.Pool(
                processes=1,
                initializer=_init_worker,
                initargs=(generator.library_path,),
            )
            for _ in range(processes)
        ]
        _FLEET_KEY = key
    return _FLEET


def close_fleet() -> None:
    global _FLEET, _FLEET_KEY
    if _FLEET is not None:
        for pool in _FLEET:
            pool.close()
        for pool in _FLEET:
            pool.join()
    _FLEET = None
    _FLEET_KEY = None


atexit.register(close_fleet)


def _worker(task):
    worker_id, processes, op, bitness, shared_payload, indexed_cases = task
    assert _WORKER_GENERATOR is not None
    if op == "tree_values" or op == "table_values":
        assert shared_payload is not None
        row_ids = [row_id for row_id, _ in indexed_cases]
        case_ids = [case_id for _, case_id in indexed_cases]
        reps, seed = shared_payload
        assert all(
            _route(bitness, case_id, processes) == worker_id
            for case_id in case_ids
        )
        if op == "tree_values":
            samples = _WORKER_GENERATOR.tree_value_tensor(
                bitness,
                case_ids,
                reps,
                seed,
            )
        else:
            samples = _WORKER_GENERATOR.table_value_tensor(
                bitness,
                case_ids,
                reps,
                seed,
            )
        return [
            (row_id, samples[result_id])
            for result_id, row_id in enumerate(row_ids)
        ]

    if op == "tree_restrictions" or op == "table_restrictions":
        assert shared_payload is not None
        row_ids = [row_id for row_id, _ in indexed_cases]
        case_ids = [case_id for _, case_id in indexed_cases]
        reps, seed = shared_payload
        assert all(
            _route(bitness, case_id, processes) == worker_id
            for case_id in case_ids
        )
        if op == "tree_restrictions":
            samples = _WORKER_GENERATOR.tree_restrictions_tensor(
                bitness,
                case_ids,
                reps,
                seed,
            )
        else:
            samples = _WORKER_GENERATOR.table_restrictions_tensor(
                bitness,
                case_ids,
                reps,
                seed,
            )
        return [
            (row_id, samples[result_id])
            for result_id, row_id in enumerate(row_ids)
        ]

    assert shared_payload is None
    results = []
    for row_id, case_id in indexed_cases:
        assert _route(bitness, case_id, processes) == worker_id
        if op == "tree_depths":
            samples = np.float32(_WORKER_GENERATOR.tree_depth(bitness, case_id))
        elif op == "table_depths":
            samples = np.float32(_WORKER_GENERATOR.table_depth(bitness, case_id))
        elif op == "tree_nodes":
            samples = np.float32(_WORKER_GENERATOR.tree_nodes(bitness, case_id))
        elif op == "table_nodes":
            samples = np.float32(_WORKER_GENERATOR.table_nodes(bitness, case_id))
        else:
            assert False, op
        results.append((row_id, samples))
    return results

class GeneratorProxy:
    def __init__(self, processes: int):
        assert processes > 0, processes
        self.processes = processes
        self._generator = load_generator()
        self._pools = _get_fleet(self._generator, processes)
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        close_fleet()
        self._pools = []
        self._closed = True

    def __enter__(self) -> GeneratorProxy:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def tree_cases_number(self, bitness: int) -> int:
        return self._generator.tree_cases_number(bitness)

    def table_cases_number(self, bitness: int) -> int:
        return self._generator.table_cases_number(bitness)

    def solvable_bitness(self) -> int:
        return self._generator.table_solvable_bitness()

    def min_tree_bitness(self) -> int:
        return self._generator.min_tree_bitness()

    def tree_nodes(self, bitness: int, case_id: int) -> int:
        return self._generator.tree_nodes(bitness, case_id)

    def tree_depth(self, bitness: int, case_id: int) -> int:
        return self._generator.tree_depth(bitness, case_id)

    def tree_values(
            self,
            bitness: int,
            case_id: int,
            input_bits: Sequence[str],
    ) -> np.ndarray:
        return self._generator.tree_values(bitness, case_id, input_bits)

    def tree_value_tensors(
            self,
            hint: str,
            bitness: int,
            case_ids: list[int],
            reps: int,
            seed: int,
    ) -> np.ndarray:
        return self._value_tensors(
            hint,
            "tree_values",
            bitness,
            case_ids,
            reps,
            seed,
        )

    def table_value_tensors(
            self,
            hint: str,
            bitness: int,
            case_ids: list[int],
            reps: int,
            seed: int,
    ) -> np.ndarray:
        return self._value_tensors(
            hint,
            "table_values",
            bitness,
            case_ids,
            reps,
            seed,
        )

    # Result shape: cases x reps x (2 * bitness + 1).
    def _value_tensors(
            self,
            hint: str,
            type: str,
            bitness: int,
            case_ids: list[int],
            reps: int,
            seed: int,
    ) -> np.ndarray:
        case_ids = list(case_ids)
        assert case_ids, "empty cases"
        assert type in ("tree_values", "table_values"), type
        assert reps > 0, reps
        assert reps % 2 == 0, reps

        x = np.empty(
            (len(case_ids), reps, sample_point_dim(bitness)),
            dtype=np.float32,
        )
        results = self._dispatch(
            type,
            bitness,
            case_ids,
            shared_payload=(reps, seed),
        )
        for row_id, samples in tqdm(
            results,
            total=len(case_ids),
            desc=f"{hint}:{type} b={bitness}",
        ):
            x[row_id] = samples
        return x

    def tree_depth_tensors(
            self,
            hint: str,
            bitness: int,
            case_ids: list[int],
    ) -> np.ndarray:
        return self._depth_tensors(hint, "tree_depths", bitness, case_ids)

    def table_depth_tensors(
            self,
            hint: str,
            bitness: int,
            case_ids: list[int],
    ) -> np.ndarray:
        assert bitness <= self.solvable_bitness(), bitness
        return self._depth_tensors(hint, "table_depths", bitness, case_ids)

    def _depth_tensors(
            self,
            hint: str,
            op: str,
            bitness: int,
            case_ids: list[int],
    ) -> np.ndarray:
        case_ids = list(case_ids)
        y = np.empty(len(case_ids), dtype=np.float32)
        results = self._dispatch(op, bitness, case_ids)
        for row_id, depth in tqdm(
            results,
            total=len(case_ids),
            desc=f"{hint}:{op} b={bitness}",
        ):
            y[row_id] = depth
        return y

    def restrictions_tensors(
            self,
            type: str,
            bitness: int,
            case_ids: list[int],
            reps: int,
            seed: int,
            progress,
    ) -> np.ndarray:
        assert type in ("tree_restrictions", "table_restrictions"), type
        case_ids = list(case_ids)
        point_dim = restriction_point_dim(bitness)
        restrictions_per_case = bitness * 2
        assert case_ids, "empty cases"
        assert reps > 0, reps
        assert reps % 2 == 0, reps
        x = np.empty(
            (len(case_ids) * restrictions_per_case, reps, point_dim),
            dtype=np.float32,
        )
        results = self._dispatch(
            type,
            bitness,
            case_ids,
            shared_payload=(reps, seed),
        )
        for row_id, samples in results:
            start = row_id * restrictions_per_case
            x[start : start + restrictions_per_case] = samples
            progress.update(1)
        return x

    def _dispatch(
            self,
            op: str,
            bitness: int,
            case_ids: list[int],
            shared_payload: tuple[int, int] | None = None,
    ) -> Iterator[tuple[int, np.ndarray]]:
        assert not self._closed
        buckets = [[] for _ in range(self.processes)]
        for row_id, case_id in enumerate(case_ids):
            worker_id = _route(bitness, case_id, self.processes)
            buckets[worker_id].append((row_id, case_id))

        pending = []
        for worker_id, indexed_cases in enumerate(buckets):
            if indexed_cases:
                task = (
                    worker_id,
                    self.processes,
                    op,
                    bitness,
                    shared_payload,
                    indexed_cases,
                )
                pending.append(self._pools[worker_id].apply_async(_worker, (task,)))

        for async_result in pending:
            yield from async_result.get()
