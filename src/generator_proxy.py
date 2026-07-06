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
    worker_id, processes, op, bitness, indexed_payloads = task
    assert _WORKER_GENERATOR is not None
    results = []
    for row_id, case_id, payload in indexed_payloads:
        assert _route(bitness, case_id, processes) == worker_id
        if op == "tree_values":
            samples = _WORKER_GENERATOR.tree_values(bitness, case_id, payload)
        elif op == "table_values":
            samples = _WORKER_GENERATOR.table_values(bitness, case_id, payload)
        elif op == "tree_depths":
            samples = np.float32(_WORKER_GENERATOR.tree_depth(bitness, case_id))
        elif op == "table_depths":
            samples = np.float32(_WORKER_GENERATOR.table_depth(bitness, case_id))
        elif op == "tree_nodes":
            samples = np.float32(_WORKER_GENERATOR.tree_nodes(bitness, case_id))
        elif op == "table_nodes":
            samples = np.float32(_WORKER_GENERATOR.table_nodes(bitness, case_id))
        elif op == "restrictions" or op == "restrictions_rnd":
            samples = _sample_restrictions(
                _WORKER_GENERATOR,
                bitness,
                case_id,
                payload,
                op == "restrictions_rnd",
            )
        else:
            assert False, op
        results.append((row_id, samples))
    return results


def _sample_restrictions(
        generator: Generator,
        bitness: int,
        case_id: int,
        reps: int,
        random_values: bool,
) -> np.ndarray:
    point_dim = restriction_point_dim(bitness)
    samples = np.empty((bitness * 2, reps, point_dim), dtype=np.float32)
    for rep in range(reps):
        if random_values:
            samples[:, rep, :] = generator.table_restrictions(bitness, case_id, rep)
        elif bitness <= generator.table_solvable_bitness():
            samples[:, rep, :] = generator.table_restrictions(bitness, case_id, rep)
        else:
            samples[:, rep, :] = generator.tree_restrictions(bitness, case_id, rep)
    return samples


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
            bitness: int,
            case_ids: list[int],
            input_bits: Sequence[Sequence[str]],
    ) -> np.ndarray:
        return self._value_tensors("tree_values", bitness, case_ids, input_bits)

    def table_value_tensors(
            self,
            bitness: int,
            case_ids: list[int],
            input_bits: Sequence[Sequence[str]],
    ) -> np.ndarray:
        return self._value_tensors("table_values", bitness, case_ids, input_bits)

    # Result shape: cases x reps x (2 * bitness + 1).
    def _value_tensors(
            self,
            op: str,
            bitness: int,
            case_ids: list[int],
            input_bits: Sequence[Sequence[str]],
    ) -> np.ndarray:
        case_ids = list(case_ids)
        assert len(case_ids) == len(input_bits)
        assert input_bits, "empty input"
        reps = len(input_bits[0])
        assert all(len(case_input_bits) == reps for case_input_bits in input_bits)

        x = np.empty(
            (len(case_ids), reps, sample_point_dim(bitness)),
            dtype=np.float32,
        )
        results = self._dispatch(op, bitness, case_ids, input_bits)
        for row_id, samples in tqdm(
            results,
            total=len(case_ids),
            desc=f"{op} b={bitness}",
        ):
            x[row_id] = samples
        return x

    def generate_depths_tensors(
            self,
            bitness: int,
            case_ids: list[int],
    ) -> np.ndarray:
        case_ids = list(case_ids)
        y = np.empty(len(case_ids), dtype=np.float32)
        op = "table_depths" if bitness <= self.solvable_bitness() else "tree_depths"
        results = self._dispatch(op, bitness, case_ids, [None] * len(case_ids))
        for row_id, depth in tqdm(
            results,
            total=len(case_ids),
            desc=f"{op} b={bitness}",
        ):
            y[row_id] = depth
        return y

    def generate_node_tensors(
            self,
            bitness: int,
            case_ids: list[int],
    ) -> np.ndarray:
        case_ids = list(case_ids)
        y = np.empty(len(case_ids), dtype=np.float32)
        op = "table_nodes" if bitness <= self.solvable_bitness() else "tree_nodes"
        results = self._dispatch(op, bitness, case_ids, [None] * len(case_ids))
        for row_id, nodes in tqdm(
            results,
            total=len(case_ids),
            desc=f"{op} b={bitness}",
        ):
            y[row_id] = nodes
        return y

    def generate_restriction_tensors(
            self,
            bitness: int,
            case_ids: list[int],
            reps: int,
    ) -> np.ndarray:
        case_ids = list(case_ids)
        point_dim = restriction_point_dim(bitness)
        restrictions_per_case = bitness * 2
        x = np.empty(
            (len(case_ids) * restrictions_per_case, reps, point_dim),
            dtype=np.float32,
        )
        results = self._dispatch("restrictions", bitness, case_ids, [reps] * len(case_ids))
        for row_id, samples in tqdm(
            results,
            total=len(case_ids),
            desc=f"restrictions b={bitness}",
        ):
            start = row_id * restrictions_per_case
            x[start : start + restrictions_per_case] = samples
        return x

    def generate_restriction_tensors_rnd(
            self,
            bitness: int,
            case_ids: list[int],
            reps: int,
    ) -> np.ndarray:
        case_ids = list(case_ids)
        point_dim = restriction_point_dim(bitness)
        restrictions_per_case = bitness * 2
        x = np.empty(
            (len(case_ids) * restrictions_per_case, reps, point_dim),
            dtype=np.float32,
        )
        results = self._dispatch("restrictions_rnd", bitness, case_ids, [reps] * len(case_ids))
        for row_id, samples in tqdm(
            results,
            total=len(case_ids),
            desc=f"restrictions_rnd b={bitness}",
        ):
            start = row_id * restrictions_per_case
            x[start : start + restrictions_per_case] = samples
        return x

    def _dispatch(
            self,
            op: str,
            bitness: int,
            case_ids: list[int],
            payloads: Sequence,
    ) -> Iterator[tuple[int, np.ndarray]]:
        assert not self._closed
        assert len(case_ids) == len(payloads)
        buckets = [[] for _ in range(self.processes)]
        for row_id, (case_id, payload) in enumerate(zip(case_ids, payloads)):
            worker_id = _route(bitness, case_id, self.processes)
            buckets[worker_id].append((row_id, case_id, payload))

        pending = []
        for worker_id, indexed_payloads in enumerate(buckets):
            if indexed_payloads:
                task = (worker_id, self.processes, op, bitness, indexed_payloads)
                pending.append(self._pools[worker_id].apply_async(_worker, (task,)))

        for async_result in pending:
            yield from async_result.get()
