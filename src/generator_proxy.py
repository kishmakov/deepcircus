from __future__ import annotations

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


def _init_worker() -> None:
    global _WORKER_GENERATOR
    _WORKER_GENERATOR = load_generator()
    _ = _WORKER_GENERATOR.library


def _worker(task):
    worker_id, processes, op, bitness, indexed_payloads = task
    assert _WORKER_GENERATOR is not None
    results = []
    for row_id, case_id, payload in indexed_payloads:
        assert case_id % processes == worker_id
        if op == "values":
            samples = _WORKER_GENERATOR.case_values(bitness, case_id, payload)
        elif op == "depths":
            samples = np.float32(_WORKER_GENERATOR.case_depth(bitness, case_id))
        elif op == "nodes":
            samples = np.float32(_WORKER_GENERATOR.case_nodes(bitness, case_id))
        elif op == "restrictions":
            samples = _sample_restrictions(_WORKER_GENERATOR, bitness, case_id, payload)
        else:
            assert False, op
        results.append((row_id, samples))
    return results


def _sample_restrictions(
        generator: Generator,
        bitness: int,
        case_id: int,
        reps: int,
) -> np.ndarray:
    point_dim = restriction_point_dim(bitness)
    samples = np.empty((bitness * 2, reps, point_dim), dtype=np.float32)
    for rep in range(reps):
        samples[:, rep, :] = generator.case_restrictions(bitness, case_id, rep)
    return samples


class GeneratorProxy:
    def __init__(self, processes: int):
        assert processes > 0, processes
        self.processes = processes
        self._generator = load_generator()
        self._context = mp.get_context("fork")
        self._pools = [
            self._context.Pool(processes=1, initializer=_init_worker)
            for _ in range(processes)
        ]

    def close(self) -> None:
        for pool in self._pools:
            pool.terminate()
        for pool in self._pools:
            pool.join()
        self._pools = []

    def __enter__(self) -> GeneratorProxy:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def cases_number(self, bitness: int) -> int:
        return self._generator.cases_number(bitness)

    def case_nodes(self, bitness: int, case_id: int) -> int:
        return self._generator.case_nodes(bitness, case_id)

    def case_depth(self, bitness: int, case_id: int) -> int:
        return self._generator.case_depth(bitness, case_id)

    def case_values(
            self,
            bitness: int,
            case_id: int,
            input_bits: Sequence[str],
    ) -> np.ndarray:
        return self._generator.case_values(bitness, case_id, input_bits)

    def generate_value_tensors(
            self,
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
        results = self._dispatch("values", bitness, case_ids, input_bits)
        for row_id, samples in tqdm(
            results,
            total=len(case_ids),
            desc=f"values b={bitness}",
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
        results = self._dispatch("depths", bitness, case_ids, [None] * len(case_ids))
        for row_id, depth in tqdm(
            results,
            total=len(case_ids),
            desc=f"depths b={bitness}",
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
        results = self._dispatch("nodes", bitness, case_ids, [None] * len(case_ids))
        for row_id, nodes in tqdm(
            results,
            total=len(case_ids),
            desc=f"nodes b={bitness}",
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

    def _dispatch(
            self,
            op: str,
            bitness: int,
            case_ids: list[int],
            payloads: Sequence,
    ) -> Iterator[tuple[int, np.ndarray]]:
        assert len(case_ids) == len(payloads)
        buckets = [[] for _ in range(self.processes)]
        for row_id, (case_id, payload) in enumerate(zip(case_ids, payloads)):
            buckets[case_id % self.processes].append((row_id, case_id, payload))

        pending = []
        for worker_id, indexed_payloads in enumerate(buckets):
            if indexed_payloads:
                task = (worker_id, self.processes, op, bitness, indexed_payloads)
                pending.append(self._pools[worker_id].apply_async(_worker, (task,)))

        for async_result in pending:
            yield from async_result.get()
