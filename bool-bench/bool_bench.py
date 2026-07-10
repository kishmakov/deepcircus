from __future__ import annotations

import ctypes
import os
import weakref
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np


_LIBRARY_NAME = "libbb.so"


def _find_library() -> Path:
    override = os.environ.get("BOOL_BENCH_LIBRARY")
    if override:
        return Path(override)
    here = Path(__file__).resolve().parent
    for base in (here, *here.parents):
        for build_dir in ("build", "cmake-build-debug"):
            candidate = base / build_dir / _LIBRARY_NAME
            if candidate.exists():
                return candidate
    return here / "build" / _LIBRARY_NAME


LIBRARY = _find_library()


def sample_point_dim(bitness: int) -> int:
    return 2 * bitness + 1


def restriction_point_dim(bitness: int) -> int:
    return sample_point_dim(bitness - 1)


def circuit_sample_point_dim(inputs: int, outputs: int) -> int:
    return inputs + outputs * (inputs + 1)


def _wrap_owned_float_buffer(
    library: ctypes.CDLL,
    pointer,
    shape: tuple[int, ...],
) -> np.ndarray:
    assert pointer
    array = np.ctypeslib.as_array(pointer, shape=shape)
    weakref.finalize(array, library.bb_float_buffer_destroy, pointer)
    return array


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
        library = self._generator.library
        library.bb_tensor_acquire(self._handle)
        pointer = library.bb_tensor_take_values(self._handle)
        shape = (
            self.cases,
            2 * self.bitness,
            self.reps,
            restriction_point_dim(self.bitness),
        )
        self._values = _wrap_owned_float_buffer(library, pointer, shape)
        library.bb_tensor_release(self._generator.context, self._handle)
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
        library = self._generator.library
        library.bb_data_acquire(self._handle)
        self.bitness = int(library.bb_data_bitness(self._handle))
        self.cases = int(library.bb_data_cases(self._handle))
        self.reps = int(library.bb_data_reps(self._handle))

        values_pointer = library.bb_data_take_values(self._handle)
        assert values_pointer
        self.values = _wrap_owned_float_buffer(
            library,
            values_pointer,
            (self.cases, self.reps, sample_point_dim(self.bitness)),
        )

        targets_pointer = library.bb_data_take_targets(self._handle)
        if targets_pointer:
            self.targets = _wrap_owned_float_buffer(
                library,
                targets_pointer,
                (self.cases,),
            )
        else:
            assert self.recursive_table
            self.targets = None
        self._acquired = True

        if not self.recursive_table:
            library.bb_data_release(self._generator.context, self._handle)
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
        assert first_case + cases <= self.cases, (
            first_case,
            cases,
            self.cases,
        )
        handle = self._generator.library.bb_table_restrictions_tensor(
            self._generator.context,
            self._handle,
            first_case,
            cases,
        )
        assert handle
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
            self._generator.library.bb_data_release(
                self._generator.context,
                self._handle,
            )
            self._generator._forget(self)
            self._handle = None
        self._acquired = False

    def _invalidate(self) -> None:
        self._handle = None


class Generator:
    def __init__(self, library_path: Path, workers: int = 1):
        assert workers > 0, workers
        self.library_path = str(library_path)
        self.workers = workers
        self._library: ctypes.CDLL | None = None
        self._context: int | None = None
        self._owned: set[GeneratedData | GeneratedTensor] = set()

    @property
    def library(self) -> ctypes.CDLL:
        if self._library is None:
            self._library = self._load_library()
        return self._library

    @property
    def context(self) -> int:
        if self._context is None:
            context = self.library.bb_generator_create(self.workers)
            assert context
            self._context = context
        return self._context

    def close(self) -> None:
        if self._context is None:
            return
        self.library.bb_generator_destroy(self._context)
        self._context = None
        for owner in list(self._owned):
            owner._invalidate()
        self._owned.clear()

    def _remember(self, owner: GeneratedData | GeneratedTensor) -> None:
        assert owner not in self._owned
        self._owned.add(owner)

    def _forget(self, owner: GeneratedData | GeneratedTensor) -> None:
        assert owner in self._owned
        self._owned.remove(owner)

    def _load_library(self) -> ctypes.CDLL:
        library = ctypes.CDLL(self.library_path)
        float_pointer = ctypes.POINTER(ctypes.c_float)

        library.bb_generator_create.argtypes = [ctypes.c_size_t]
        library.bb_generator_create.restype = ctypes.c_void_p
        library.bb_generator_destroy.argtypes = [ctypes.c_void_p]
        library.bb_generator_destroy.restype = None

        library.bb_data_acquire.argtypes = [ctypes.c_void_p]
        library.bb_data_acquire.restype = None
        library.bb_data_bitness.argtypes = [ctypes.c_void_p]
        library.bb_data_bitness.restype = ctypes.c_uint16
        library.bb_data_cases.argtypes = [ctypes.c_void_p]
        library.bb_data_cases.restype = ctypes.c_size_t
        library.bb_data_reps.argtypes = [ctypes.c_void_p]
        library.bb_data_reps.restype = ctypes.c_size_t
        library.bb_data_take_values.argtypes = [ctypes.c_void_p]
        library.bb_data_take_values.restype = float_pointer
        library.bb_data_take_targets.argtypes = [ctypes.c_void_p]
        library.bb_data_take_targets.restype = float_pointer
        library.bb_data_release.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        library.bb_data_release.restype = None

        library.bb_tensor_acquire.argtypes = [ctypes.c_void_p]
        library.bb_tensor_acquire.restype = None
        library.bb_tensor_take_values.argtypes = [ctypes.c_void_p]
        library.bb_tensor_take_values.restype = float_pointer
        library.bb_tensor_release.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        library.bb_tensor_release.restype = None
        library.bb_float_buffer_destroy.argtypes = [float_pointer]
        library.bb_float_buffer_destroy.restype = None

        library.bb_tree_cases_number.argtypes = [ctypes.c_uint16]
        library.bb_tree_cases_number.restype = ctypes.c_size_t
        library.bb_table_cases_number.argtypes = [ctypes.c_uint16]
        library.bb_table_cases_number.restype = ctypes.c_size_t
        library.bb_table_solvable_bitness.argtypes = []
        library.bb_table_solvable_bitness.restype = ctypes.c_uint16
        library.bb_min_tree_bitness.argtypes = []
        library.bb_min_tree_bitness.restype = ctypes.c_uint16

        library.bb_tree_value.argtypes = [
            ctypes.c_uint16,
            ctypes.c_size_t,
            ctypes.c_char_p,
        ]
        library.bb_tree_value.restype = ctypes.c_char_p
        library.bb_tree_value_tensor.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint64,
        ]
        library.bb_tree_value_tensor.restype = ctypes.c_void_p

        library.bb_table_value.argtypes = [
            ctypes.c_uint16,
            ctypes.c_size_t,
            ctypes.c_char_p,
        ]
        library.bb_table_value.restype = ctypes.c_char_p
        library.bb_table_value_tensor.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint16,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint64,
        ]
        library.bb_table_value_tensor.restype = ctypes.c_void_p
        library.bb_table_restrictions_tensor.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        library.bb_table_restrictions_tensor.restype = ctypes.c_void_p

        library.bb_circuit_sets.argtypes = []
        library.bb_circuit_sets.restype = ctypes.c_char_p
        library.bb_circuit_cases.argtypes = [ctypes.c_char_p]
        library.bb_circuit_cases.restype = ctypes.c_char_p
        library.bb_circuit_inputs.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        library.bb_circuit_inputs.restype = ctypes.c_size_t
        library.bb_circuit_outputs.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        library.bb_circuit_outputs.restype = ctypes.c_size_t
        library.bb_circuit_value.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        library.bb_circuit_value.restype = ctypes.c_char_p
        return library

    def tree_cases_number(self, bitness: int) -> int:
        return int(self.library.bb_tree_cases_number(bitness))

    def table_cases_number(self, bitness: int) -> int:
        return int(self.library.bb_table_cases_number(bitness))

    def min_tree_bitness(self) -> int:
        return int(self.library.bb_min_tree_bitness())

    def table_solvable_bitness(self) -> int:
        return int(self.library.bb_table_solvable_bitness())

    def tree_value_tensor(
        self,
        bitness: int,
        cases: int,
        reps: int,
        seed: int,
    ) -> GeneratedData:
        handle = self.library.bb_tree_value_tensor(
            self.context,
            bitness,
            cases,
            reps,
            seed,
        )
        assert handle
        data = GeneratedData(self, handle, recursive_table=False)
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
        handle = self.library.bb_table_value_tensor(
            self.context,
            bitness,
            cases,
            reps,
            restriction_chunk_cases,
            seed,
        )
        assert handle
        data = GeneratedData(self, handle, recursive_table=recursive)
        self._remember(data)
        return data

    def tree_value(self, bitness: int, case_id: int, input_bits: str) -> np.ndarray:
        return self._value(self.library.bb_tree_value, bitness, case_id, input_bits)

    def tree_values(
        self,
        bitness: int,
        case_id: int,
        input_bits: Sequence[str],
    ) -> np.ndarray:
        return self._values(self.library.bb_tree_value, bitness, case_id, input_bits)

    def table_value(self, bitness: int, case_id: int, input_bits: str) -> np.ndarray:
        return self._value(self.library.bb_table_value, bitness, case_id, input_bits)

    def table_values(
        self,
        bitness: int,
        case_id: int,
        input_bits: Sequence[str],
    ) -> np.ndarray:
        return self._values(self.library.bb_table_value, bitness, case_id, input_bits)

    def _value(
        self,
        value_fn: Callable[[int, int, bytes], bytes],
        bitness: int,
        case_id: int,
        input_bits: str,
    ) -> np.ndarray:
        value = value_fn(bitness, case_id, input_bits.encode("ascii"))
        return _ascii_bits_to_signed(value, sample_point_dim(bitness))

    def _values(
        self,
        value_fn: Callable[[int, int, bytes], bytes],
        bitness: int,
        case_id: int,
        input_bits: Sequence[str],
    ) -> np.ndarray:
        samples = np.empty(
            (len(input_bits), sample_point_dim(bitness)),
            dtype=np.float32,
        )
        for row_id, bits in enumerate(input_bits):
            samples[row_id] = self._value(value_fn, bitness, case_id, bits)
        return samples

    def circuit_sets(self) -> list[str]:
        return _split_newlines(self.library.bb_circuit_sets())

    def circuit_cases(self, set_name: str) -> list[str]:
        return _split_newlines(self.library.bb_circuit_cases(set_name.encode("ascii")))

    def circuit_inputs(self, set_name: str, case_name: str) -> int:
        return int(self.library.bb_circuit_inputs(
            set_name.encode("ascii"),
            case_name.encode("ascii"),
        ))

    def circuit_outputs(self, set_name: str, case_name: str) -> int:
        return int(self.library.bb_circuit_outputs(
            set_name.encode("ascii"),
            case_name.encode("ascii"),
        ))

    def circuit_value(self, set_name: str, case_name: str, input_state: str) -> np.ndarray:
        value = self.library.bb_circuit_value(
            set_name.encode("ascii"),
            case_name.encode("ascii"),
            input_state.encode("ascii"),
        )
        point_dim = circuit_sample_point_dim(
            self.circuit_inputs(set_name, case_name),
            self.circuit_outputs(set_name, case_name),
        )
        return _ascii_bits_to_signed(value, point_dim)


def load_generator(workers: int = 1) -> Generator:
    return Generator(LIBRARY, workers)


def _ascii_bits_to_signed(value: bytes, expected_len: int) -> np.ndarray:
    assert len(value) == expected_len
    bits = np.frombuffer(value, dtype=np.uint8).astype(np.int8) - ord("0")
    return bits * 2 - 1


def _split_newlines(value: bytes) -> list[str]:
    text = value.decode("ascii")
    return [] if not text else text.split("\n")
