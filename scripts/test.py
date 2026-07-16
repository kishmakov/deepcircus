import ctypes
from pathlib import Path


LIBRARY = Path(__file__).resolve().parents[1] / "cpp" / "build" / "libgen.so"
CIRCUITS = Path(__file__).resolve().parents[1] / "data" / "circuits"

# Fixed seed the pinned golden values below were generated with; case structure
# is a function of (bitness, case_id, seed).
SEED = 0xC0FFEE

TABLE_SOLVABLE_CASES = [
    (4, 0, "0101", "010100000", 0, 0),
    (4, 3190, "0001", "000100100", 4, 7),
    (4, 11304, "1101", "110111001", 4, 6),
    (5, 3261348405, "11000", "11000010010", 5, 14),
    (5, 390455940, "01001", "01001101111", 5, 16),
    (6, 2547012052, "110111", "1101110000000", 6, 16),
    (6, 883941716, "011111", "0111110000000", 6, 17),
    (7, 42, "0101010", "010101001001110", 7, 59),
    (7, 239, "1010101", "101010100000000", 7, 53),
    (
        11,
        23901,
        "01010101010",
        "01010101010111110010100",
        11,
        859,
    ),
]

TABLE_BIG_CASES = [
    (
        16,
        42,
        "0101010101010101",
        "010101010101010111011010100110110",
    ),
    (
        16,
        239566,
        "1010101010101010",
        "101010101010101011001110011111001",
    ),
    (
        17,
        42,
        "01010101010101010",
        "01010101010101010000110111110000000",
    ),
    (
        24,
        188,
        "110010100111000101010011",
        "1100101001110001010100111111000101111101101011001",
    ),
    (
        32,
        320,
        "01010101010101010101010101010101",
        "01010101010101010101010101010101011010100011011000111001011101111",
    ),
    (
        48,
        480,
        "110010101100101011001010110010101100101011001010",
        "1100101011001010110010101100101011001010110010100100101001000110111001110111110010011011100000100",
    ),
    (
        64,
        640,
        "0011010100110101001101010011010100110101001101010011010100110101",
        "001101010011010100110101001101010011010100110101001101010011010111100001010010001001000001010110110010001000111100101010111011000",
    ),
    (
        100,
        1000,
        "0100110011010011001101001100110100110011010011001101001100110100110011010011001101001100110100110011",
        "010011001101001100110100110011010011001101001100110100110011010011001101001100110100110011010011001101001110001110001100101101000010110001011001001000100001100111011110100101100111100001110011010001010",
    ),
    (
        128,
        1280,
        "01101001100101100110100110010110011010011001011001101001100101100110100110010110011010011001011001101001100101100110100110010110",
        "01101001100101100110100110010110011010011001011001101001100101100110100110010110011010011001011001101001100101100110100110010110101010111101000100100101010010001100110111111111110011101100111011100001011100100000100110010001010110111001001000101111001101101",
    ),
]

def load_library():
    library = ctypes.CDLL(str(LIBRARY))

    library.gen_tree_cases_number.argtypes = [ctypes.c_uint16]
    library.gen_tree_cases_number.restype = ctypes.c_size_t

    library.gen_table_solvable_bitness.argtypes = []
    library.gen_table_solvable_bitness.restype = ctypes.c_uint16

    library.gen_table_value.argtypes = [
        ctypes.c_uint16,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_char_p,
    ]
    library.gen_table_value.restype = ctypes.c_char_p

    library.gen_circuit_sets.argtypes = []
    library.gen_circuit_sets.restype = ctypes.c_char_p

    library.gen_circuit_cases.argtypes = [ctypes.c_char_p]
    library.gen_circuit_cases.restype = ctypes.c_char_p

    library.gen_circuit_inputs.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.gen_circuit_inputs.restype = ctypes.c_size_t

    library.gen_circuit_outputs.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.gen_circuit_outputs.restype = ctypes.c_size_t

    library.gen_circuit_value.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    library.gen_circuit_value.restype = ctypes.c_char_p

    return library


def split_list(value):
    text = value.decode("ascii")
    return [] if not text else text.split("\n")


def expected_circuit_sets():
    return sorted(
        path.name
        for path in CIRCUITS.iterdir()
        if path.is_dir() and any(path.glob("*.aig"))
    )


def expected_circuit_cases(set_name):
    return sorted(path.stem for path in (CIRCUITS / set_name).glob("*.aig"))


def read_aig_metadata(set_name, case_name):
    path = CIRCUITS / set_name / f"{case_name}.aig"
    with path.open("rb") as file:
        header = file.readline().decode("ascii").split()

    assert header[0] == "aig"
    assert len(header) >= 6

    inputs = int(header[2])
    latches = int(header[3])
    outputs = int(header[4])
    bads = int(header[6]) if len(header) > 6 else 0
    return inputs + latches, outputs + bads


def circuit_value(library, set_name, case_name, input_state):
    return library.gen_circuit_value(
        set_name.encode("ascii"),
        case_name.encode("ascii"),
        input_state.encode("ascii"),
    ).decode("ascii")


def table_value(library, bitness, case_id, seed, input_bits):
    return library.gen_table_value(
        bitness,
        case_id,
        seed,
        input_bits.encode("ascii"),
    ).decode("ascii")


def bit_to_signed_float(bit):
    assert bit == "0" or bit == "1", bit
    return 1.0 if bit == "1" else -1.0


def assert_block_inputs_consistent(generated_inputs, bitness, tensor_name):
    reps = len(generated_inputs)
    block_reps = reps // 2
    blocks = (block_reps - 1).bit_length()
    base = generated_inputs[0]
    start = 0
    bit_blocks = []
    for block_id in range(blocks):
        size = bitness // blocks + (1 if block_id < bitness % blocks else 0)
        bit_blocks.append(range(start, start + size))
        start += size
    for rep in range(block_reps):
        expected_bits = list(base)
        for block_id, bit_ids in enumerate(bit_blocks):
            if ((rep >> block_id) & 1) == 0:
                continue
            for bit_id in bit_ids:
                expected_bits[bit_id] = "0" if expected_bits[bit_id] == "1" else "1"
        assert generated_inputs[rep] == "".join(expected_bits), (
            tensor_name,
            generated_inputs,
        )


def assert_case_consistent(library, value_func, value_name, bitness, case_id, seed, input_bits):
    value = value_func(library, bitness, case_id, seed, input_bits)
    assert len(value) == 2 * bitness + 1, (
        f"{value_name}({bitness}, {case_id}, {input_bits}) length: "
        f"actual={len(value)}, expected={2 * bitness + 1}, value={value}"
    )
    assert value[:bitness] == input_bits, (
        f"{value_name}({bitness}, {case_id}, {input_bits}) input prefix: "
        f"actual={value[:bitness]}, expected={input_bits}, value={value}"
    )

    repeat_value = value_func(library, bitness, case_id, seed, input_bits)
    assert value == repeat_value, (
        f"{value_name}({bitness}, {case_id}, {input_bits}) is not stable: "
        f"first={value}, second={repeat_value}"
    )

    for bit_id in range(bitness):
        flipped = list(input_bits)
        flipped[bit_id] = "1" if flipped[bit_id] == "0" else "0"
        flipped_value = value_func(library, bitness, case_id, seed, "".join(flipped))
        assert value[bitness + 1 + bit_id] == flipped_value[bitness], (
            f"{value_name}({bitness}, {case_id}, {input_bits}) "
            f"flip bit {bit_id}: sampled={value[bitness + 1 + bit_id]}, "
            f"direct={flipped_value[bitness]}, flipped_input={''.join(flipped)}"
        )

    return value


def test_table_solvable_cases(library):
    print(f"Check solvable table cases ...")

    for (
        bitness,
        case_id,
        input_bits,
        expected_value,
        expected_depth,
        expected_nodes,
    ) in TABLE_SOLVABLE_CASES:
        value = assert_case_consistent(
            library,
            table_value,
            "gen_table_value",
            bitness,
            case_id,
            SEED,
            input_bits,
        )
        assert value == expected_value, (
            f"gen_table_value({bitness}, {case_id}, {input_bits}): "
            f"actual={value}, expected={expected_value}"
        )

def test_table_big_cases(library):
    print(f"Check big table cases ...")

    for bitness, case_id, input_bits, expected_value in TABLE_BIG_CASES:
        assert bitness > library.gen_table_solvable_bitness(), bitness
        value = assert_case_consistent(
            library,
            table_value,
            "gen_table_value",
            bitness,
            case_id,
            SEED,
            input_bits,
        )
        assert len(value) == 2 * bitness + 1, (bitness, case_id, value)


def test_circuit_discovery(library):
    print(f"Check gen_circuit discovery ...")

    sets = split_list(library.gen_circuit_sets())
    assert sets == expected_circuit_sets()

    for set_name in sets:
        assert split_list(library.gen_circuit_cases(set_name.encode("ascii"))) == (
            expected_circuit_cases(set_name)
        )


def test_circuit_metadata(library):
    print(f"Check gen_circuit metadata ...")

    for set_name in expected_circuit_sets():
        for case_name in expected_circuit_cases(set_name):
            expected_inputs, expected_outputs = read_aig_metadata(set_name, case_name)
            assert library.gen_circuit_inputs(
                set_name.encode("ascii"),
                case_name.encode("ascii"),
            ) == expected_inputs
            assert library.gen_circuit_outputs(
                set_name.encode("ascii"),
                case_name.encode("ascii"),
            ) == expected_outputs

    assert library.gen_circuit_inputs(b"iscas85", b"c17") == 5
    assert library.gen_circuit_outputs(b"iscas85", b"c17") == 2

    assert library.gen_circuit_inputs(b"iscas87", b"s27") == 7
    assert library.gen_circuit_outputs(b"iscas87", b"s27") == 1


def test_circuit_value(library):
    print(f"Check gen_circuit_value ...")

    c17 = circuit_value(library, "iscas85", "c17", "00000")
    assert len(c17) == 5 + 2 * 6
    assert c17 == circuit_value(library, "iscas85", "c17", "00000")

    s27 = circuit_value(library, "iscas87", "s27", "0000000")
    assert len(s27) == 7 + 1 * 8
    assert s27 == circuit_value(library, "iscas87", "s27", "0000000")


if __name__ == "__main__":
    library = load_library()
    test_table_solvable_cases(library)
    test_table_big_cases(library)
    test_circuit_discovery(library)
    test_circuit_metadata(library)
    test_circuit_value(library)
