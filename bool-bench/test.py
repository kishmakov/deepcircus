import ctypes
from pathlib import Path


LIBRARY = Path(__file__).resolve().parent / "build" / "libbb.so"
CIRCUITS = Path(__file__).resolve().parent / "circuits"

TABLE_SOLVABLE_CASES = [
    (4, 0, "0101", "010100000", 0, 0),
    (4, 3190, "0001", "000100100", 4, 7),
    (4, 11304, "1101", "110111001", 4, 6),
    (5, 3261348405, "11000", "11000010010", 5, 14),
    (5, 390455940, "01001", "01001101111", 5, 16),
    (6, 2547012052, "110111", "1101110000000", 6, 16),
    (6, 883941716, "011111", "0111110000000", 6, 17),
    (7, 42, "0101010", "010101000010010", 7, 59),
    (7, 239, "1010101", "101010110101101", 7, 58),
    (
        11,
        23901,
        "01010101010",
        "01010101010011100010111",
        11,
        842,
    ),
    (
        16,
        42,
        "0101010101010101",
        "010101010101010100111110101000000",
        16,
        26467,
    ),
    (
        16,
        239566,
        "1010101010101010",
        "101010101010101000110101001010010",
        16,
        26543,
    ),
]

TABLE_BIG_CASES = [
    (
        17,
        42,
        "01010101010101010",
        "01010101010101010001000111101010011",
    ),
    (
        24,
        188,
        "110010100111000101010011",
        "1100101001110001010100110111011010001001101100000",
    ),
    (
        32,
        320,
        "01010101010101010101010101010101",
        "01010101010101010101010101010101001111001010100111010100100101001",
    ),
    (
        48,
        480,
        "110010101100101011001010110010101100101011001010",
        "1100101011001010110010101100101011001010110010101000111001110111110010001101100001101111101011101",
    ),
    (
        64,
        640,
        "0011010100110101001101010011010100110101001101010011010100110101",
        "001101010011010100110101001101010011010100110101001101010011010100110001110000001111101000011110110111110001011111000011001001010",
    ),
    (
        100,
        1000,
        "0100110011010011001101001100110100110011010011001101001100110100110011010011001101001100110100110011",
        "010011001101001100110100110011010011001101001100110100110011010011001101001100110100110011010011001110100010110000101001000010010000110000000010111001011011100010011010011000110011011011110010010110000",
    ),
    (
        128,
        1280,
        "01101001100101100110100110010110011010011001011001101001100101100110100110010110011010011001011001101001100101100110100110010110",
        "01101001100101100110100110010110011010011001011001101001100101100110100110010110011010011001011001101001100101100110100110010110101001111111111101001011000101100011011100110010110101111001111001110110111111011101110001111011001100000001101010110110000110111",
    ),
]

TREE_CASES = [
    (
        17,
        42,
        "01010101010101010",
        "01010101010101010010010100010000010",
        9,
        42,
    ),
    (
        24,
        188,
        "110010100111000101010011",
        "1100101001110001010100110000000000010000100001000",
        17,
        188,
    ),
    (
        32,
        320,
        "01010101010101010101010101010101",
        "01010101010101010101010101010101111110111101010111101111011111111",
        18,
        320,
    ),
    (
        48,
        480,
        "110010101100101011001010110010101100101011001010",
        "1100101011001010110010101100101011001010110010101111111011111011110111111101111111111011111111110",
        20,
        480,
    ),
    (
        64,
        640,
        "0011010100110101001101010011010100110101001101010011010100110101",
        "001101010011010100110101001101010011010100110101001101010011010111111111111111111111111111111111111111111111111111111111111111110",
        20,
        640,
    ),
    (
        100,
        1000,
        "0100110011010011001101001100110100110011010011001101001100110100110011010011001101001100110100110011",
        "010011001101001100110100110011010011001101001100110100110011010011001101001100110100110011010011001111111111111111001111111111111111111111111111111111011111001111111111111011111110111111111101111111111",
        20,
        1000,
    ),
    (
        128,
        1280,
        "01101001100101100110100110010110011010011001011001101001100101100110100110010110011010011001011001101001100101100110100110010110",
        "01101001100101100110100110010110011010011001011001101001100101100110100110010110011010011001011001101001100101100110100110010110111111111111111111111111111111101111111111111100111111111111101111111111111111111111111111111111111111111111101111111111111111110",
        21,
        1280,
    ),
]

def load_library():
    library = ctypes.CDLL(str(LIBRARY))

    library.bb_tree_cases_number.argtypes = [ctypes.c_uint16]
    library.bb_tree_cases_number.restype = ctypes.c_size_t

    library.bb_table_solvable_bitness.argtypes = []
    library.bb_table_solvable_bitness.restype = ctypes.c_uint16

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
    return library.bb_circuit_value(
        set_name.encode("ascii"),
        case_name.encode("ascii"),
        input_state.encode("ascii"),
    ).decode("ascii")


def tree_value(library, bitness, case_id, input_bits):
    return library.bb_tree_value(
        bitness,
        case_id,
        input_bits.encode("ascii"),
    ).decode("ascii")


def table_value(library, bitness, case_id, input_bits):
    return library.bb_table_value(
        bitness,
        case_id,
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


def assert_metric_tensor_consistent(
    tensor_func,
    tensor_name,
    bitness,
    case_ids,
    expected,
):
    case_array = (ctypes.c_size_t * len(case_ids))(*case_ids)
    output = (ctypes.c_float * len(case_ids))()
    tensor_func(
        bitness,
        case_array,
        len(case_ids),
        output,
    )

    actual = list(output)
    expected = [float(value) for value in expected]
    assert actual == expected, (tensor_name, actual, expected)


def assert_value_tensor_consistent(
    library,
    tensor_func,
    value_func,
    tensor_name,
    bitness,
    case_ids,
    reps,
    seed,
    expected_inputs=None,
):
    case_array = (ctypes.c_size_t * len(case_ids))(*case_ids)
    sample_size = 2 * bitness + 1
    output = (ctypes.c_float * (len(case_ids) * reps * sample_size))()
    tensor_func(
        bitness,
        case_array,
        len(case_ids),
        reps,
        seed,
        output,
    )

    repeated = (ctypes.c_float * len(output))()
    tensor_func(
        bitness,
        case_array,
        len(case_ids),
        reps,
        seed,
        repeated,
    )
    assert list(output) == list(repeated), tensor_name

    for case_index, case_id in enumerate(case_ids):
        generated_inputs = []
        for rep in range(reps):
            offset = (case_index * reps + rep) * sample_size
            actual = list(output[offset : offset + sample_size])
            bits = "".join("1" if value == 1.0 else "0" for value in actual[:bitness])
            generated_inputs.append(bits)
            value = value_func(library, bitness, case_id, bits)
            expected = [bit_to_signed_float(bit) for bit in value]
            assert actual == expected, (
                f"{tensor_name}({bitness}) mismatch: "
                f"case_index={case_index}, case_id={case_id}, rep={rep}, "
                f"input={bits}, actual={actual}, expected={expected}"
            )

        assert_block_inputs_consistent(generated_inputs, bitness, tensor_name)
        if expected_inputs is not None:
            assert generated_inputs == expected_inputs[case_index], (
                tensor_name,
                generated_inputs,
                expected_inputs[case_index],
            )


def assert_restrictions_tensor_consistent(
    library,
    tensor_func,
    value_func,
    tensor_name,
    bitness,
    case_ids,
    reps,
    seed,
):
    free_bits = bitness - 1
    sample_size = 2 * free_bits + 1
    restrictions = bitness * 2
    case_array = (ctypes.c_size_t * len(case_ids))(*case_ids)
    output = (ctypes.c_float * (
        len(case_ids) * restrictions * reps * sample_size
    ))()
    tensor_func(
        bitness,
        case_array,
        len(case_ids),
        reps,
        seed,
        output,
    )

    repeated = (ctypes.c_float * len(output))()
    tensor_func(
        bitness,
        case_array,
        len(case_ids),
        reps,
        seed,
        repeated,
    )
    assert list(output) == list(repeated), tensor_name

    for case_index, case_id in enumerate(case_ids):
        for fixed_bit_id in range(bitness):
            for fixed_bit_value in range(2):
                restriction_id = fixed_bit_id * 2 + fixed_bit_value
                generated_inputs = []

                for rep in range(reps):
                    offset = (
                        (case_index * restrictions + restriction_id) * reps + rep
                    ) * sample_size
                    actual = list(output[offset : offset + sample_size])
                    free_input = "".join(
                        "1" if value == 1.0 else "0"
                        for value in actual[:free_bits]
                    )
                    generated_inputs.append(free_input)

                    full_input = list("0" * bitness)
                    full_input[fixed_bit_id] = str(fixed_bit_value)
                    for coord, bit_value in enumerate(free_input):
                        full_bit_id = coord if coord < fixed_bit_id else coord + 1
                        full_input[full_bit_id] = bit_value
                    full_input = "".join(full_input)

                    direct_value = value_func(library, bitness, case_id, full_input)
                    expected_bits = (
                        free_input
                        + direct_value[bitness]
                        + "".join(
                            direct_value[bitness + 1 + full_bit_id]
                            for full_bit_id in range(bitness)
                            if full_bit_id != fixed_bit_id
                        )
                    )
                    expected = [bit_to_signed_float(bit) for bit in expected_bits]
                    assert actual == expected, (
                        f"{tensor_name}({bitness}) mismatch: "
                        f"case_index={case_index}, case_id={case_id}, "
                        f"restriction_id={restriction_id}, rep={rep}, "
                        f"input={free_input}, actual={actual}, expected={expected}"
                    )

                assert_block_inputs_consistent(generated_inputs, free_bits, tensor_name)


def assert_case_consistent(library, value_func, value_name, bitness, case_id, input_bits):
    value = value_func(library, bitness, case_id, input_bits)
    assert len(value) == 2 * bitness + 1, (
        f"{value_name}({bitness}, {case_id}, {input_bits}) length: "
        f"actual={len(value)}, expected={2 * bitness + 1}, value={value}"
    )
    assert value[:bitness] == input_bits, (
        f"{value_name}({bitness}, {case_id}, {input_bits}) input prefix: "
        f"actual={value[:bitness]}, expected={input_bits}, value={value}"
    )

    repeat_value = value_func(library, bitness, case_id, input_bits)
    assert value == repeat_value, (
        f"{value_name}({bitness}, {case_id}, {input_bits}) is not stable: "
        f"first={value}, second={repeat_value}"
    )

    for bit_id in range(bitness):
        flipped = list(input_bits)
        flipped[bit_id] = "1" if flipped[bit_id] == "0" else "0"
        flipped_value = value_func(library, bitness, case_id, "".join(flipped))
        assert value[bitness + 1 + bit_id] == flipped_value[bitness], (
            f"{value_name}({bitness}, {case_id}, {input_bits}) "
            f"flip bit {bit_id}: sampled={value[bitness + 1 + bit_id]}, "
            f"direct={flipped_value[bitness]}, flipped_input={''.join(flipped)}"
        )

    return value


def test_tree_cases(library):
    print(f"Check tree cases ...")

    for case in TREE_CASES:
        (
            bitness,
            case_id,
            input_bits,
            expected_value,
            expected_depth,
            expected_nodes,
        ) = case
        value = assert_case_consistent(
            library,
            tree_value,
            "bb_tree_value",
            bitness,
            case_id,
            input_bits,
        )
        assert value == expected_value, (
            f"bb_tree_value({bitness}, {case_id}, {input_bits}): "
            f"actual={value}, expected={expected_value}"
        )

        flipped = list(input_bits)
        flipped[0] = "1" if flipped[0] == "0" else "0"
        flipped = "".join(flipped)

        first = tree_value(library, bitness, case_id, input_bits)
        other = tree_value(library, bitness, case_id, flipped)
        first_repeat = tree_value(library, bitness, case_id, input_bits)
        other_repeat = tree_value(library, bitness, case_id, flipped)
        assert first == first_repeat, (
            f"bb_tree_value({bitness}, {case_id}, {input_bits}) repeat: "
            f"first={first}, second={first_repeat}"
        )
        assert other == other_repeat, (
            f"bb_tree_value({bitness}, {case_id}, {flipped}) repeat: "
            f"first={other}, second={other_repeat}"
        )


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
            "bb_table_value",
            bitness,
            case_id,
            input_bits,
        )
        assert value == expected_value, (
            f"bb_table_value({bitness}, {case_id}, {input_bits}): "
            f"actual={value}, expected={expected_value}"
        )

def test_table_big_cases(library):
    print(f"Check big table cases ...")

    for bitness, case_id, input_bits, expected_value in TABLE_BIG_CASES:
        assert bitness > library.bb_table_solvable_bitness(), bitness
        value = assert_case_consistent(
            library,
            table_value,
            "bb_table_value",
            bitness,
            case_id,
            input_bits,
        )
        assert len(value) == 2 * bitness + 1, (bitness, case_id, value)


def legacy_test_metric_tensors(library):
    print("Check depth/node tensor APIs ...")

    assert_metric_tensor_consistent(
        library.bb_tree_nodes_tensor,
        "bb_tree_nodes_tensor",
        17,
        [0, 42, 239],
        [0, 42, 239],
    )
    assert_metric_tensor_consistent(
        library.bb_tree_depth_tensor,
        "bb_tree_depth_tensor",
        17,
        [0, 42, 239],
        [0, 9, 16],
    )
    assert_metric_tensor_consistent(
        library.bb_table_nodes_tensor,
        "bb_table_nodes_tensor",
        4,
        [0, 3190, 11304],
        [0, 7, 6],
    )
    assert_metric_tensor_consistent(
        library.bb_table_depth_tensor,
        "bb_table_depth_tensor",
        7,
        [42, 239],
        [7, 7],
    )


def legacy_test_value_tensors(library):
    print(f"Check value tensor APIs ...")

    assert_value_tensor_consistent(
        library,
        library.bb_tree_value_tensor,
        tree_value,
        "bb_tree_value_tensor",
        17,
        [42],
        8,
        239,
        [[
            "10011010010000001",
            "01100101110000001",
            "10011010001111110",
            "01100101101111110",
            "00000111011011001",
            "01110000001001110",
            "00111101111110101",
            "10110110111110001",
        ]],
    )
    assert_value_tensor_consistent(
        library,
        library.bb_tree_value_tensor,
        tree_value,
        "bb_tree_value_tensor",
        24,
        [188, 189],
        4,
        12_345,
    )
    assert_value_tensor_consistent(
        library,
        library.bb_table_value_tensor,
        table_value,
        "bb_table_value_tensor",
        4,
        [0],
        8,
        0,
    )
    assert_value_tensor_consistent(
        library,
        library.bb_table_value_tensor,
        table_value,
        "bb_table_value_tensor",
        7,
        [42, 239],
        6,
        0xffffffffffffffff,
    )
    assert_value_tensor_consistent(
        library,
        library.bb_table_value_tensor,
        table_value,
        "bb_table_value_tensor",
        17,
        [42, 43],
        4,
        42,
    )
    assert_restrictions_tensor_consistent(
        library,
        library.bb_tree_restrictions_tensor,
        tree_value,
        "bb_tree_restrictions_tensor",
        17,
        [42, 43],
        4,
        239,
    )
    assert_restrictions_tensor_consistent(
        library,
        library.bb_table_restrictions_tensor,
        table_value,
        "bb_table_restrictions_tensor",
        7,
        [42, 239],
        4,
        17,
    )
    assert_restrictions_tensor_consistent(
        library,
        library.bb_table_restrictions_tensor,
        table_value,
        "bb_table_restrictions_tensor",
        17,
        [42],
        4,
        42,
    )


def legacy_test_table_value_tensor_golden(library):
    print("Check bb_table_value_tensor ...")

    bitness = 8
    case_id = 509_888_371  # random.Random(239).randrange(1 << 32)
    reps = 128
    seed = 20_250_710
    sample_size = 2 * bitness + 1
    case_ids = (ctypes.c_size_t * 1)(case_id)
    output = (ctypes.c_float * (reps * sample_size))()

    library.bb_table_value_tensor(
        bitness,
        case_ids,
        1,
        reps,
        seed,
        output,
    )

    actual_bits = [
        "".join(
            "1" if value == 1.0 else "0"
            for value in output[rep * sample_size : (rep + 1) * sample_size]
        )
        for rep in range(reps)
    ]
    actual = [
        f"{row[:bitness]}|{row[bitness]}|{row[bitness + 1:]}"
        for row in actual_bits
    ]

    expected_block_inversions = """
10101111|1|11010000
01101111|1|11100110
10011111|0|10101000
01011111|0|01011110
10100111|0|11001111
01100111|0|11001000
10010111|1|10000110
01010111|1|01000000
10101011|0|10011101
01101011|1|01110111
10011011|0|00101010
01011011|1|00110010
10100011|1|00110000
01100011|0|00001000
10010011|1|01110110
01010011|0|10001101
10101101|0|00101010
01101101|1|00110110
10011101|0|11011101
01011101|1|11110101
10100101|1|10000000
01100101|0|01001000
10010101|1|10000110
01010101|0|01001010
10101001|0|11000001
01101001|1|11010110
10011001|1|00001000
01011001|1|00100111
10100001|0|00100111
01100001|0|00101001
10010001|1|11011110
01010001|0|11011000
10101110|0|01101101
01101110|0|10110101
10011110|0|10010010
01011110|0|01110010
10100110|1|01100000
01100110|0|10000000
10010110|0|11010001
01010110|0|11000101
10101010|1|01100010
01101010|1|10110001
10011010|0|00010000
01011010|0|00111011
10100010|0|00101111
01100010|0|00111010
10010010|0|01010001
01010010|1|10110000
10101100|0|10100100
01101100|0|01110001
10011100|1|11010000
01011100|1|11110101
10100100|0|00110111
01100100|0|00010100
10010100|0|00111001
01010100|0|00101000
10101000|1|01111010
01101000|0|10101011
10011000|0|00110101
01011000|1|00010101
10100000|1|11011000
01100000|1|11010000
10010000|0|11100001
01010000|0|11101010
""".strip().splitlines()
    expected_random = """
00111000|0|10001110
00001110|0|11011101
10100111|0|11001111
00101011|1|01000110
11100001|0|00111001
01100011|0|00001000
01101000|0|10101011
11010101|0|01001100
11100010|0|00001110
00011001|0|11001100
00011001|0|11001100
11011110|0|00101010
00000011|0|10000001
10000010|1|10001101
01001100|1|10010111
11111010|0|10010111
10110110|0|00010010
00110001|0|00100011
01110110|0|00001110
11010000|1|00000011
10111011|1|01001100
11110011|0|01101010
01000100|0|01001000
11110100|0|11000000
00100110|0|10100001
10000010|1|10001101
01011111|0|01011110
11011101|1|10000001
01110100|1|01001100
01110001|0|10001001
10101011|0|10011101
00110001|0|00100011
11101111|1|11111001
10110010|0|00000011
01111111|0|11010111
11110001|1|00101000
01010000|0|11101010
00111101|0|01100011
10001000|1|01100110
00011000|0|01001100
11000001|1|11010010
00111100|1|01111010
00100100|0|00111101
10101001|0|11000001
11000100|0|01001000
00110111|1|00111100
00101001|1|01000010
10101000|1|01111010
10001110|1|00001110
01110010|1|00101010
10111101|0|00000010
11010001|1|01110011
11001011|1|10001100
11010011|1|01010011
00001010|1|11001000
01110101|0|00001001
10010000|0|11100001
11001001|0|00101011
01000001|1|10000000
01011100|1|11110101
00011001|0|11001100
11110011|0|01101010
01001111|1|11100111
10101111|1|11010000
""".strip().splitlines()

    expected = expected_block_inversions + expected_random
    assert actual == expected, (actual, expected)


def collect_owned_data(
    library,
    workers,
    kind,
    bitness,
    cases,
    reps,
    seed,
    restriction_chunk_cases=0,
):
    generator = library.bb_generator_create(workers)
    assert generator
    if kind == "tree":
        data = library.bb_tree_value_tensor(
            generator,
            bitness,
            cases,
            reps,
            seed,
        )
    else:
        data = library.bb_table_value_tensor(
            generator,
            bitness,
            cases,
            reps,
            restriction_chunk_cases,
            seed,
        )
    assert data
    library.bb_data_acquire(data)
    assert library.bb_data_bitness(data) == bitness
    assert library.bb_data_cases(data) == cases
    assert library.bb_data_reps(data) == reps

    value_count = cases * reps * (2 * bitness + 1)
    value_pointer = library.bb_data_take_values(data)
    values = list(value_pointer[:value_count])
    library.bb_float_buffer_destroy(value_pointer)
    target_pointer = library.bb_data_take_targets(data)
    targets = None
    if target_pointer:
        targets = list(target_pointer[:cases])
        library.bb_float_buffer_destroy(target_pointer)

    restrictions = []
    if restriction_chunk_cases:
        for first_case in range(0, cases, restriction_chunk_cases):
            chunk_cases = min(restriction_chunk_cases, cases - first_case)
            tensor = library.bb_table_restrictions_tensor(
                generator,
                data,
                first_case,
                chunk_cases,
            )
            library.bb_tensor_acquire(tensor)
            count = chunk_cases * 2 * bitness * reps * (2 * bitness - 1)
            restriction_pointer = library.bb_tensor_take_values(tensor)
            restrictions.extend(restriction_pointer[:count])
            library.bb_float_buffer_destroy(restriction_pointer)
            library.bb_tensor_release(generator, tensor)

    library.bb_data_release(generator, data)
    library.bb_generator_destroy(generator)
    return values, targets, restrictions if restrictions else None


def test_owned_data(library):
    print("Check C++-owned threaded data ...")

    for kind, bitness, chunk in (
        ("tree", 17, 0),
        ("table", 8, 0),
        ("table", 17, 2),
    ):
        single = collect_owned_data(
            library,
            1,
            kind,
            bitness,
            4,
            8,
            20_250_710,
            chunk,
        )
        threaded = collect_owned_data(
            library,
            16,
            kind,
            bitness,
            4,
            8,
            20_250_710,
            chunk,
        )
        for first, second in zip(single, threaded):
            if first is None:
                assert second is None
            else:
                assert first == second, (kind, bitness)

    values, targets, restrictions = collect_owned_data(
        library,
        4,
        "table",
        8,
        1,
        8,
        2025,
    )
    assert targets == [0.0], targets
    assert restrictions is None
    rows = [values[rep * 17 : (rep + 1) * 17] for rep in range(8)]
    inputs = [
        "".join("1" if value == 1.0 else "0" for value in row[:8])
        for row in rows
    ]
    assert inputs == [
        "11010011",
        "00100011",
        "11011100",
        "00101100",
        "01011111",
        "11001111",
        "01010111",
        "11100011",
    ], inputs
    assert_block_inputs_consistent(inputs, 8, "bb_table_value_tensor")


def test_circuit_discovery(library):
    print(f"Check bb_circuit discovery ...")

    sets = split_list(library.bb_circuit_sets())
    assert sets == expected_circuit_sets()

    for set_name in sets:
        assert split_list(library.bb_circuit_cases(set_name.encode("ascii"))) == (
            expected_circuit_cases(set_name)
        )


def test_circuit_metadata(library):
    print(f"Check bb_circuit metadata ...")

    for set_name in expected_circuit_sets():
        for case_name in expected_circuit_cases(set_name):
            expected_inputs, expected_outputs = read_aig_metadata(set_name, case_name)
            assert library.bb_circuit_inputs(
                set_name.encode("ascii"),
                case_name.encode("ascii"),
            ) == expected_inputs
            assert library.bb_circuit_outputs(
                set_name.encode("ascii"),
                case_name.encode("ascii"),
            ) == expected_outputs

    assert library.bb_circuit_inputs(b"iscas85", b"c17") == 5
    assert library.bb_circuit_outputs(b"iscas85", b"c17") == 2

    assert library.bb_circuit_inputs(b"iscas87", b"s27") == 7
    assert library.bb_circuit_outputs(b"iscas87", b"s27") == 1


def test_circuit_value(library):
    print(f"Check bb_circuit_value ...")

    c17 = circuit_value(library, "iscas85", "c17", "00000")
    assert len(c17) == 5 + 2 * 6
    assert c17 == circuit_value(library, "iscas85", "c17", "00000")

    s27 = circuit_value(library, "iscas87", "s27", "0000000")
    assert len(s27) == 7 + 1 * 8
    assert s27 == circuit_value(library, "iscas87", "s27", "0000000")


if __name__ == "__main__":
    library = load_library()
    test_tree_cases(library)
    test_table_solvable_cases(library)
    test_table_big_cases(library)
    test_owned_data(library)
    test_circuit_discovery(library)
    test_circuit_metadata(library)
    test_circuit_value(library)
