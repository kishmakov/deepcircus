#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "func/table.h"
#include "generator.h"
#include "sample.h"
#include "tools/solver.h"

namespace {

struct TableGoldenCase {
    uint16_t bitness;
    uint64_t seed;
    std::string_view input;
    std::string_view expected_value;
    size_t expected_depth;
    size_t expected_internal_nodes;
};

// Golden (bitness, seed, input) -> value samples for solvable bitnesses,
// plus the optimal depth and internal-node count solved from the truth table.
// Pinned so refactors of the table sampler and solver stay deterministic.
constexpr TableGoldenCase kTableGoldenCases[] = {
    {4, 0, "0101", "010111111", 4, 7},
    {4, 3190, "0001", "000101000", 4, 7},
    {4, 11304, "1101", "110110110", 4, 7},
    {5, 3261348405, "11000", "11000101111", 5, 13},
    {5, 390455940, "01001", "01001000011", 5, 16},
    {6, 2547012052, "110111", "1101110101110", 6, 26},
    {6, 883941716, "011111", "0111110011110", 6, 31},
    {7, 42, "0101010", "010101001001110", 7, 59},
    {7, 239, "1010101", "101010100000000", 7, 53},
    {11, 23901, "01010101010", "01010101010111110010100", 11, 859},
};

struct TableBigGoldenCase {
    uint16_t bitness;
    uint64_t seed;
    std::string_view input;
    std::string_view expected_value;
};

// Golden value samples above the solvable bitness (sparse tables).
constexpr TableBigGoldenCase kTableBigGoldenCases[] = {
    {16, 42, "0101010101010101", "010101010101010111011010100110110"},
    {16, 239566, "1010101010101010", "101010101010101011001110011111001"},
    {17, 42, "01010101010101010", "01010101010101010000110111110000000"},
    {24, 188, "110010100111000101010011", "1100101001110001010100111111000101111101101011001"},
    {32, 320, "01010101010101010101010101010101", "01010101010101010101010101010101011010100011011000111001011101111"},
    {48, 480, "110010101100101011001010110010101100101011001010",
     "1100101011001010110010101100101011001010110010100100101001000110111001110111110010011011100000100"},
    {64, 640, "0011010100110101001101010011010100110101001101010011010100110101",
     "0011010100110101001101010011010100110101001101010011010100110101"
     "11100001010010001001000001010110110010001000111100101010111011000"},
    {100, 1000, "0100110011010011001101001100110100110011010011001101001100110100110011010011001101001100110100110011",
     "0100110011010011001101001100110100110011010011001101001100110100110011010011001101001100110100110011"
     "01001110001110001100101101000010110001011001001000100001100111011110100101100111100001110011010001010"},
    {128, 1280,
     "01101001100101100110100110010110011010011001011001101001100101100110100110010110011010011001011001101001"
     "100101100110100110010110",
     "01101001100101100110100110010110011010011001011001101001100101100110100110010110011010011001011001101001"
     "100101100110100110010110101010111101000100100101010010001100110111111111110011101100111011100001011100"
     "100000100110010001010110111001001000101111001101101"},
};

std::string TableValue(uint16_t bitness, uint64_t seed, std::string_view input) {
    return func::TableValue(bitness, seed, tools::BitsFromChars(input));
}

void CheckValueConsistency(uint16_t bitness, uint64_t seed, std::string_view input) {
    const std::string value = TableValue(bitness, seed, input);

    ASSERT_EQ(value.size(), 2u * bitness + 1) << "bitness=" << bitness << " seed=" << seed;
    EXPECT_EQ(value.substr(0, bitness), input) << "bitness=" << bitness << " seed=" << seed;

    // The value must be stable across repeated calls.
    EXPECT_EQ(value, TableValue(bitness, seed, input)) << "bitness=" << bitness << " seed=" << seed;

    // Each sampled "flip bit i" slot must match evaluating the table on the
    // input with bit i flipped.
    for (uint16_t bit_id = 0; bit_id < bitness; ++bit_id) {
        std::string flipped(input);
        flipped[bit_id] = flipped[bit_id] == '0' ? '1' : '0';
        const std::string flipped_value = TableValue(bitness, seed, flipped);
        EXPECT_EQ(value[bitness + 1 + bit_id], flipped_value[bitness])
            << "bitness=" << bitness << " seed=" << seed << " bit_id=" << bit_id;
    }
}

}  // namespace

TEST(TableTest, SolvableGoldenValues) {
    for (const TableGoldenCase& c : kTableGoldenCases) {
        ASSERT_LE(c.bitness, func::TableSolvableBitness());

        const std::string value = TableValue(c.bitness, c.seed, c.input);
        EXPECT_EQ(value, c.expected_value) << "bitness=" << c.bitness << " seed=" << c.seed;

        const func::TableCase table(c.bitness, c.seed);
        EXPECT_EQ(tools::SolveForDepth(c.bitness, table.TruthTable()), c.expected_depth)
            << "bitness=" << c.bitness << " seed=" << c.seed;
        EXPECT_EQ(tools::SolveForSize(c.bitness, table.TruthTable()), c.expected_internal_nodes)
            << "bitness=" << c.bitness << " seed=" << c.seed;
    }
}

TEST(TableTest, SolvableValueConsistency) {
    for (const TableGoldenCase& c : kTableGoldenCases) {
        CheckValueConsistency(c.bitness, c.seed, c.input);
    }
}

TEST(TableTest, BigGoldenValues) {
    for (const TableBigGoldenCase& c : kTableBigGoldenCases) {
        ASSERT_GT(c.bitness, func::TableSolvableBitness());

        const std::string value = TableValue(c.bitness, c.seed, c.input);
        EXPECT_EQ(value, c.expected_value) << "bitness=" << c.bitness << " seed=" << c.seed;
    }
}

TEST(TableTest, BigValueConsistency) {
    for (const TableBigGoldenCase& c : kTableBigGoldenCases) {
        CheckValueConsistency(c.bitness, c.seed, c.input);
    }
}

TEST(TableTest, TruthTableFollowsSeed) {
    const std::vector<bool> table = func::TableCase(11, 239).TruthTable();
    ASSERT_EQ(table.size(), size_t{1} << 11);

    // The function is the seed and nothing else: the same seed rebuilds it, a
    // neighbouring one does not.
    EXPECT_EQ(func::TableCase(11, 239).TruthTable(), table);
    EXPECT_NE(func::TableCase(11, 240).TruthTable(), table);
}
