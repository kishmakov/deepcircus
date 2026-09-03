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
    {8, 0, "01010101", "01010101011000101", 8, 108},
    {8, 3190, "00011010", "00011010101000101", 8, 114},
    {8, 11304, "11011001", "11011001011011101", 8, 115},
    {9, 3261348405, "110001010", "1100010101010101111", 9, 216},
    {9, 390455940, "010011101", "0100111010010101101", 9, 218},
    {10, 2547012052, "1101110010", "110111001001000001100", 10, 426},
    {10, 883941716, "0111110101", "011111010111110110100", 10, 426},
    {11, 42, "01010101010", "01010101010111010000010", 11, 844},
    {11, 23901, "01010101010", "01010101010111110010100", 11, 859},
    {12, 239, "101010100110", "1010101001101100001000010", 12, 1675},
};

struct TableBigGoldenCase {
    uint16_t bitness;
    uint64_t seed;
    std::string_view input;
    std::string_view expected_value;
};

// Golden value samples above the solvable bitness. The bitness-16 rows still
// materialize their truth table and the rest are sparse, so the set straddles
// the boundary where TableCase stops holding the function outright.
constexpr TableBigGoldenCase kTableBigGoldenCases[] = {
    {16, 42, "0101010101010101", "010101010101010101000011111100000"},
    {16, 239566, "1010101010101010", "101010101010101010001010011101000"},
    {17, 42, "01010101010101010", "01010101010101010111100111000001010"},
    {24, 188, "110010100111000101010011", "1100101001110001010100111100010000010110110001000"},
    {32, 320, "01010101010101010101010101010101", "01010101010101010101010101010101100011100100011101011011010101000"},
    {48, 480, "110010101100101011001010110010101100101011001010",
     "1100101011001010110010101100101011001010110010101001101001011011001010101001010101110000001000101"},
    {64, 640, "0011010100110101001101010011010100110101001101010011010100110101",
     "00110101001101010011010100110101001101010011010100110101001101011010011100011111100010010110001011010100"
     "1001111000010101111100100"},
    {100, 1000, "0100110011010011001101001100110100110011010011001101001100110100110011010011001101001100110100110011",
     "01001100110100110011010011001101001100110100110011010011001101001100110100110011010011001101001100111100"
     "0000000001100110100011100000100010000110110110010110101101010101010101100011010110011011100011001"},
    {128, 1280,
     "01101001100101100110100110010110011010011001011001101001100101100110100110010110011010011001011001101001"
     "100101100110100110010110",
     "01101001100101100110100110010110011010011001011001101001100101100110100110010110011010011001011001101001"
     "10010110011010011001011000000100000010001100001111110110101010101010100010010000110100001001111001100100"
     "1111010011001110001010110101001101011000000111100"},
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
        ASSERT_LE(c.bitness, tools::kMaxSolvableBitness);

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
        ASSERT_GT(c.bitness, tools::kMaxSolvableBitness);

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

TEST(TableTest, SerializeStoresSeedLittleEndian) {
    const func::TableCase table(8, 0x0123456789abcdefull);
    EXPECT_EQ(table.serialize(), (std::vector<uint8_t>{0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01}));
}

TEST(TableTest, DeserializeRestoresSeededTable) {
    const func::TableCase original(8, 0x0123456789abcdefull);
    const std::vector<uint8_t> bytes = original.serialize();
    const func::TableCase restored(8, bytes);

    EXPECT_EQ(restored.TruthTable(), original.TruthTable());
    EXPECT_EQ(restored.serialize(), bytes);
}
