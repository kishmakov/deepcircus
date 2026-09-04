#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "func/table.h"
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
    {8, 0, "01010101", "01010101111001101", 8, 98},
    {8, 3190, "00011010", "00011010010101000", 8, 106},
    {8, 11304, "11011001", "11011001110011100", 8, 110},
    {9, 3261348405, "110001010", "1100010101111011000", 9, 222},
    {9, 390455940, "010011101", "0100111011011111111", 9, 224},
    {10, 2547012052, "1101110010", "110111001001110001100", 10, 423},
    {10, 883941716, "0111110101", "011111010111111110111", 10, 429},
    {11, 42, "01010101010", "01010101010000101100011", 11, 829},
    {11, 23901, "01010101010", "01010101010101101110101", 11, 853},
    {12, 239, "101010100110", "1010101001100010110011100", 12, 1659},
};

struct TableBigGoldenCase {
    uint16_t bitness;
    uint64_t seed;
    std::string_view input;
    std::string_view expected_value;
};

// Golden value samples above the solvable bitness.
constexpr TableBigGoldenCase kTableBigGoldenCases[] = {
    {16, 42, "0101010101010101", "010101010101010111011010001101010"},
    {16, 239566, "1010101010101010", "101010101010101000111111000100010"},
    {17, 42, "01010101010101010", "01010101010101010110110100011010100"},
    {24, 188, "110010100111000101010011", "1100101001110001010100111100010000010110110001000"},
    {32, 320, "01010101010101010101010101010101", "01010101010101010101010101010101100011100100011101011011010101000"},
    {48, 480, "110010101100101011001010110010101100101011001010",
     "1100101011001010110010101100101011001010110010100000001111100110010000101111100111110011001001001"},
    {64, 640, "0011010100110101001101010011010100110101001101010011010100110101",
     "00110101001101010011010100110101001101010011010100110101001101011010011100011111100010010110001011010100"
     "1001111000010101111100101"},
    {100, 1000, "0100110011010011001101001100110100110011010011001101001100110100110011010011001101001100110100110011",
     "01001100110100110011010011001101001100110100110011010011001101001100110100110011010011001101001100111100"
     "0000000001100110100011100000100010000110110110010110101101010101010101100011010110011011100011000"},
    {128, 1280,
     "01101001100101100110100110010110011010011001011001101001100101100110100110010110011010011001011001101001"
     "100101100110100110010110",
     "01101001100101100110100110010110011010011001011001101001100101100110100110010110011010011001011001101001"
     "10010110011010011001011010110100010000001111000111001010111110000101001001011010100100101110010010100001"
     "1101000001000110110010111000111000000101010101110"},
};

std::string TableValue(uint16_t bitness, uint64_t seed, std::string_view input) {
    func::TableFunc table(bitness, seed);
    std::vector<bool> point = tools::BitsFromChars(input);

    std::string value(input);
    value.push_back(table(point) ? '1' : '0');
    for (uint16_t bit_id = 0; bit_id < bitness; ++bit_id) {
        point[bit_id] = !point[bit_id];
        value.push_back(table(point) ? '1' : '0');
        point[bit_id] = !point[bit_id];
    }
    return value;
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

        const func::TableFunc table(c.bitness, c.seed);
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
    const std::vector<bool> table = func::TableFunc(11, 239).TruthTable();
    ASSERT_EQ(table.size(), size_t{1} << 11);

    // The function is the seed and nothing else: the same seed rebuilds it, a
    // neighbouring one does not.
    EXPECT_EQ(func::TableFunc(11, 239).TruthTable(), table);
    EXPECT_NE(func::TableFunc(11, 240).TruthTable(), table);
}

TEST(TableTest, SerializeStoresSeedLittleEndian) {
    const func::TableFunc table(8, 0x0123456789abcdefull);
    EXPECT_EQ(table.serialize(), (std::vector<uint8_t>{0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01}));
}

TEST(TableTest, DeserializeRestoresSeededTable) {
    const func::TableFunc original(8, 0x0123456789abcdefull);
    const std::vector<uint8_t> bytes = original.serialize();
    const func::TableFunc restored(8, bytes);

    EXPECT_EQ(restored.TruthTable(), original.TruthTable());
    EXPECT_EQ(restored.serialize(), bytes);
}
