#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "generator.h"
#include "tree.h"

namespace {

// Fixed seed the golden values below were generated with; tree structure is a
// function of (bitness, case_id, seed).
constexpr uint64_t kGoldenSeed = 0xC0FFEE;

struct TreeGoldenCase {
    uint16_t bitness;
    size_t case_id;
    std::string_view input;
    std::string_view expected_value;
    size_t expected_depth;
    size_t expected_internal_nodes;
};

// Golden (bitness, case_id, input) -> value samples, plus the optimal depth and
// internal-node count of the built tree. Pinned so refactors of the tree
// builder, evaluator, or sampler stay deterministic.
constexpr TreeGoldenCase kTreeGoldenCases[] = {
    {17, 42, "01010101010101010", "01010101010101010000000000010100000", 10, 42},
    {24, 188, "110010100111000101010011", "1100101001110001010100110000000100000000000011000", 16, 188},
    {32, 320, "01010101010101010101010101010101",
     "01010101010101010101010101010101111111110001111111111111101111111", 19, 320},
    {48, 480, "110010101100101011001010110010101100101011001010",
     "1100101011001010110010101100101011001010110010101011111111111111111111111111111111111111111111111", 18, 480},
    {64, 640, "0011010100110101001101010011010100110101001101010011010100110101",
     "0011010100110101001101010011010100110101001101010011010100110101"
     "11110111111110111111111111011111111101110111111111011111111111111",
     21, 640},
    {100, 1000,
     "0100110011010011001101001100110100110011010011001101001100110100110011010011001101001100110100110011",
     "010011001101001100110100110011010011001101001100110100110011010011001101001100110100110011010011001100000000000010000000000000000000000000000001000000101000000100001010000000000000000010000000100000000",
     21, 1000},
    {128, 1280,
     "01101001100101100110100110010110011010011001011001101001100101100110100110010110011010011001011001101001"
     "100101100110100110010110",
     "01101001100101100110100110010110011010011001011001101001100101100110100110010110011010011001011001101001"
     "1001011001101001100101100000000000000000000000000000000000000000000000000000000000000000000000000000000000"
     "00000000000000000000000000000000000001010000000",
     23, 1280},
};

std::string TreeValue(const TreeGoldenCase& c, std::string_view input) {
    return gen::TreeValue(c.bitness, c.case_id, kGoldenSeed, input);
}

}  // namespace

TEST(TreeTest, GoldenValues) {
    for (const TreeGoldenCase& c : kTreeGoldenCases) {
        const std::string value = TreeValue(c, c.input);
        EXPECT_EQ(value, c.expected_value) << "bitness=" << c.bitness << " case_id=" << c.case_id;

        const gen::TreeCase tree(c.bitness, c.case_id, kGoldenSeed);
        EXPECT_EQ(tree.depth, c.expected_depth) << "bitness=" << c.bitness << " case_id=" << c.case_id;
        EXPECT_EQ(tree.nodes.size() - tree.num_leafs, c.expected_internal_nodes)
            << "bitness=" << c.bitness << " case_id=" << c.case_id;
    }
}

TEST(TreeTest, ValueConsistency) {
    for (const TreeGoldenCase& c : kTreeGoldenCases) {
        const std::string value = TreeValue(c, c.input);

        ASSERT_EQ(value.size(), 2u * c.bitness + 1) << "bitness=" << c.bitness << " case_id=" << c.case_id;
        EXPECT_EQ(value.substr(0, c.bitness), c.input) << "bitness=" << c.bitness << " case_id=" << c.case_id;

        // The value must be stable across repeated calls.
        EXPECT_EQ(value, TreeValue(c, c.input)) << "bitness=" << c.bitness << " case_id=" << c.case_id;

        // Each sampled "flip bit i" slot must match evaluating the tree on the
        // input with bit i flipped.
        for (uint16_t bit_id = 0; bit_id < c.bitness; ++bit_id) {
            std::string flipped(c.input);
            flipped[bit_id] = flipped[bit_id] == '0' ? '1' : '0';
            const std::string flipped_value = TreeValue(c, flipped);
            EXPECT_EQ(value[c.bitness + 1 + bit_id], flipped_value[c.bitness])
                << "bitness=" << c.bitness << " case_id=" << c.case_id << " bit_id=" << bit_id;
        }
    }
}
