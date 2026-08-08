#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "generator.h"
#include "sample.h"
#include "tree.h"

namespace {

struct TreeGoldenCase {
    uint16_t bitness;
    uint64_t seed;
    std::string_view input;
    std::string_view expected_value;
    size_t expected_depth;
    size_t expected_internal_nodes;
};

// Golden (bitness, seed, input) -> value samples, plus the depth and
// internal-node count of the built tree, a function of (bitness, seed).
// Pinned so refactors of the tree builder, evaluator, or sampler stay
// deterministic.
constexpr TreeGoldenCase kTreeGoldenCases[] = {
    {17, 1048, "01010101010101010", "01010101010101010110011100111111010", 17, 1379},
    {24, 378, "110010100111000101010011", "1100101001110001010100110000010000000000010010000", 24, 2502},
    {32, 1088, "01010101010101010101010101010101",
     "01010101010101010101010101010101000010000000100000000000000000000", 23, 1189},
    {48, 566, "110010101100101011001010110010101100101011001010",
     "1100101011001010110010101100101011001010110010100000000000000000000000000000000000000010000000000", 21, 941},
    {64, 750, "0011010100110101001101010011010100110101001101010011010100110101",
     "0011010100110101001101010011010100110101001101010011010100110101"
     "11111111101111111111111111111111111111111111111111111111111011111",
     24, 1546},
    {100, 1231,
     "0100110011010011001101001100110100110011010011001101001100110100110011010011001101001100110100110011",
     "010011001101001100110100110011010011001101001100110100110011010011001101001100110100110011010011001100000000000000000000000000000000000000000000100000000000000000000100000000100000000001000010000000000",
     25, 2411},
    {128, 1540,
     "01101001100101100110100110010110011010011001011001101001100101100110100110010110011010011001011001101001"
     "100101100110100110010110",
     "01101001100101100110100110010110011010011001011001101001100101100110100110010110011010011001011001101001"
     "100101100110100110010110000000000000000000000000000000000100000000000000000000000000000100000000000000"
     "000000000000000000000000010000000000000000000000000",
     18, 310},
};

std::string TreeValue(const TreeGoldenCase& c, std::string_view input) {
    return gen::TreeValue(c.bitness, c.seed, tools::BitsFromChars(input));
}

}  // namespace

TEST(TreeTest, GoldenValues) {
    for (const TreeGoldenCase& c : kTreeGoldenCases) {
        const std::string value = TreeValue(c, c.input);
        EXPECT_EQ(value, c.expected_value) << "bitness=" << c.bitness << " seed=" << c.seed;

        const gen::TreeCase tree(c.bitness, c.seed);
        EXPECT_EQ(tree.depth, c.expected_depth) << "bitness=" << c.bitness << " seed=" << c.seed;
        EXPECT_EQ(tree.nodes.size() - tree.num_leafs, c.expected_internal_nodes)
            << "bitness=" << c.bitness << " seed=" << c.seed;
    }
}

TEST(TreeTest, ValueConsistency) {
    for (const TreeGoldenCase& c : kTreeGoldenCases) {
        const std::string value = TreeValue(c, c.input);

        ASSERT_EQ(value.size(), 2u * c.bitness + 1) << "bitness=" << c.bitness << " seed=" << c.seed;
        EXPECT_EQ(value.substr(0, c.bitness), c.input) << "bitness=" << c.bitness << " seed=" << c.seed;

        // The value must be stable across repeated calls.
        EXPECT_EQ(value, TreeValue(c, c.input)) << "bitness=" << c.bitness << " seed=" << c.seed;

        // Each sampled "flip bit i" slot must match evaluating the tree on the
        // input with bit i flipped. One case built once: at the high bitnesses
        // here rebuilding it per bit would dominate the suite's runtime.
        const gen::TreeCase tree(c.bitness, c.seed);
        for (uint16_t bit_id = 0; bit_id < c.bitness; ++bit_id) {
            std::string flipped(c.input);
            flipped[bit_id] = flipped[bit_id] == '0' ? '1' : '0';
            const std::string flipped_value = tree.SampledValueString(tools::BitsFromChars(flipped));
            EXPECT_EQ(value[c.bitness + 1 + bit_id], flipped_value[c.bitness])
                << "bitness=" << c.bitness << " seed=" << c.seed << " bit_id=" << bit_id;
        }
    }
}
