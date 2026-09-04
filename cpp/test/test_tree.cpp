#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "func/tree.h"
#include "tools/binary_tree.h"

TEST(TreeTest, EvaluatesKnownTree) {
    tools::BinaryTree tree;
    const std::array<uint32_t, 2> children = tree.Split(0);
    tree.Value(0) = 3;
    tree.Value(children[tools::BinaryTree::kLeft]) = 0;
    tree.Value(children[tools::BinaryTree::kRight]) = 1;

    const func::TreeFunc function(func::kMinBitness, std::move(tree));
    std::vector<bool> input(func::kMinBitness, false);
    EXPECT_FALSE(function(input));
    input[3] = true;
    EXPECT_TRUE(function(input));
    EXPECT_EQ(function.Depth(), 1u);
    EXPECT_EQ(function.Size(), 1u);
}

TEST(TreeTest, SamplingIsDeterministic) {
    constexpr uint64_t seed = 42;
    const func::TreeFunc first(12, seed);
    const func::TreeFunc second(12, seed);

    EXPECT_EQ(first.serialize(), second.serialize());
    EXPECT_EQ(first.Depth(), second.Depth());
    EXPECT_EQ(first.Size(), second.Size());
}

TEST(TreeTest, SerializationRoundTrips) {
    const func::TreeFunc original(12, 1234);
    const std::vector<uint8_t> bytes = original.serialize();
    const func::TreeFunc restored(12, bytes);

    EXPECT_EQ(restored.serialize(), bytes);
    EXPECT_EQ(restored.Depth(), original.Depth());
    EXPECT_EQ(restored.Size(), original.Size());

    std::vector<bool> input(12, false);
    for (uint16_t bit = 0; bit < input.size(); ++bit) {
        input[bit] = true;
        EXPECT_EQ(restored(input), original(input));
    }
}
