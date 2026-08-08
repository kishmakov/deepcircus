#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "tools/random.h"

TEST(RandomTest, SplitMixGoldenSequence) {
    uint64_t state = 0;
    EXPECT_EQ(tools::SplitMix64(state), 0xe220a8397b1dcdafull);
    EXPECT_EQ(tools::SplitMix64(state), 0x6e789e6aa1b965f4ull);
    EXPECT_EQ(tools::Mix(0), 0xe220a8397b1dcdafull);
    EXPECT_EQ(tools::Mix64(0), 0);
}

TEST(RandomTest, RandomUsesSplitMixSequence) {
    tools::Random random(0);
    EXPECT_EQ(random.Next(), 0xe220a8397b1dcdafull);
    EXPECT_EQ(random.Next(), 0x6e789e6aa1b965f4ull);

    tools::Random bounded(17);
    for (size_t draw = 0; draw < 100; ++draw) {
        EXPECT_LT(bounded.Below(7), 7);
    }
    EXPECT_EQ(bounded.Below(1), 0);
}

TEST(RandomTest, BoolIsLowBit) {
    tools::Random values(42);
    tools::Random bits(42);
    for (size_t draw = 0; draw < 100; ++draw) {
        EXPECT_EQ(bits.Bool(), (values.Next() & 1) != 0);
    }
}
