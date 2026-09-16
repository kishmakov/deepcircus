#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

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
    EXPECT_EQ(random.NextU64(), 0xe220a8397b1dcdafull);
    EXPECT_EQ(random.NextU64(), 0x6e789e6aa1b965f4ull);

    tools::Random bounded(17);
    for (size_t draw = 0; draw < 100; ++draw) {
        EXPECT_LT(bounded.Below(7), 7);
    }
    EXPECT_EQ(bounded.Below(1), 0);
}

TEST(RandomTest, BitsConsumeWholeWords) {
    tools::Random values(42);
    tools::Random bits(42);
    for (size_t draw = 0; draw < 100; ++draw) {
        const uint64_t word = values.NextU64();
        for (unsigned bit = 0; bit < 64; ++bit) {
            EXPECT_EQ(bits.NextBool(), ((word >> bit) & 1) != 0);
        }
    }
}

TEST(RandomTest, BytesConsumeWholeWords) {
    tools::Random values(42);
    tools::Random bytes(42);
    for (size_t draw = 0; draw < 100; ++draw) {
        const uint64_t word = values.NextU64();
        for (unsigned byte = 0; byte < 8; ++byte) {
            EXPECT_EQ(bytes.NextU8(), static_cast<uint8_t>(word >> (byte * 8)));
        }
    }
}

TEST(RandomTest, MixedWidthsPreserveTheBitStream) {
    tools::Random random(0);
    uint64_t state = 0;
    std::vector<bool> expected;
    for (unsigned word = 0; word < 200; ++word) {
        const uint64_t value = tools::SplitMix64(state);
        for (unsigned bit = 0; bit < 64; ++bit) expected.push_back((value >> bit) & 1);
    }
    size_t position = 0;
    const auto take = [&](unsigned count) {
        uint64_t value = 0;
        for (unsigned bit = 0; bit < count; ++bit) value |= uint64_t{expected[position++]} << bit;
        return value;
    };
    for (unsigned draw = 0; draw < 100; ++draw) {
        EXPECT_EQ(random.NextBool(), take(1));
        EXPECT_EQ(random.NextU8(), take(8));
        EXPECT_EQ(random.NextU64(), take(64));
    }
}
