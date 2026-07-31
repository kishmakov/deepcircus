#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "sample.h"

// The bit-to-group mapping feeds deterministic input generation, so the
// contiguous way-0 split and its rotation by way % bitness are pinned here.
TEST(SampleTest, SplitBitsInGroups) {
    const std::vector<uint16_t> way0 = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    const std::vector<uint16_t> way1 = {4, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4};
    const std::vector<uint16_t> way2 = {4, 4, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4};

    EXPECT_EQ(tools::SplitBitsInGroups(20, 5, 0), way0);
    EXPECT_EQ(tools::SplitBitsInGroups(20, 5, 1), way1);
    EXPECT_EQ(tools::SplitBitsInGroups(20, 5, 2), way2);
}
