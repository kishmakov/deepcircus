#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

#include "sample.h"

TEST(SampleTest, DeterministicGoldenPoints) {
    size_t draw = 0;
    const std::vector<bool> points = tools::SampleInputs({3, 4}, 5, [&draw] { return draw++ % 3 == 0; });
    const std::string expected = "100100001111011011110100100111110001011000100000111110011011";
    ASSERT_EQ(points.size(), expected.size());
    for (size_t bit = 0; bit < points.size(); ++bit) {
        EXPECT_EQ(points[bit], expected[bit] == '1') << "bit " << bit;
    }
}

TEST(SampleTest, RejectsOnePointBatch) {
    EXPECT_DEATH(tools::SampleInputs({2, 1}, 5, [] { return false; }), "shape.batch_size > 1");
}
