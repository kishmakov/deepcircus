#include "case.h"
#include "generator.h"

#include <gtest/gtest.h>

#include <vector>

namespace gen {
namespace {

class ConstantCase : public Case {
public:
    using Case::Case;

private:
    bool Evaluate(const std::vector<bool>&) const override { return false; }
};

TEST(CaseTest, SampleValuesEmbedsExpandInputs) {
    const uint16_t bitness = 7;
    const InputShape shape{3, 8};

    ConstantCase drawer(bitness, 42);
    std::vector<std::vector<bool>> sequences;
    for (uint16_t batch = 0; batch < shape.batches; ++batch) {
        sequences.push_back(drawer.GenerateSequence());
    }
    const std::vector<bool> points = ExpandInputs(shape, sequences);

    ConstantCase sampler(bitness, 42);
    const std::vector<bool> values = sampler.SampleValues(shape);

    const size_t point_count = static_cast<size_t>(shape.batches) * shape.batch_size;
    const size_t sample_size = 2 * bitness + 1;
    ASSERT_EQ(points.size(), point_count * bitness);
    ASSERT_EQ(values.size(), point_count * sample_size);
    for (size_t point = 0; point < point_count; ++point) {
        for (uint16_t bit = 0; bit < bitness; ++bit) {
            EXPECT_EQ(values[point * sample_size + bit], points[point * bitness + bit]);
        }
    }
}

}  // namespace
}  // namespace gen
