#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "dataset.h"
#include "func/table.h"
#include "offline/read_write.h"
#include "tools/score.h"

namespace {

constexpr uint16_t kBitness = 8;
constexpr serving::SamplingShape kShape{2, 4, 239};

offline::Entry TablePair(uint64_t g_seed, uint64_t f_seed, uint8_t depth, uint16_t size) {
    return offline::Entry{{offline::FunctionKind::kTable, func::TableFunc(kBitness, g_seed).serialize()},
                          {offline::FunctionKind::kTable, func::TableFunc(kBitness, f_seed).serialize()},
                          depth,
                          size};
}

offline::Entry Unsolved(uint64_t g_seed, uint64_t f_seed) {
    return TablePair(g_seed, f_seed, offline::kUnknownDepth, offline::kUnknownSize);
}

std::string WriteFile(const std::string& name, const std::vector<offline::Entry>& entries) {
    const std::string path = testing::TempDir() + name;
    offline::Writer writer(path, static_cast<uint32_t>(entries.size()), kBitness);
    for (const offline::Entry& entry : entries) writer.Write(entry);
    return path;
}

bool RowBit(const serving::Cases& cases, uint32_t row, uint64_t bit) {
    const uint8_t byte = cases.values[row * cases.RowBytes() + bit / 8];
    return ((byte >> (bit % 8)) & 1u) != 0;
}

}  // namespace

TEST(ServingTest, SkipsEntriesWithoutTarget) {
    const std::string path =
        WriteFile("deepcircus_serving_unsolved.bin", {TablePair(1, 2, 4, 11), Unsolved(3, 4), TablePair(5, 6, 2, 3)});
    const serving::Dataset dataset(path, serving::Split::kTrain, kShape);

    EXPECT_EQ(dataset.Entries(), 3u);
    EXPECT_EQ(dataset.CaseCount(), 2u);
    EXPECT_EQ(dataset.Sample(1).cases, 2u);

    std::filesystem::remove(path);
}

TEST(ServingTest, PacksTheDocumentedPointLayout) {
    const uint8_t depth = 5;
    const uint16_t size = 37;
    const std::string path = WriteFile("deepcircus_serving_layout.bin", {TablePair(11, 12, depth, size)});
    const serving::Dataset dataset(path, serving::Split::kTrain, kShape);

    const serving::Cases cases = dataset.Sample(1);
    const uint64_t points = uint64_t{kShape.batches} * kShape.points_in_batch;
    EXPECT_EQ(cases.columns, points * serving::PointDim(kBitness));
    EXPECT_EQ(serving::PointDim(kBitness), 3 * kBitness + 2);

    const func::TableFunc g(kBitness, 11);
    const func::TableFunc f(kBitness, 12);
    for (uint64_t point = 0; point < points; ++point) {
        const uint64_t base = point * serving::PointDim(kBitness);
        std::vector<bool> input(kBitness);
        for (uint16_t bit = 0; bit < kBitness; ++bit) input[bit] = RowBit(cases, 0, base + bit);

        // The input bits, then g's value and its single-bit flips, then f's.
        EXPECT_EQ(RowBit(cases, 0, base + kBitness), g(input));
        EXPECT_EQ(RowBit(cases, 0, base + 2 * kBitness + 1), f(input));
        for (uint16_t bit = 0; bit < kBitness; ++bit) {
            std::vector<bool> flipped = input;
            flipped[bit] = !flipped[bit];
            EXPECT_EQ(RowBit(cases, 0, base + kBitness + 1 + bit), g(flipped));
            EXPECT_EQ(RowBit(cases, 0, base + 2 * kBitness + 2 + bit), f(flipped));
        }
    }

    ASSERT_EQ(cases.targets.size(), 2u);
    EXPECT_FLOAT_EQ(cases.targets[0], tools::DepthScore(kBitness, depth));
    EXPECT_FLOAT_EQ(cases.targets[1], tools::SizeScore(kBitness, size));
    std::filesystem::remove(path);
}

TEST(ServingTest, EachEpochResamplesTheSameCases) {
    const std::string path = WriteFile("deepcircus_serving_epochs.bin",
                                       {TablePair(21, 22, 3, 5), TablePair(23, 24, 4, 9), TablePair(25, 26, 6, 21)});
    const serving::Dataset dataset(path, serving::Split::kTrain, kShape);

    const serving::Cases first = dataset.Sample(1);
    const serving::Cases second = dataset.Sample(2);
    // Same cases in the same order, drawn at inputs of the epoch's own.
    EXPECT_EQ(first.cases, second.cases);
    EXPECT_EQ(first.targets, second.targets);
    EXPECT_NE(first.values, second.values);

    // An epoch asked for twice is the same epoch, however many threads sampled
    // it.
    EXPECT_EQ(dataset.Sample(1).values, first.values);
    std::filesystem::remove(path);
}

TEST(ServingTest, SplitsSampleDifferentInputs) {
    const std::string path = WriteFile("deepcircus_serving_splits.bin", {TablePair(31, 32, 3, 5)});
    const serving::Dataset train(path, serving::Split::kTrain, kShape);
    const serving::Dataset validation(path, serving::Split::kValidation, kShape);

    EXPECT_NE(train.Sample(1).values, validation.Sample(1).values);
    EXPECT_EQ(train.Sample(1).targets, validation.Sample(1).targets);
    std::filesystem::remove(path);
}
