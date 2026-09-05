#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "dataset.h"
#include "func/table.h"
#include "func/tree.h"
#include "offline/read_write.h"
#include "tools/score.h"

namespace {

constexpr uint16_t kBitness = 8;
constexpr server::SamplingShape kShape{2, 4, 239};

offline::Entry TablePair(uint64_t g_seed, uint64_t f_seed, uint8_t depth, uint16_t size) {
    return offline::Entry{{offline::FunctionKind::kTable, func::TableFunc(kBitness, g_seed).serialize()},
                          {offline::FunctionKind::kTable, func::TableFunc(kBitness, f_seed).serialize()},
                          depth,
                          size};
}

offline::Entry Unsolved(uint64_t g_seed, uint64_t f_seed) {
    return TablePair(g_seed, f_seed, offline::kUnknownDepth, offline::kUnknownSize);
}

// The shape an `M_2` entry takes above bitness 12: `g` is the witness tree, and
// the second function is the indicator of the subset it is scored on.
offline::Entry TreeOverSubset(const func::TreeFunc& witness, uint64_t subset_seed) {
    return offline::Entry{{offline::FunctionKind::kTree, witness.serialize()},
                          {offline::FunctionKind::kTable, func::TableFunc(kBitness, subset_seed).serialize()},
                          static_cast<uint8_t>(witness.Depth()),
                          static_cast<uint16_t>(witness.Size())};
}

std::string WriteFile(const std::string& name, const std::vector<offline::Entry>& entries) {
    const std::string path = testing::TempDir() + name;
    offline::Writer writer(path, static_cast<uint32_t>(entries.size()), kBitness);
    for (const offline::Entry& entry : entries) writer.Write(entry);
    return path;
}

bool RowBit(const server::Cases& cases, uint32_t row, uint64_t bit) {
    const uint8_t byte = cases.values[row * cases.RowBytes() + bit / 8];
    return ((byte >> (bit % 8)) & 1u) != 0;
}

}  // namespace

TEST(ServingTest, InstallsTargetsForEntriesWithoutOne) {
    const std::string path =
        WriteFile("deepcircus_serving_unsolved.bin", {TablePair(1, 2, 4, 11), Unsolved(3, 4), TablePair(5, 6, 2, 3)});
    server::Dataset dataset(path, server::Split::kTrain, kShape);

    EXPECT_EQ(dataset.Entries(), 3u);
    EXPECT_EQ(dataset.KnownCases(), 2u);
    EXPECT_EQ(dataset.UnknownCases(), 1u);
    dataset.SetUnknownTargets({1.25f, 2.5f});
    const server::Cases cases = dataset.Sample(1);
    EXPECT_EQ(cases.cases, 3u);
    EXPECT_FLOAT_EQ(cases.targets[2], 1.25f);
    EXPECT_FLOAT_EQ(cases.targets[3], 2.5f);

    std::filesystem::remove(path);
}

TEST(ServingTest, SamplesBootstrapReductions) {
    const std::string path = WriteFile("deepcircus_serving_reductions.bin", {Unsolved(3, 4), Unsolved(5, 6)});
    const server::Dataset dataset(path, server::Split::kTrain, kShape);
    const func::TableFunc g(kBitness, 3);
    const func::TableFunc f(kBitness, 4);

    const server::Cases primary = dataset.SamplePrimaryReductions(0, 1);
    const uint16_t child_bitness = kBitness - 1;
    EXPECT_EQ(primary.cases, 2 * kBitness);
    EXPECT_EQ(primary.columns, uint64_t{kShape.batches} * kShape.points_in_batch * server::PointDim(child_bitness));
    for (uint16_t fixed_bit = 0; fixed_bit < kBitness; ++fixed_bit) {
        for (uint16_t fixed_value = 0; fixed_value <= 1; ++fixed_value) {
            const uint32_t row = 2 * fixed_bit + fixed_value;
            std::vector<bool> input(kBitness);
            input[fixed_bit] = fixed_value != 0;
            for (uint16_t free_bit = 0; free_bit < child_bitness; ++free_bit) {
                const uint16_t full_bit = free_bit < fixed_bit ? free_bit : free_bit + 1;
                input[full_bit] = RowBit(primary, row, free_bit);
            }
            EXPECT_EQ(RowBit(primary, row, child_bitness), g(input));
            EXPECT_EQ(RowBit(primary, row, 2 * child_bitness + 1), f(input));
            for (uint16_t free_bit = 0; free_bit < child_bitness; ++free_bit) {
                const uint16_t full_bit = free_bit < fixed_bit ? free_bit : free_bit + 1;
                input[full_bit] = !input[full_bit];
                EXPECT_EQ(RowBit(primary, row, child_bitness + 1 + free_bit), g(input));
                EXPECT_EQ(RowBit(primary, row, 2 * child_bitness + 2 + free_bit), f(input));
                input[full_bit] = !input[full_bit];
            }
        }
    }

    const server::Cases helper = dataset.SampleHelperReductions(0, 1);
    EXPECT_EQ(helper.cases, 2u);
    EXPECT_EQ(helper.columns, uint64_t{kShape.batches} * kShape.points_in_batch * server::PointDim(kBitness));
    for (uint16_t fixed_value = 0; fixed_value <= 1; ++fixed_value) {
        std::vector<bool> input(kBitness);
        for (uint16_t bit = 0; bit < kBitness; ++bit) input[bit] = RowBit(helper, fixed_value, bit);
        EXPECT_EQ(RowBit(helper, fixed_value, kBitness), g(input));
        EXPECT_EQ(RowBit(helper, fixed_value, 2 * kBitness + 1), f(input) == (fixed_value != 0));
    }

    const server::Cases both = dataset.SamplePrimaryReductions(0, 2);
    const server::Cases second = dataset.SamplePrimaryReductions(1, 1);
    const size_t second_offset = size_t{2 * kBitness} * both.RowBytes();
    EXPECT_EQ(std::vector<uint8_t>(both.values.begin() + second_offset, both.values.end()), second.values);

    std::filesystem::remove(path);
}

TEST(ServingTest, PacksTheDocumentedPointLayout) {
    const uint8_t depth = 5;
    const uint16_t size = 37;
    const std::string path = WriteFile("deepcircus_serving_layout.bin", {TablePair(11, 12, depth, size)});
    const server::Dataset dataset(path, server::Split::kTrain, kShape);

    const server::Cases cases = dataset.Sample(1);
    const uint64_t points = uint64_t{kShape.batches} * kShape.points_in_batch;
    EXPECT_EQ(cases.columns, points * server::PointDim(kBitness));
    EXPECT_EQ(server::PointDim(kBitness), 3 * kBitness + 2);

    const func::TableFunc g(kBitness, 11);
    const func::TableFunc f(kBitness, 12);
    for (uint64_t point = 0; point < points; ++point) {
        const uint64_t base = point * server::PointDim(kBitness);
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
    const server::Dataset dataset(path, server::Split::kTrain, kShape);

    const server::Cases first = dataset.Sample(1);
    const server::Cases second = dataset.Sample(2);
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
    const server::Dataset train(path, server::Split::kTrain, kShape);
    const server::Dataset validation(path, server::Split::kValidation, kShape);

    EXPECT_NE(train.Sample(1).values, validation.Sample(1).values);
    EXPECT_EQ(train.Sample(1).targets, validation.Sample(1).targets);
    std::filesystem::remove(path);
}

TEST(ServingTest, ServesTreeBackedFunctions) {
    // `M_2` above bitness 12 stores `g` as a tree rather than a table, and the
    // serving side has to rebuild it from the kind byte alone.
    const func::TreeFunc witness(kBitness, 71);
    const std::string path = WriteFile("deepcircus_serving_tree.bin", {TreeOverSubset(witness, 72)});
    const server::Dataset dataset(path, server::Split::kTrain, kShape);

    EXPECT_EQ(dataset.Entries(), 1u);
    EXPECT_EQ(dataset.KnownCases(), 1u);

    const server::Cases cases = dataset.Sample(1);
    ASSERT_EQ(cases.cases, 1u);
    const func::TableFunc subset(kBitness, 72);
    const uint64_t points = uint64_t{kShape.batches} * kShape.points_in_batch;
    for (uint64_t point = 0; point < points; ++point) {
        const uint64_t base = point * server::PointDim(kBitness);
        std::vector<bool> input(kBitness);
        for (uint16_t bit = 0; bit < kBitness; ++bit) input[bit] = RowBit(cases, 0, base + bit);

        EXPECT_EQ(RowBit(cases, 0, base + kBitness), witness(input));
        EXPECT_EQ(RowBit(cases, 0, base + 2 * kBitness + 1), subset(input));
    }

    EXPECT_FLOAT_EQ(cases.targets[0], tools::DepthScore(kBitness, witness.Depth()));
    EXPECT_FLOAT_EQ(cases.targets[1], tools::SizeScore(kBitness, witness.Size()));
    std::filesystem::remove(path);
}

TEST(ServingTest, NamesTheFileOfEachModel) {
    EXPECT_EQ(server::FilePath("data", server::Model::kM1, 8, server::Split::kTrain), "data/m1_08.train");
    EXPECT_EQ(server::FilePath("data", server::Model::kM2, 8, server::Split::kValidation), "data/m2_08.val");
    EXPECT_EQ(server::FilePath("data", server::Model::kM2, 13, server::Split::kTrain), "data/m2_13.train");
}
