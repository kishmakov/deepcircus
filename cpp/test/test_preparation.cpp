#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "func/table.h"
#include "func/tree.h"
#include "func/tt.h"
#include "sampler.h"
#include "tools/solver.h"

namespace {

constexpr prep::Parameters kParameters{239};

void ExpectSameEntry(const offline::Entry& first, const offline::Entry& second) {
    EXPECT_EQ(first.g.kind, second.g.kind);
    EXPECT_EQ(first.g.payload, second.g.payload);
    EXPECT_EQ(first.f.kind, second.f.kind);
    EXPECT_EQ(first.f.payload, second.f.payload);
    EXPECT_EQ(first.min_depth, second.min_depth);
    EXPECT_EQ(first.min_size, second.min_size);
}

void ExpectExactTargets(const offline::Entry& entry, prep::Model model, uint16_t bitness, const func::Func& target) {
    const auto g = target.TruthTable();
    const auto f = func::TableFunc(bitness, entry.f.payload).TruthTable();
    ASSERT_TRUE(entry.TargetKnown());
    if (model == prep::Model::kM1) {
        EXPECT_EQ(entry.min_depth, tools::SolveForDepthGiven(bitness, g, f));
        EXPECT_EQ(entry.min_size, tools::SolveForSizeGiven(bitness, g, f));
    } else {
        EXPECT_EQ(entry.min_depth, tools::SolveForDepthRestricted(bitness, g, f));
        EXPECT_EQ(entry.min_size, tools::SolveForSizeRestricted(bitness, g, f));
    }
}

}  // namespace

TEST(PreparationTest, TTUsesMatchingTableAtEveryBitness) {
    for (const auto model : {prep::Model::kM1, prep::Model::kM2}) {
        for (const uint16_t bitness : {8, 12, 13, 255}) {
            const auto entry = prep::TTEntry(kParameters, model, bitness, 0);
            ASSERT_EQ(entry.g.kind, offline::FunctionKind::kTreeOverTable);
            ASSERT_EQ(entry.f.kind, offline::FunctionKind::kTable);
            ASSERT_TRUE(entry.TargetKnown());
            uint32_t table_bytes = 0;
            ASSERT_GE(entry.g.payload.size(), sizeof(table_bytes));
            std::memcpy(&table_bytes, entry.g.payload.data(), sizeof(table_bytes));
            const auto table_begin = entry.g.payload.begin() + sizeof(table_bytes);
            ASSERT_EQ(table_bytes, entry.f.payload.size());
            ASSERT_GE(entry.g.payload.size(), sizeof(table_bytes) + table_bytes);
            EXPECT_EQ(std::vector<uint8_t>(table_begin, table_begin + table_bytes), entry.f.payload);
            const func::TreeFunc witness(bitness + 1,
                                         std::vector<uint8_t>(table_begin + table_bytes, entry.g.payload.end()));
            if (bitness <= tools::kMaxSolvableBitness) {
                ExpectExactTargets(entry, model, bitness, func::TTFunc(bitness, entry.g.payload));
                EXPECT_LE(entry.min_depth, witness.Depth());
                EXPECT_LE(entry.min_size, witness.Size());
            } else {
                EXPECT_EQ(entry.min_depth, witness.Depth());
                EXPECT_EQ(entry.min_size, witness.Size());
            }
            ExpectSameEntry(entry, prep::TTEntry(kParameters, model, bitness, 0));
        }
    }
}

TEST(PreparationTest, GeneralUsesExactTargetsThroughTwelveAndMarkersAbove) {
    for (const auto model : {prep::Model::kM1, prep::Model::kM2}) {
        for (const uint16_t bitness : {8, 12, 13, 255}) {
            const auto entry = prep::GeneralEntry(kParameters, model, bitness, 0);
            ASSERT_EQ(entry.g.kind, offline::FunctionKind::kTable);
            ASSERT_EQ(entry.f.kind, offline::FunctionKind::kTable);
            EXPECT_NE(entry.g.payload, entry.f.payload);
            if (bitness <= tools::kMaxSolvableBitness) {
                ExpectExactTargets(entry, model, bitness, func::TableFunc(bitness, entry.g.payload));
            } else {
                EXPECT_FALSE(entry.TargetKnown());
                EXPECT_EQ(entry.min_depth, offline::kUnknownDepth);
                EXPECT_EQ(entry.min_size, offline::kUnknownSize);
            }
            ExpectSameEntry(entry, prep::GeneralEntry(kParameters, model, bitness, 0));
        }
    }
}
