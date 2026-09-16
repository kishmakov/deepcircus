#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "dense_graph.h"
#include "graph.h"
#include "sample.h"
#include "tools/random.h"

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

namespace {

void CheckGraph(const tools::Graph& graph) {
    ASSERT_EQ(graph.layers.size(), graph.bitness + 2);
    ASSERT_EQ(graph.layers[0].size(), 1);
    ASSERT_EQ(graph.layers[0][0].get(), graph.root);
    ASSERT_EQ(graph.layers[1].size(), 2 * (graph.bitness + 1));
    ASSERT_EQ(graph.root->questions.size(), graph.bitness + 1);
    const tools::DenseGraph dense(graph);
    EXPECT_EQ(dense.bitness, graph.bitness);
    EXPECT_EQ(dense.cells, graph.cells);
    std::set<std::string> restrictions;
    std::set<uint32_t> reached{graph.root->cell_id};
    size_t count = 0;
    for (size_t depth = 0; depth < graph.layers.size(); ++depth) {
        const auto& layer = dense.layers[depth];
        ASSERT_EQ(layer.cell.size(), graph.layers[depth].size() + 1);
        ASSERT_EQ(layer.offsets.size(), layer.cell.size() + 1);
        ASSERT_EQ(layer.offsets[0], 0);
        ASSERT_EQ(layer.offsets[1], 0);
        ASSERT_EQ(layer.cell[0], UINT32_MAX);
        ASSERT_EQ(layer.zero.size(), layer.one.size());
        ASSERT_EQ(layer.offsets.back(), layer.zero.size());
        for (size_t index = 0; index < graph.layers[depth].size(); ++index) {
            const auto& node = graph.layers[depth][index];
            ASSERT_LT(node->cell_id, graph.cells.size());
            ASSERT_TRUE(reached.contains(node->cell_id));
            const auto& cell = graph.cells[node->cell_id];
            EXPECT_EQ(cell.fixed.count(), depth);
            EXPECT_TRUE((cell.values & ~cell.fixed).none());
            EXPECT_TRUE(restrictions.insert(cell.fixed.to_string() + cell.values.to_string()).second);
            EXPECT_EQ(layer.cell[index + 1], node->cell_id);
            EXPECT_EQ(layer.offsets[index + 2] - layer.offsets[index + 1], node->questions.size());
            std::set<uint16_t> axes;
            for (size_t question_id = 0; question_id < node->questions.size(); ++question_id) {
                const auto& question = node->questions[question_id];
                ASSERT_LE(question.axis, graph.bitness);
                ASSERT_FALSE(cell.fixed[question.axis]);
                ASSERT_TRUE(axes.insert(question.axis).second);
                const size_t offset = layer.offsets[index + 1] + question_id;
                for (unsigned answer = 0; answer < 2; ++answer) {
                    const auto* child = answer ? question.one : question.zero;
                    ASSERT_NE(child, nullptr);
                    ASSERT_LT(depth + 1, graph.layers.size());
                    auto expected = cell;
                    expected.fixed.set(question.axis);
                    expected.values.set(question.axis, answer);
                    EXPECT_EQ(graph.cells[child->cell_id], expected);
                    reached.insert(child->cell_id);
                    const uint32_t child_index = answer ? layer.one[offset] : layer.zero[offset];
                    ASSERT_GT(child_index, 0);
                    ASSERT_LT(child_index, dense.layers[depth + 1].cell.size());
                    EXPECT_EQ(dense.layers[depth + 1].cell[child_index], child->cell_id);
                }
            }
            ++count;
        }
    }
    EXPECT_EQ(count, graph.cells.size());
    EXPECT_EQ(reached.size(), count);
}

}  // namespace

TEST(SampleGraphTest, StartsWithEveryOneAndTwoAxisRestriction) {
    const auto graph = tools::BuildGraph(8, 239, 19);
    CheckGraph(graph);
    EXPECT_EQ(graph.cells.size(), 163);
    EXPECT_EQ(graph.layers[2].size(), 144);
    for (uint16_t axis = 0; axis <= graph.bitness; ++axis) {
        const auto& question = graph.root->questions[axis];
        EXPECT_EQ(question.axis, axis);
        EXPECT_FALSE(graph.cells[question.zero->cell_id].values[axis]);
        EXPECT_TRUE(graph.cells[question.one->cell_id].values[axis]);
    }
}

TEST(SampleGraphTest, TemplatesIncludeUpToTwoHoles) {
    constexpr uint16_t bitness = 5;
    constexpr uint64_t seed = 239;
    constexpr uint32_t cells_number = 600;
    tools::Random random(tools::DomainSeed(seed, 0x67726170685f6f72ull, bitness));
    std::vector<uint16_t> order(bitness + 1);
    std::iota(order.begin(), order.end(), 0);
    for (size_t remaining = order.size(); remaining > 1; --remaining) {
        std::swap(order[remaining - 1], order[random.Below(remaining)]);
    }
    const auto graph = tools::BuildGraph(bitness, seed, cells_number);
    CheckGraph(graph);
    EXPECT_GE(graph.cells.size(), cells_number);
    EXPECT_LT(graph.cells.size(), cells_number + 2 * (bitness + 1));
    std::set<size_t> hole_counts;
    for (const auto& cell : graph.cells) {
        if (cell.fixed.count() <= 2) continue;
        size_t end = order.size();
        while (!cell.fixed[order[end - 1]]) --end;
        const size_t holes = end - cell.fixed.count();
        EXPECT_LE(holes, 2);
        hole_counts.insert(holes);
    }
    EXPECT_EQ(hole_counts, (std::set<size_t>{0, 1, 2}));
}

TEST(SampleGraphTest, RepeatsOrdersUntilBudgetAndSharesCells) {
    // One order with up to two holes cannot cover all masks on six axes.
    const auto graph = tools::BuildGraph(5, 239, 729);
    CheckGraph(graph);
    EXPECT_EQ(graph.cells.size(), 729);
    const auto full = tools::BuildGraph(2, 239, 27);
    CheckGraph(full);
    EXPECT_EQ(full.cells.size(), 27);
    EXPECT_EQ(full.layers.back().size(), 8);
}

TEST(SampleGraphTest, AssemblyIncludesEveryAvailableSplit) {
    const auto graph = tools::BuildGraph(2, 239, 27);
    CheckGraph(graph);
    for (size_t depth = 0; depth < graph.layers.size(); ++depth) {
        for (const auto& node : graph.layers[depth]) {
            EXPECT_EQ(node->questions.size(), graph.bitness + 1 - depth);
            for (size_t index = 1; index < node->questions.size(); ++index) {
                EXPECT_LT(node->questions[index - 1].axis, node->questions[index].axis);
            }
        }
    }
}

TEST(SampleGraphTest, Deterministic) {
    const auto graph = tools::BuildGraph(8, 239, 1200);
    CheckGraph(graph);
    const tools::DenseGraph dense(graph);
    const tools::DenseGraph repeated(tools::BuildGraph(8, 239, 1200));
    EXPECT_EQ(dense.cells, repeated.cells);
    for (size_t depth = 0; depth < dense.layers.size(); ++depth) {
        EXPECT_EQ(dense.layers[depth].offsets, repeated.layers[depth].offsets);
        EXPECT_EQ(dense.layers[depth].zero, repeated.layers[depth].zero);
        EXPECT_EQ(dense.layers[depth].one, repeated.layers[depth].one);
        EXPECT_EQ(dense.layers[depth].cell, repeated.layers[depth].cell);
    }
    EXPECT_NE(graph.cells, tools::BuildGraph(8, 240, 1200).cells);
    EXPECT_GE(graph.cells.size(), 1200);
    EXPECT_LT(graph.cells.size(), 1218);
}

TEST(SampleGraphTest, Supports255Bits) {
    const auto graph = tools::BuildGraph(255, 239, 131073);
    EXPECT_EQ(graph.cells.size(), 131073);
    EXPECT_EQ(graph.layers.size(), 257);
    EXPECT_EQ(graph.layers[1].size(), 512);
    EXPECT_EQ(graph.layers[2].size(), 130560);
    ASSERT_EQ(graph.root->questions.size(), 256);
    const auto& helper = graph.root->questions.back();
    EXPECT_EQ(helper.axis, 255);
    EXPECT_TRUE(graph.cells[helper.one->cell_id].values[255]);

    std::vector<tools::Graph::Cell> cells(3);
    cells[1].fixed.set(254);
    cells[2] = cells[1];
    cells[2].values.set(254);
    const tools::Graph sparse(255, std::move(cells));
    const auto inputs = tools::SampleGraphInputs(sparse, 42, 8);
    ASSERT_EQ(inputs.size(), 8 * 32);
    std::set<unsigned> high_bits;
    for (size_t row = 0; row < 8; ++row) {
        const uint8_t last = inputs[row * 32 + 31];
        EXPECT_EQ(last >> 7, 0);
        high_bits.insert((last >> 6) & 1);
    }
    EXPECT_EQ(high_bits, (std::set<unsigned>{0, 1}));
}

TEST(SampleGraphTest, SamplingCoversAllCellsWithoutChangingTheGraph) {
    for (uint16_t bitness : {3, 13}) {
        const auto graph = tools::BuildGraph(bitness, 239, 4 * (bitness + 1) + 1);
        const auto cells = graph.cells;
        const uint32_t points = static_cast<uint32_t>(cells.size());
        const auto inputs = tools::SampleGraphInputs(graph, 42, points);
        EXPECT_EQ(inputs, tools::SampleGraphInputs(graph, 42, points));
        EXPECT_EQ(graph.cells, cells);
        const size_t row_bytes = (bitness + 7) / 8;
        ASSERT_EQ(inputs.size(), points * row_bytes);
        std::vector<tools::Graph::Bits> rows(points);
        for (size_t row = 0; row < points; ++row) {
            for (uint16_t axis = 0; axis < bitness; ++axis) {
                rows[row][axis] = (inputs[row * row_bytes + axis / 8] >> (axis % 8)) & 1;
            }
            if (bitness % 8) EXPECT_EQ(inputs[(row + 1) * row_bytes - 1] >> (bitness % 8), 0);
        }
        for (const auto& cell : cells) {
            auto fixed = cell.fixed;
            fixed.reset(bitness);
            bool covered = false;
            for (const auto& row : rows) covered |= (row & fixed) == (cell.values & fixed);
            EXPECT_TRUE(covered);
        }
        EXPECT_NE(inputs, tools::SampleGraphInputs(graph, 43, points));
    }
}

TEST(SampleGraphTest, RejectsInvalidShapesAndInsufficientSamples) {
    EXPECT_DEATH(tools::BuildGraph(0, 239, 9), "bitness");
    EXPECT_DEATH(tools::BuildGraph(256, 239, 1000), "bitness");
    EXPECT_DEATH(tools::BuildGraph(8, 239, 18), "cells_number");
    EXPECT_DEATH(tools::BuildGraph(1, 239, 10), "cells_number");
    EXPECT_DEATH(tools::BuildGraph(8, 239, UINT32_MAX), "cells_number");
    const auto graph = tools::BuildGraph(1, 239, 9);
    EXPECT_DEATH(tools::SampleGraphInputs(graph, 239, 0), "points");
    EXPECT_DEATH(tools::SampleGraphInputs(graph, 239, 1), "points");
}
