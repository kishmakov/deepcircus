#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <set>
#include <string>
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

TEST(SampleGraphTest, ReachesMinimumWithCompleteOrders) {
    const auto initial = tools::BuildGraph(5, 239, 13);
    CheckGraph(initial);
    EXPECT_EQ(initial.cells.size(), 73);
    EXPECT_EQ(initial.layers[2].size(), 60);

    const auto first = tools::BuildGraph(5, 239, 74);
    const auto more = tools::BuildGraph(5, 239, 649);
    const auto full = tools::BuildGraph(5, 239, 729);
    CheckGraph(first);
    CheckGraph(more);
    CheckGraph(full);
    EXPECT_GT(first.cells.size(), 74);
    EXPECT_EQ(first.layers.back().size(), 2);  // One assignment, with both final branches.
    // The first estimated pass falls short; subsequent orders must fill the gap.
    EXPECT_LT(first.cells.size(), 649);
    EXPECT_GE(more.cells.size(), 649);
    EXPECT_EQ(full.cells.size(), 729);
    for (const auto& cell : first.cells) {
        EXPECT_NE(std::find(more.cells.begin(), more.cells.end(), cell), more.cells.end());
    }
    EXPECT_EQ(more.cells, tools::BuildGraph(5, 239, 649).cells);
    EXPECT_EQ(full.layers.back().size(), 64);
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

TEST(SampleGraphTest, SamplingCoversAllCellsWithoutChangingTheGraph) {
    for (uint16_t bitness : {3, 13}) {
        const auto graph = tools::BuildGraph(bitness, 239, 4 * (bitness + 1) + 1);
        ASSERT_EQ(graph.cells.size(), 1 + 2 * (bitness + 1) * (bitness + 1));
        const auto cells = graph.cells;
        const uint32_t points = static_cast<uint32_t>(cells.size());
        const std::vector<tools::BitsSet> inputs = tools::SampleGraphInputs(graph, 42, points);
        EXPECT_EQ(inputs, tools::SampleGraphInputs(graph, 42, points));
        EXPECT_EQ(graph.cells, cells);
        ASSERT_EQ(inputs.size(), points);
        for (const auto& input : inputs) EXPECT_TRUE((input >> bitness).none());
        for (const auto& cell : cells) {
            auto fixed = cell.fixed;
            fixed.reset(bitness);
            bool covered = false;
            for (const auto& input : inputs) covered |= (input & fixed) == (cell.values & fixed);
            EXPECT_TRUE(covered);
        }
        EXPECT_NE(inputs, tools::SampleGraphInputs(graph, 43, points));
    }
}

TEST(SampleGraphTest, SamplingCoversFullGraphWithExactlyEnoughPoints) {
    const auto graph = tools::BuildGraph(2, 239, 27);
    const std::vector<tools::BitsSet> inputs = tools::SampleGraphInputs(graph, 42, 4);
    ASSERT_EQ(inputs.size(), 4);
    std::set<unsigned long> values;
    for (const auto& input : inputs) values.insert(input.to_ulong());
    EXPECT_EQ(values, (std::set<unsigned long>{0, 1, 2, 3}));
}

TEST(SampleGraphTest, RejectsInvalidShapesAndInsufficientSamples) {
    EXPECT_DEATH(tools::BuildGraph(0, 239, 1), "bitness");
    EXPECT_DEATH(tools::BuildGraph(256, 239, 1), "bitness");
    EXPECT_DEATH(tools::BuildGraph(8, 239, 18), "cells_number");
    EXPECT_DEATH(tools::BuildGraph(1, 239, 10), "cells_number");
    EXPECT_DEATH(tools::BuildGraph(8, 239, UINT32_MAX), "cells_number");
    const auto graph = tools::BuildGraph(1, 239, 9);
    EXPECT_DEATH(tools::SampleGraphInputs(graph, 239, 0), "points");
    EXPECT_DEATH(tools::SampleGraphInputs(graph, 239, 1), "points");
}
