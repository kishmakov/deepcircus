#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "tools/binary_tree.h"

TEST(BinaryTreeTest, SplitsLeaves) {
    tools::BinaryTree tree;
    const auto [left, right] = tree.Split(0);
    tree.Split(right);
    tree.Value(0) = 20;
    tree.Value(left) = 31;

    EXPECT_EQ(tree.InternalNodes(), 2u);
    EXPECT_EQ(tree.Depth(), 2u);
    EXPECT_EQ(tree[0].value, 20u);
    EXPECT_EQ(tree[0].child[tools::BinaryTree::kLeft], left);
    EXPECT_EQ(tree[0].child[tools::BinaryTree::kRight], right);
    EXPECT_EQ(tree[left].value, 31u);
    EXPECT_EQ(tree[right].value, 0u);
    EXPECT_TRUE(tree[left].IsLeaf());
    EXPECT_FALSE(tree[right].IsLeaf());
}

namespace {

// Recomputes the invariants `BinaryTree` caches, so the fields can be checked
// against a walk of the nodes.
struct Walk {
    uint32_t internal_nodes = 0;
    uint32_t depth = 0;

    void Run(const tools::BinaryTree& tree, uint32_t id, uint32_t level) {
        depth = std::max(depth, level);
        EXPECT_EQ(tree[id].depth, level);
        if (tree[id].IsLeaf()) return;
        ++internal_nodes;
        Run(tree, tree[id].child[tools::BinaryTree::kLeft], level + 1);
        Run(tree, tree[id].child[tools::BinaryTree::kRight], level + 1);
    }
};

Walk Check(const tools::BinaryTree& tree) {
    Walk walk;
    walk.Run(tree, 0, 0);
    EXPECT_EQ(walk.internal_nodes, tree.InternalNodes());
    EXPECT_EQ(walk.depth, tree.Depth());
    return walk;
}

}  // namespace

TEST(BinaryTreeTest, SamplesSingleNodeForZeroSize) {
    const tools::BinaryTree tree = tools::BinaryTree::Sample(1, 4, 0, 4u);

    EXPECT_EQ(tree.InternalNodes(), 0u);
    EXPECT_EQ(tree.Depth(), 0u);
    EXPECT_TRUE(tree[0].IsLeaf());
}

TEST(BinaryTreeTest, SamplesRequestedSize) {
    for (uint32_t max_depth = 1; max_depth <= 6; ++max_depth) {
        const uint32_t capacity = (uint32_t{1} << max_depth) - 1;
        for (uint32_t size = 0; size <= capacity; ++size) {
            const tools::BinaryTree tree =
                tools::BinaryTree::Sample(uint64_t{size} * 31 + max_depth, max_depth, size, max_depth);
            const Walk walk = Check(tree);
            EXPECT_EQ(walk.internal_nodes, size);
            EXPECT_LE(walk.depth, max_depth);
        }
    }
}

TEST(BinaryTreeTest, SamplingIsDeterministic) {
    const tools::BinaryTree one = tools::BinaryTree::Sample(7, 8, 20, 8u);
    const tools::BinaryTree two = tools::BinaryTree::Sample(7, 8, 20, 8u);
    const tools::BinaryTree other = tools::BinaryTree::Sample(8, 8, 20, 8u);

    ASSERT_EQ(one.InternalNodes(), two.InternalNodes());
    ASSERT_EQ(one.InternalNodes(), other.InternalNodes());
    EXPECT_EQ(one.Depth(), two.Depth());

    bool differs = false;
    for (uint32_t id = 0; id < 2 * one.InternalNodes() + 1; ++id) {
        EXPECT_EQ(one[id].child[tools::BinaryTree::kLeft], two[id].child[tools::BinaryTree::kLeft]);
        differs = differs || one[id].child[tools::BinaryTree::kLeft] != other[id].child[tools::BinaryTree::kLeft];
    }
    EXPECT_TRUE(differs);
}

namespace {

// Walks every root-to-leaf path, checking that no id repeats along it and that
// each one came out of the requested id range.
void CheckPath(const tools::BinaryTree& tree, uint32_t node, uint32_t ids_num,
               std::vector<uint32_t>& path) {
    if (tree[node].IsLeaf()) {
        EXPECT_LE(tree[node].value, 1u) << "a leaf holds an output, not an id";
        return;
    }

    const uint32_t id = tree[node].value;
    EXPECT_LT(id, ids_num);
    EXPECT_EQ(std::find(path.begin(), path.end(), id), path.end()) << "id " << id << " repeats along a path";

    path.push_back(id);
    CheckPath(tree, tree[node].child[tools::BinaryTree::kLeft], ids_num, path);
    CheckPath(tree, tree[node].child[tools::BinaryTree::kRight], ids_num, path);
    path.pop_back();
}

}  // namespace

TEST(BinaryTreeTest, SamplesIdsUnrepeatedAlongPaths) {
    for (uint32_t max_depth = 1; max_depth <= 6; ++max_depth) {
        const uint32_t capacity = (uint32_t{1} << max_depth) - 1;
        for (uint32_t size = 0; size <= capacity; ++size) {
            for (uint64_t seed = 0; seed < 4; ++seed) {
                const uint32_t ids_num = max_depth + 2;
                const tools::BinaryTree tree =
                    tools::BinaryTree::Sample(seed * 17 + size, max_depth, size, ids_num);
                std::vector<uint32_t> path;
                CheckPath(tree, 0, ids_num, path);
            }
        }
    }
}

TEST(BinaryTreeTest, SamplesSiblingLeavesApart) {
    const tools::BinaryTree tree = tools::BinaryTree::Sample(3, 6, 20, 6u);
    const uint32_t nodes = 2 * tree.InternalNodes() + 1;

    for (uint32_t node = 0; node < nodes; ++node) {
        if (tree[node].IsLeaf()) continue;
        const uint32_t zero = tree[node].child[tools::BinaryTree::kLeft];
        const uint32_t one = tree[node].child[tools::BinaryTree::kRight];
        if (!tree[zero].IsLeaf() || !tree[one].IsLeaf()) continue;
        EXPECT_NE(tree[zero].value, tree[one].value) << "node " << node << " has two leaves that agree";
    }
}
