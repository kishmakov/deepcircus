#include "tools/binary_tree.h"

#include <assert.h>

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "tools/random.h"

namespace tools {
namespace {

// Internal nodes a full binary tree of depth at most `max_depth` can hold.
uint32_t Capacity(uint32_t max_depth) {
    assert(max_depth < BinaryTree::kMaxDepth);
    return (uint32_t{1} << max_depth) - 1;
}

// Grows the leaf `node` into a subtree of exactly `size` internal nodes over
// `max_depth` further levels, dealing `size - 1` of them to the two sides.
void Grow(BinaryTree& tree, Random& random, uint32_t node, uint32_t max_depth, uint32_t size) {
    // the invariant every call is entered with
    assert(size <= Capacity(max_depth));
    if (size == 0) return;  // nothing to do

    const uint32_t capacity = Capacity(max_depth - 1);
    const uint32_t rest = size - 1;

    // What one side cannot hold, the other one has to.
    const uint32_t max_subtree = std::min(rest, capacity);
    const uint32_t min_subtree = rest - max_subtree;
    const uint64_t var_subtree = random.Below(max_subtree - min_subtree + 1);

    const uint32_t left_size = min_subtree + static_cast<uint32_t>(var_subtree);
    const uint32_t right_size = rest - left_size;

    const std::array<uint32_t, 2> children = tree.Split(node);
    Grow(tree, random, children[BinaryTree::kLeft], max_depth - 1, left_size);
    Grow(tree, random, children[BinaryTree::kRight], max_depth - 1, right_size);
}

// Assigns each internal node an id unused earlier on its path.
void AssignIds(BinaryTree& tree, Random& random, uint32_t node, std::vector<uint32_t> available) {
    if (tree[node].IsLeaf()) return;
    // A path the ids cannot cover has nothing left to ask.
    assert(!available.empty());

    const uint32_t id = available[random.Below(available.size())];
    tree.Value(node) = id;

    // Every copy of it goes, so the subtrees below cannot draw it again.
    std::vector<uint32_t> rest;
    rest.reserve(available.size());
    for (uint32_t other : available) {
        if (other != id) rest.push_back(other);
    }
    AssignIds(tree, random, tree[node].child[BinaryTree::kLeft], rest);
    AssignIds(tree, random, tree[node].child[BinaryTree::kRight], std::move(rest));
}

// Randomizes leaves and makes sibling leaves distinct.
void AssignLeaves(BinaryTree& tree, Random& random) {
    const uint32_t nodes = 2 * tree.InternalNodes() + 1;
    for (uint32_t node = 0; node < nodes; ++node) {
        if (tree[node].IsLeaf()) tree.Value(node) = random.Bool();
    }
    for (uint32_t node = 0; node < nodes; ++node) {
        if (tree[node].IsLeaf()) continue;
        const uint32_t zero = tree[node].child[BinaryTree::kLeft];
        const uint32_t one = tree[node].child[BinaryTree::kRight];
        if (tree[zero].IsLeaf() && tree[one].IsLeaf() && tree[zero].value == tree[one].value) {
            tree.Value(one) = 1 - tree[zero].value;
        }
    }
}

}  // namespace

BinaryTree BinaryTree::Sample(uint64_t seed, uint32_t max_depth, uint32_t size, uint32_t ids_num) {
    assert(max_depth > 0);
    assert(size <= Capacity(max_depth));
    assert(ids_num >= max_depth);

    std::vector<uint32_t> ids(ids_num);
    std::iota(ids.begin(), ids.end(), uint32_t{0});

    Random random(seed);
    BinaryTree tree;
    Grow(tree, random, 0, max_depth, size);
    AssignIds(tree, random, 0, ids);
    AssignLeaves(tree, random);
    return tree;
}

std::array<uint32_t, 2> BinaryTree::Split(uint32_t leaf) {
    assert(leaf < nodes_.size());
    assert(nodes_[leaf].IsLeaf());
    // Two more nodes have to stay addressable by a uint32_t id.
    assert(nodes_.size() + 2 <= UINT32_MAX);

    const uint32_t depth = nodes_[leaf].depth + 1;
    const uint32_t left = static_cast<uint32_t>(nodes_.size());
    nodes_.push_back(Node{0, depth});
    nodes_.push_back(Node{0, depth});
    nodes_[leaf].child[kLeft] = left;
    nodes_[leaf].child[kRight] = left + 1;

    ++internal_nodes_;
    depth_ = std::max(depth_, depth);
    return {left, left + 1};
}

uint32_t& BinaryTree::Value(uint32_t id) {
    assert(id < nodes_.size());
    return nodes_[id].value;
}

const BinaryTree::Node& BinaryTree::operator[](uint32_t id) const {
    assert(id < nodes_.size());
    return nodes_[id];
}

}  // namespace tools
