#include "func/tree.h"

#include <assert.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include "tools/binary_tree.h"
#include "tools/random.h"

namespace func {

namespace {

using tools::BinaryTree;

// Marks a serialized word as a leaf. Bit ids stay below kMaxTreeBitness, so
// the tag can never be mistaken for one.
constexpr uint32_t kLeafTag = uint32_t{1} << 31;

// Internal nodes a full binary tree of depth at most `max_depth` can hold.
uint32_t Capacity(uint32_t max_depth) {
    assert(max_depth < BinaryTree::kMaxDepth);
    return (uint32_t{1} << max_depth) - 1;
}

BinaryTree SampleTree(uint16_t bitness, uint64_t seed) {
    tools::Random random(seed);

    const uint32_t depth_bound = std::min<uint32_t>(bitness, kMaxTreeDepth);
    const uint32_t max_depth = 1 + static_cast<uint32_t>(random.Below(depth_bound));
    const uint32_t size_bound = std::min(Capacity(max_depth), kMaxTreeSize);
    const uint32_t size = static_cast<uint32_t>(random.Below(size_bound + 1));

    // The tree draws on its own stream, so the choices above stay clear of it.
    return BinaryTree::Sample(random.Next(), max_depth, size, bitness);
}

// Reads the subtree rooted at `node`, which is a leaf `Split` has yet to grow,
// off the pre-order words from `position` on.
void Deserialize(BinaryTree& tree, const std::vector<uint32_t>& words, size_t& position, uint32_t node) {
    assert(position < words.size());
    const uint32_t word = words[position];
    ++position;

    if ((word & kLeafTag) != 0) {
        tree.Value(node) = word & ~kLeafTag;
        return;
    }

    const std::array<uint32_t, 2> children = tree.Split(node);
    tree.Value(node) = word;
    Deserialize(tree, words, position, children[BinaryTree::kLeft]);
    Deserialize(tree, words, position, children[BinaryTree::kRight]);
}

BinaryTree DeserializeTree(const std::vector<uint8_t>& bytes) {
    assert(bytes.size() >= sizeof(uint32_t));
    assert(bytes.size() % sizeof(uint32_t) == 0);

    std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
    memcpy(words.data(), bytes.data(), bytes.size());

    const uint32_t nodes = words[0];
    // A binary tree has an odd node count, and every node but the root's is a
    // child of an internal one.
    assert(nodes % 2 == 1);
    assert(words.size() == size_t{nodes} + 1);

    BinaryTree tree;
    size_t position = 1;
    Deserialize(tree, words, position, 0);
    assert(position == words.size());
    assert(2 * tree.InternalNodes() + 1 == nodes);
    return tree;
}

// Appends the subtree rooted at `node` in the order `Deserialize` reads it.
void Serialize(const BinaryTree& tree, uint32_t node, std::vector<uint32_t>& words) {
    if (tree[node].IsLeaf()) {
        words.push_back(kLeafTag | tree[node].value);
        return;
    }
    words.push_back(tree[node].value);
    Serialize(tree, tree[node].child[BinaryTree::kLeft], words);
    Serialize(tree, tree[node].child[BinaryTree::kRight], words);
}

// Checks what makes the tree readable as a function: an internal node tests a
// bit the input has, and a leaf holds one bit of answer.
void AssertReadable(const BinaryTree& tree, uint16_t bitness) {
    const uint32_t nodes = 2 * tree.InternalNodes() + 1;
    for (uint32_t node = 0; node < nodes; ++node) {
        assert(tree[node].value < (tree[node].IsLeaf() ? 2u : uint32_t{bitness}));
    }
    assert(tree.Depth() <= bitness);
}

}  // namespace

TreeFunc::TreeFunc(uint16_t bitness, uint64_t seed) : TreeFunc(bitness, SampleTree(bitness, seed)) {}

TreeFunc::TreeFunc(uint16_t bitness, tools::BinaryTree tree) : func::Func(bitness), tree_(std::move(tree)) {
    assert(bitness_ >= kMinTreeBitness && bitness_ <= kMaxTreeBitness);
    AssertReadable(tree_, bitness_);
}

TreeFunc::TreeFunc(uint16_t bitness, const std::vector<uint8_t>& bytes) : TreeFunc(bitness, DeserializeTree(bytes)) {}

bool TreeFunc::operator()(const FuncInput& input) const {
    assert(input.size() == bitness_);

    uint32_t node = 0;
    while (!tree_[node].IsLeaf()) {
        const bool bit = input[tree_[node].value];
        node = tree_[node].child[bit ? BinaryTree::kRight : BinaryTree::kLeft];
    }
    return tree_[node].value != 0;
}

std::vector<uint8_t> TreeFunc::serialize() const {
    const uint32_t nodes = 2 * tree_.InternalNodes() + 1;

    std::vector<uint32_t> words;
    words.reserve(size_t{nodes} + 1);
    words.push_back(nodes);
    Serialize(tree_, 0, words);
    assert(words.size() == size_t{nodes} + 1);

    std::vector<uint8_t> bytes(words.size() * sizeof(uint32_t));
    memcpy(bytes.data(), words.data(), bytes.size());
    return bytes;
}

}  // namespace func
