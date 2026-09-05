#pragma once

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <vector>

namespace tools {

class BinaryTree {
public:
    static constexpr size_t kLeft = 0;
    static constexpr size_t kRight = 1;
    static constexpr uint32_t kMaxDepth = 32;

    // Random binary trees, deterministic in `seed`: `size` internal nodes over
    // at most `max_depth` levels, each holding one of `ids` as its value.
    static BinaryTree Sample(uint64_t seed, uint32_t max_depth, uint32_t size, uint32_t ids_num);

    struct Node {
        uint32_t value;
        uint32_t depth = 0;
        uint32_t child[2] = {};

        bool IsLeaf() const { return child[kLeft] == 0; }
    };

    BinaryTree() = default;

    // Turns `leaf` into an internal node over two fresh leaves.
    std::array<uint32_t, 2> Split(uint32_t leaf);

    uint32_t& Value(uint32_t id);
    const Node& operator[](uint32_t id) const;

    uint32_t InternalNodes() const { return internal_nodes_; }
    uint32_t Depth() const { return depth_; }

private:
    // Starts as the single leaf that is the root.
    std::vector<Node> nodes_{Node{}};
    uint32_t internal_nodes_ = 0;
    uint32_t depth_ = 0;
};

}  // namespace tools
