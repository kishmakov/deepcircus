#pragma once

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "func/func.h"
#include "tools/binary_tree.h"

namespace func {

inline constexpr uint16_t kMinTreeBitness = 8;
// Technical limitation for a while.
inline constexpr uint16_t kMaxTreeBitness = 256;

// `bitness` bounds path depth; these cap BinaryTree addressing and case memory.
inline constexpr uint32_t kMaxTreeDepth = 24;
inline constexpr uint32_t kMaxTreeSize = uint32_t{1} << 20;

// A decision-tree boolean function: internal values select input bits, with 0
// going left and 1 right; leaves hold results.
class TreeFunc : public func::Func {
public:
    // Uniformly samples depth, size, shape, and values from `(bitness, seed)`.
    TreeFunc(uint16_t bitness, uint64_t seed);
    TreeFunc(uint16_t bitness, tools::BinaryTree tree);
    TreeFunc(uint16_t bitness, const std::vector<uint8_t>& bytes);

    using Func::operator();
    bool operator()(const FuncInput& input) const override;

    // Pre-order: uint32_t node count, then bit ids or high-bit-marked leaves.
    // This matches `Split` id order, preserving exact tree round trips.
    std::vector<uint8_t> serialize() const override;

    // Case scoring metrics.
    uint32_t Size() const { return tree_.InternalNodes(); }
    uint32_t Depth() const { return tree_.Depth(); }

private:
    const tools::BinaryTree tree_;
};

}  // namespace func
