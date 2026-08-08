#pragma once

#include <stddef.h>
#include <stdint.h>

#include <variant>
#include <vector>

#include "case.h"

namespace gen {

inline constexpr uint16_t kMinTreeBitness = 10;
inline constexpr uint16_t kMaxTreeBitness = 256;

// Ceiling on a case's internal-node budget. The builder's per-level clamps
// already bound a tree by what `bitness` bits can hold; this bounds the other
// side, so a case seeded at a high bitness stays buildable instead of
// following its budget into hundreds of millions of nodes. It sits above what
// the trained bitnesses reach, so it only binds well past them.
inline constexpr size_t kMaxTreeSize = size_t{1} << 20;

struct Div {
    size_t bitId;
    size_t child0;
    size_t child1;
};

using Node = std::variant<Div, bool>;

struct TreeCase : Case {
    TreeCase(uint16_t bitness, uint64_t seed);

    std::vector<Node> nodes;
    std::vector<bool> used_bits;
    size_t num_leafs = 0;
    size_t depth = 0;

    size_t AddLeaf(bool value);

    bool Evaluate(const std::vector<bool>& input) const override;

private:
    size_t BuildSubtree(size_t budget, std::vector<bool>& path_used_bits, size_t path_used_count, bool required_value);
};

}  // namespace gen

