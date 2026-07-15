#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string_view>
#include <variant>
#include <vector>

#include "case.h"

namespace gen {

inline constexpr size_t kTreeCasesNumber = size_t{1} << 32;
inline constexpr uint16_t kMinTreeBitness = 10;
inline constexpr uint16_t kMaxTreeBitness = 256;

struct Div {
    size_t bitId;
    size_t child0;
    size_t child1;
};

using Node = std::variant<Div, bool>;

struct DecisionTree : Case {
    DecisionTree(uint16_t bitness, size_t case_id, uint64_t seed);

    std::vector<Node> nodes;
    std::vector<bool> used_bits;
    size_t num_leafs = 0;
    size_t depth = 0;

    size_t AddLeaf(bool value);

    void Finalize();

    bool Evaluate(std::string_view input) const;

    void FillValueTensor(size_t reps, uint64_t seed, float* out) const;

private:
    size_t BuildSubtree(size_t budget, std::vector<bool>& path_used_bits, size_t path_used_count, bool required_value);
};

}  // namespace gen

