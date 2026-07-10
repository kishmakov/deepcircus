#pragma once

#include "bool_bench.h"

#include <stddef.h>
#include <stdint.h>

#include <random>
#include <string_view>
#include <variant>
#include <vector>

class RandomBoolGenerator;

struct Div {
    size_t bitId;
    size_t child0;
    size_t child1;
};

using Node = std::variant<Div, bool>;

struct DecisionTree {
    explicit DecisionTree(uint16_t bitness);

    uint16_t Bitness() const;

    std::vector<Node> nodes;
    std::vector<bool> used_bits;
    size_t num_leafs = 0;
    size_t depth = 0;

    size_t AddLeaf(bool value);

    size_t BuildSubtree(
        size_t budget,
        std::vector<bool>& path_used_bits,
        size_t path_used_count,
        bool required_value,
        RandomBoolGenerator& rng);

    void Finalize();

    bool Evaluate(std::string_view input) const;

    void FillValueTensor(
        size_t reps,
        uint64_t seed,
        float* out) const;

    void FillRestrictionsTensor(
        size_t reps,
        uint64_t seed,
        float* out) const;

private:
    uint16_t bitness_;
};

size_t SolveForDepth(uint16_t bitness, const std::vector<bool>& truth_table);
size_t SolveForSize(uint16_t bitness, const std::vector<bool>& truth_table);
