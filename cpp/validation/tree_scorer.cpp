#include "tree_scorer.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <utility>

#include "tools/solver.h"

namespace func {

namespace {

// The size score the model is trained against (`gen::SizeScore`):
// log2(2^slots - size), where `size` counts internal nodes.
double ScoreLogSize(size_t slots, size_t size) {
    const double max_size = std::exp2(static_cast<double>(slots));
    assert(static_cast<double>(size) < max_size);
    return std::log2(max_size - static_cast<double>(size));
}

// The exact solvers index by row, so an evaluation is scattered back into a
// dense truth table. Every row is present while evaluations stay exhaustive.
std::vector<bool> DenseValues(const Evaluation& evaluation) {
    std::vector<bool> dense(evaluation.Rows());
    for (size_t id = 0; id < evaluation.Rows(); ++id) {
        assert(evaluation.rows[id] < dense.size());
        dense[evaluation.rows[id]] = evaluation.values[id];
    }
    return dense;
}

}  // namespace

size_t Evaluation::Slots() const {
    assert(std::has_single_bit(Rows()));
    return static_cast<size_t>(std::countr_zero(Rows()));
}

bool Evaluation::UsesEverySlot() const {
    const std::vector<bool> dense = DenseValues(*this);

    for (size_t id = 0; id < Slots(); ++id) {
        const size_t bit = size_t{1} << id;
        bool sensitive = false;
        for (size_t row = 0; row < dense.size() && !sensitive; ++row) {
            sensitive = dense[row] != dense[row ^ bit];
        }
        if (!sensitive) {
            return false;
        }
    }
    return true;
}

Evaluation Tabulate(const Scheme& scheme) {
    const size_t rows = size_t{1} << scheme.InputCount(0);

    Evaluation table;
    for (size_t row = 0; row < rows; ++row) {
        const SchemeInput input = static_cast<SchemeInput>(row);
        table.Append(input, scheme.ComputeValue(input));
    }
    return table;
}

TreeScore Score(const Evaluation& evaluation) {
    const std::vector<bool> dense = DenseValues(evaluation);
    const uint16_t slots = static_cast<uint16_t>(evaluation.Slots());
    assert(slots <= tools::kMaxSolvableBitness);

    size_t size = tools::SolveForSize(slots, dense);
    return {tools::SolveForDepth(slots, dense), ScoreLogSize(slots, size)};
}

}  // namespace func
