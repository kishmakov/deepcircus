#include "tree_scorer.h"

#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>

#include "solver.h"

namespace func {

namespace {

// The size score the model is trained against (`gen::SizeScore`):
// log2(2^slots - size), where `size` counts internal nodes.
double ScoreLogSize(size_t slots, size_t size) {
    const double max_size = std::exp2(static_cast<double>(slots));
    assert(static_cast<double>(size) < max_size);
    return std::log2(max_size - static_cast<double>(size));
}

}  // namespace

TreeScore Score(const std::vector<bool>& truth_table) {
    assert(std::has_single_bit(truth_table.size()));
    const uint16_t slots = static_cast<uint16_t>(std::countr_zero(truth_table.size()));
    assert(slots <= tools::kMaxSolvableBitness);
    size_t size = tools::SolveForSize(slots, truth_table);
    return {tools::SolveForDepth(slots, truth_table), ScoreLogSize(slots, size)};
}

}  // namespace func
