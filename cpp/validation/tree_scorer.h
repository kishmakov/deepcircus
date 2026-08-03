#pragma once

// How cheap an optimal decision tree over a set of slots is -- the measure the
// reconstruction search orders its states by, on the same scale the model is
// trained to predict.

#include <cstddef>
#include <vector>

namespace func {

// The optimal decision tree over some set of slots: `depth` queries deep, and
// `log_size` its node count on the scale the model is trained against
// (`gen::SizeScore`): log2(2^slots - size). A constant function scores `slots`
// there and a full tree scores 0, so a cheaper tree has a smaller `depth` but a
// *larger* `log_size`.
struct TreeScore {
    size_t depth = 0;
    double log_size = 0.0;
};

// The optimal decision tree over a truth table, tabulated the way the exact
// solvers want it: entry `id` is the value at the assignment whose slot `k` is
// bit `k` of `id`, so the slot count is the log2 of the entry count. Only
// tables the solvers can walk have a score.
TreeScore Score(const std::vector<bool>& truth_table);

}  // namespace func
