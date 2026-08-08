#pragma once

// Exact decision-tree solvers over a full truth table: the dynamic programs
// that produce a solvable case's target scores.

#include <stddef.h>
#include <stdint.h>

#include <vector>

namespace tools {

// Largest bitness the solvers below stay affordable at.
inline constexpr uint16_t kMaxSolvableBitness = 12;

// Optimal depth, and optimal internal-node count, of a decision tree computing
// `truth_table` by querying input bits.
size_t SolveForDepth(uint16_t bitness, const std::vector<bool>& truth_table);
size_t SolveForSize(uint16_t bitness, const std::vector<bool>& truth_table);

// The same targets for a tree that may also query `helper_table` -- a second
// function of the same inputs, once per path -- which is what the paper's
// S_1[g | f] scores. `truth_table` is g, `helper_table` is f.
size_t SolveForDepthGiven(uint16_t bitness, const std::vector<bool>& truth_table,
                          const std::vector<bool>& helper_table);
size_t SolveForSizeGiven(uint16_t bitness, const std::vector<bool>& truth_table,
                         const std::vector<bool>& helper_table);

// The same targets for a tree that only has to compute `truth_table` on the
// subset `subset_table` indicates, which is what the paper's S_2[g | X] scores.
// `truth_table` is g, `subset_table` is the indicator of X: assignments outside
// it are don't-cares.
size_t SolveForDepthRestricted(uint16_t bitness, const std::vector<bool>& truth_table,
                               const std::vector<bool>& subset_table);
size_t SolveForSizeRestricted(uint16_t bitness, const std::vector<bool>& truth_table,
                              const std::vector<bool>& subset_table);

}  // namespace tools
