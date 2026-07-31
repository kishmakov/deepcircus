#pragma once

// Exact decision-tree solvers over a full truth table: the dynamic programs
// that produce a solvable case's target scores. Both walk all 3^bitness ternary
// assignments, so `kMaxSolvableBitness` is what makes a case "solvable" at all
// -- the generator's solvable-table limit is an alias of it.

#include <stddef.h>
#include <stdint.h>

#include <vector>

namespace tools {

// Largest bitness the 3^bitness dynamic programs below stay affordable at.
inline constexpr uint16_t kMaxSolvableBitness = 12;

size_t SolveForDepth(uint16_t bitness, const std::vector<bool>& truth_table);
size_t SolveForSize(uint16_t bitness, const std::vector<bool>& truth_table);

}  // namespace tools
