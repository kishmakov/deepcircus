#pragma once

// Exact decision-tree solvers over a full truth table: the dynamic programs
// that produce a solvable case's target scores. Only usable up to
// `gen::kSolvableTableBitness` (see `generator/table.h`), since both walk all
// 3^bitness ternary assignments.

#include <stddef.h>
#include <stdint.h>

#include <vector>

namespace gen {

size_t SolveForDepth(uint16_t bitness, const std::vector<bool>& truth_table);
size_t SolveForSize(uint16_t bitness, const std::vector<bool>& truth_table);

}  // namespace gen
