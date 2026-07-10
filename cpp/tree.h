#pragma once

#include <stddef.h>
#include <stdint.h>

#include "decision_tree.h"

inline constexpr size_t kTreeCasesNumber = size_t{1} << 32;
inline constexpr uint16_t kMinTreeBitness = 10;
inline constexpr uint16_t kMaxTreeBitness = 256;

DecisionTree BuildTreeCase(uint16_t bitness, size_t case_id);
