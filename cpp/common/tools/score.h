#pragma once

// The training targets a model is fitted against: the raw depth and size of a
// decision tree, mapped onto the two scores `M[g | f]` predicts. Both live in
// `[0, bitness]`, and both grow as the function gets simpler, so a constant
// function scores `bitness` twice and a full tree scores zero twice.

#include <stdint.h>

namespace tools {

// bitness - depth.
float DepthScore(uint16_t bitness, uint32_t depth);

// log2(2^bitness - size), where `size` counts internal nodes.
float SizeScore(uint16_t bitness, uint32_t size);

}  // namespace tools
