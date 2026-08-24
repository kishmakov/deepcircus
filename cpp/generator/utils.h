#pragma once

#include <stddef.h>
#include <stdint.h>

namespace gen {

size_t FullBitId(size_t bit_id, size_t fixed_id);

// Size training target: log2(2^bitness - tree_size), where tree_size counts
// internal nodes. A constant function scores bitness, a full tree scores 0.
float SizeScore(uint16_t bitness, size_t tree_size);

}  // namespace gen
