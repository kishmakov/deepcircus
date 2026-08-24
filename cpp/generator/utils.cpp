#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace gen {

size_t FullBitId(size_t bit_id, size_t fixed_id) { return bit_id < fixed_id ? bit_id : bit_id + 1; }

float SizeScore(uint16_t bitness, size_t tree_size) {
    const double max_size = std::exp2(static_cast<double>(bitness));
    assert(static_cast<double>(tree_size) < max_size);
    return static_cast<float>(std::log2(max_size - static_cast<double>(tree_size)));
}

}  // namespace gen
