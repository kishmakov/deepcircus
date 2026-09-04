#include "tools/score.h"

#include <cassert>
#include <cmath>

namespace tools {

float DepthScore(uint16_t bitness, uint32_t depth) {
    assert(depth <= bitness);
    return static_cast<float>(bitness - depth);
}

float SizeScore(uint16_t bitness, uint32_t size) {
    const double max_size = std::exp2(static_cast<double>(bitness));
    assert(static_cast<double>(size) < max_size);
    return static_cast<float>(std::log2(max_size - static_cast<double>(size)));
}

}  // namespace tools
