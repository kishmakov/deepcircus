#include "utils.h"

#include <cstdint>

namespace gen {

size_t FullBitId(size_t bit_id, size_t fixed_id) { return bit_id < fixed_id ? bit_id : bit_id + 1; }

}  // namespace gen
