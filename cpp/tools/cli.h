#pragma once

#include <cstdint>
#include <limits>
#include <string_view>

namespace tools {

uint16_t Parse16(std::string_view argument, uint16_t minimum = std::numeric_limits<uint16_t>::min(),
                 uint16_t maximum = std::numeric_limits<uint16_t>::max());
uint32_t Parse32(std::string_view argument, uint32_t minimum = std::numeric_limits<uint32_t>::min(),
                 uint32_t maximum = std::numeric_limits<uint32_t>::max());
uint64_t Parse64(std::string_view argument, uint64_t minimum = std::numeric_limits<uint64_t>::min(),
                 uint64_t maximum = std::numeric_limits<uint64_t>::max());

}  // namespace tools
