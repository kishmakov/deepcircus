#include "cli.h"

#include <cassert>
#include <charconv>

namespace tools {

uint16_t Parse16(std::string_view argument, uint16_t minimum, uint16_t maximum) {
    return static_cast<uint16_t>(Parse64(argument, minimum, maximum));
}

uint32_t Parse32(std::string_view argument, uint32_t minimum, uint32_t maximum) {
    return static_cast<uint32_t>(Parse64(argument, minimum, maximum));
}

uint64_t Parse64(std::string_view argument, uint64_t minimum, uint64_t maximum) {
    uint64_t value = 0;
    const auto [end, error] = std::from_chars(argument.data(), argument.data() + argument.size(), value);
    assert(error == std::errc{} && end == argument.data() + argument.size());
    assert(value >= minimum && value <= maximum);
    return value;
}

}  // namespace tools
