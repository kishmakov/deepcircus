#pragma once

// Deterministic offline samplers for `M_1[g | f]` and `M_2[g | X]`; see
// `docs/paper.tex`.

#include <cstdint>

#include "offline/read_write.h"

namespace prep {

enum class Model : uint8_t {
    kM1 = 1,
    kM2 = 2,
};

struct Parameters {
    uint64_t seed;
};

offline::Entry SolvedEntry(const Parameters& parameters, Model model, uint16_t bitness, uint32_t index);
offline::Entry UnsolvedEntry(const Parameters& parameters, Model model, uint16_t bitness, uint32_t index);

}  // namespace prep
