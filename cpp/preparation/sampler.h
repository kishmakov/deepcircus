#pragma once

// Deterministic offline samplers for `M_1[g | f]` (series 1) and
// `M_2[g | X]` (series 2); see `docs/paper.tex`; the `M_1` file format is in
// `docs/data_m1.md`.

#include <cstdint>

#include "offline/read_write.h"

namespace preparation {

struct Parameters {
    uint64_t seed;
    uint16_t small_size_from;
    uint16_t small_size_to;
};

// Samples `g` and `f` / `X` uniformly.
offline::Entry RandomEntry(const Parameters& parameters, uint16_t series, uint16_t bitness, uint32_t index);

// Samples `g` via small [witness] tree, paired with `f` / `X`.
offline::Entry SmallEntry(const Parameters& parameters, uint16_t series, uint16_t bitness, uint32_t index);

}  // namespace preparation
