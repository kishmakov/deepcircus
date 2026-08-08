#pragma once

// The two offline training-set sources of `docs/offline_data.md`. Both build an
// entry out of the run's parameters and the entry's coordinates alone, so a
// file stays reproducible from `conf/preparation.conf` and nothing else.

#include <cstdint>

#include "offline/read_write.h"

namespace preparation {

// What a run gives its samplers, straight out of `conf/preparation.conf`. The
// small-size range is the closed interval a `_small` entry's witness tree draws
// its node count from; `RandomEntry` has no use for it and takes it only so the
// two sources stay one signature.
struct Parameters {
    uint64_t seed;
    uint16_t small_size_from;
    uint16_t small_size_to;
};

// Both truth tables drawn uniformly. At the supported bitnesses that lands the
// size target in the hundreds -- the hard end of the range, and effectively the
// only end reachable by drawing tables at random.
offline::Entry RandomEntry(const Parameters& parameters, uint16_t series, uint16_t bitness, uint32_t index);

// A witness decision tree drawn first, with the entry read off it, which is the
// only way to reach small targets: series 1 takes `g` to be what a tree over
// the inputs plus `f` computes, series 2 takes `g` to agree with a tree over
// the inputs on `X` and to be random off it. The witness bounds the target from
// above and the solver decides it -- a pair whose distinctions partly collapse
// keeps whatever smaller size it came out at, so a `_small` file's targets
// spread over the range rather than pinning to one value per entry. The node
// count cycles over the parameters' closed range as the entry index walks.
offline::Entry SmallEntry(const Parameters& parameters, uint16_t series, uint16_t bitness, uint32_t index);

}  // namespace preparation
