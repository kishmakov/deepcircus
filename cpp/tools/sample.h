#pragma once

// Input-point sampling, and the bit primitives it is built from. Self-contained
// by design: nothing here knows about function representations or offline
// files.

#include <stddef.h>
#include <stdint.h>

#include <functional>
#include <vector>

namespace tools {

// Input sampling shape: `batches` independent samplings, each expanded into
// `batch_size` (power-of-two and greater than one) points.
struct InputShape {
    uint16_t batches;
    uint16_t batch_size;
};

// Fair-coin bit stream the base sequences are drawn from; a case supplies its
// own deterministic one, keyed by (bitness, seed).
using BitSource = std::function<bool()>;

// Draws shape.batches base sequences of `dims` bits off `next_bit` and expands
// them into the block-and-random input walk: batch 0 follows a Gray-code walk,
// and later batches flip groups keyed by batch and point id. Returns
// batches * batch_size points of `dims` bits each, concatenated.
std::vector<bool> SampleInputs(InputShape shape, uint16_t dims, const BitSource& next_bit);

}  // namespace tools
