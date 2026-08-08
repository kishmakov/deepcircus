#pragma once

// Input-point sampling, and the bit primitives it is built from. Self-contained
// by design: nothing here knows about cases, tables, trees, or circuits, so
// `cpp/tools` never includes `cpp/generator` (the dependency runs the other
// way -- `generator.h` includes this header).

#include <stddef.h>
#include <stdint.h>

#include <functional>
#include <string_view>
#include <vector>

namespace tools {

// Input sampling shape: `batches` independent samplings, each expanded into
// `batch_size` (power-of-two) points.
struct InputShape {
    uint16_t batches;
    uint16_t batch_size;
};

// Fair-coin bit stream the base sequences are drawn from; a case supplies its
// own deterministic one, keyed by (bitness, seed).
using BitSource = std::function<bool()>;

// Converts a 0/1 char string into a bit vector, one bool per char.
std::vector<bool> BitsFromChars(std::string_view input);

// Produces split of `bitness` bits into `groups` roughly equal groups.
// Approximately [0 .. bitness / groups], [bitness / groups .. 2 * bitness / groups], ...
// Then shifts this cyclically based on `way`.
std::vector<uint16_t> SplitBitsInGroups(uint16_t bitness, uint16_t groups, uint16_t way);

// Gray-code successor of `sequence`: exactly one bit changes per call and the
// walk cycles through all bit sequences of that length.
std::vector<bool> NextSequence(const std::vector<bool>& sequence);

// Draws a fresh random sequence of `length` fair-coin bits, in stream order.
std::vector<bool> GenerateSequence(uint16_t length, const BitSource& next_bit);

// Expands one base sequence per batch into the block-and-random input walk
// shared with case sampling: batch 0 follows the Gray-code NextSequence walk,
// each later batch flips bit groups keyed by (batch, point id). `sequences`
// must hold shape.batches equal-length sequences; returns batches * batch_size
// points, each sequence-length bits, concatenated.
std::vector<bool> ExpandInputs(InputShape shape, const std::vector<std::vector<bool>>& sequences);

// Draws shape.batches base sequences of `dims` bits off `next_bit` and expands
// them: the whole base-sequences-to-points path, returning
// batches * batch_size points of `dims` bits each, concatenated.
std::vector<bool> SampleInputs(InputShape shape, uint16_t dims, const BitSource& next_bit);

}  // namespace tools
