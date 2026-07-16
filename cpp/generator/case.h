#pragma once

#include "generator.h"

#include <stddef.h>
#include <stdint.h>

#include <functional>
#include <random>
#include <string>
#include <vector>

namespace gen {

// Base class owning a single generated case's deterministic randomness, keyed
// by (bitness, case_id). Provides the case's mt19937 plus a bit-buffered
// fair-coin stream shared by table and tree generation.
class Case {
public:
    Case(uint16_t bitness, size_t case_id);

    bool GenerateBool();

    // Draws a fresh random sequence of `bitness - subtract` fair-coin bits
    // (all `bitness` bits by default).
    std::vector<bool> GenerateSequence(uint16_t subtract = 0);

    // Samples the full function: batches x batch_size points, each dumped as
    // its bits followed by ComputeAt (value plus every single-bit flip).
    std::vector<bool> SampleValues(InputShape shape);

    // Samples every restriction: for each bit fixed to 0 and to 1, runs the
    // same batch/block sampling over the remaining bitness - 1 free variables.
    std::vector<bool> SampleRestrictions(InputShape shape);

    // One flip sample of `input` (which must be `bitness` bits) as a 0/1 string
    // of length 2 * bitness + 1: the input bits, the function value, then the
    // value at each single-bit flip.
    std::string SampledValueString(const std::vector<bool>& input) const;

    virtual ~Case() = default;

protected:
    // Computes a sampled block for one point: given the point's bits, returns
    // the function value there followed by its value at each single-bit flip
    // (1 + point-size values).
    using ComputeBlock = std::function<std::vector<bool>(const std::vector<bool>&)>;

    // Point evaluation of the underlying case function; implemented by the
    // concrete case kind (table lookup, tree walk, ...).
    virtual bool Evaluate(const std::vector<bool>& input) const = 0;

    // Returns bitness + 1 bits: values at input and all variations of input
    // with each bit flipped.
    std::vector<bool> ComputeAt(const std::vector<bool>& input) const;

    // Batch + block-inversion sampling over `dims` free variables: batches x
    // batch_size points, each appended as its `dims` bits followed by
    // compute(point) (which returns 1 + dims values). Shared by SampleValues
    // and SampleRestrictions.
    std::vector<bool> Sample(InputShape shape, uint16_t dims, const ComputeBlock& compute);

    uint16_t bitness_;
    size_t case_id_;

    std::mt19937 rng_;
    uint32_t bit_buffer_ = 0;
    uint8_t bits_remaining_ = 0;
};

}  // namespace gen
