#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <vector>

namespace func {

// The bitness range every kind of function is built over.
inline constexpr uint16_t kMinBitness = 8;
// Technical limitation for a while.
inline constexpr uint16_t kMaxBitness = 256;

// Widest function worth materializing as a truth table.
inline constexpr uint16_t kMaxMaterializedBitness = 16;

// Base interface for the functional object.
struct Func {
    using FuncInput = std::vector<bool>;

    Func(uint16_t bitness) : bitness_(bitness) {}

    virtual bool operator()(const FuncInput& input) const = 0;

    virtual std::vector<uint8_t> serialize() const = 0;

    // Every value, laid out by the input read as a little-endian index. Built
    // on the spot, so only ask a bitness whose table fits in memory.
    std::vector<bool> TruthTable() const {
        assert(bitness_ <= kMaxMaterializedBitness);

        std::vector<bool> truth_table(size_t{1} << bitness_);
        FuncInput input(bitness_);
        for (size_t id = 0; id < truth_table.size(); ++id) {
            for (uint16_t bit_id = 0; bit_id < bitness_; ++bit_id) {
                input[bit_id] = ((id >> bit_id) & 1u) != 0;
            }
            truth_table[id] = (*this)(input);
        }
        return truth_table;
    }

    virtual ~Func() = default;

protected:

    const uint16_t bitness_;
};

}  // namespace func
