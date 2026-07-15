#pragma once

#include <stddef.h>
#include <stdint.h>

#include <random>

namespace gen {

// Base class owning a single generated case's deterministic randomness, keyed
// by (bitness, case_id, seed). Provides the case's mt19937 plus a bit-buffered
// fair-coin stream shared by table and tree generation.
class Case {
public:
    Case(uint16_t bitness, size_t case_id, uint64_t seed);

    uint16_t Bitness() const;
    size_t CaseId() const;

    bool Generate();
    std::mt19937& RNG();

    // Draws a fresh 64-bit seed from the case's RNG, used to seed derived
    // sampling streams (e.g. value/restriction input generation).
    uint64_t NextSeed();

protected:
    uint16_t bitness_;
    size_t case_id_;

private:
    std::mt19937 rng_;
    uint32_t bit_buffer_ = 0;
    uint8_t bits_remaining_ = 0;
};

}  // namespace gen
