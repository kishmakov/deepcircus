#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "func/func.h"
#include "generator.h"

namespace func {

inline constexpr uint16_t kMinTableBitness = 8;
// Technical limitation for a while.
inline constexpr uint16_t kMaxTableBitness = 256;

// A uniformly random boolean function keyed on `seed` alone.
class TableFunc : public func::Func {
public:
    TableFunc(uint16_t bitness, uint64_t seed);
    TableFunc(uint16_t bitness, std::vector<uint8_t> bytes);

    // The point overload alone would hide the batch one inherited from Func.
    using Func::operator();
    bool operator()(const FuncInput& input) const override;
    std::vector<uint8_t> serialize() const override;

private:
    const uint64_t seed_;
};

// Input: bitness bits. Output length: 2 * bitness + 1.
std::string TableValue(uint16_t bitness, uint64_t seed, const std::vector<bool>& input);

std::vector<uint64_t> TableSampleSeeds(uint16_t bitness, size_t cases, uint64_t task_seed);

gen::GeneratedValues TableValuesForSeeds(uint16_t bitness, const std::vector<uint64_t>& seeds, gen::InputShape shape);

// Synchronously generates recursive table values and one dense restriction
// matrix for an explicit, pre-sampled chunk of case seeds.
gen::GeneratedRestrictions TableRestrictionsForSeeds(uint16_t bitness, const std::vector<uint64_t>& seeds,
                                                     gen::InputShape shape);

}  // namespace func
