#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "case.h"
#include "tools/solver.h"

namespace func {

inline constexpr uint16_t kMinTableBitness = 4;
// Technical limitation for a while.
inline constexpr uint16_t kMaxTableBitness = 256;
// Tables at or below this bitness get exact targets.
inline constexpr uint16_t kSolvableTableBitness = tools::kMaxSolvableBitness;

// A uniformly random boolean function of `bitness` inputs, drawn from `seed`
// alone. At or below kSolvableTableBitness the truth table is materialized and
// readable through TruthTable(); above it the function stays implicit and is
// hashed per point.
class TableCase : public gen::Case {
public:
    TableCase(uint16_t bitness, uint64_t seed);

    bool Evaluate(const std::vector<bool>& input) const override;
    // Only materialized at or below kSolvableTableBitness.
    const std::vector<bool>& TruthTable() const;

private:
    std::vector<bool> truth_table_;
    uint64_t sparse_seed_ = 0;
};

uint16_t TableSolvableBitness();

// Input: bitness bits. Output length: 2 * bitness + 1.
std::string TableValue(uint16_t bitness, uint64_t seed, const std::vector<bool>& input);

std::vector<uint64_t> TableSampleSeeds(uint16_t bitness, size_t cases, uint64_t task_seed);

gen::GeneratedValues TableValuesForSeeds(uint16_t bitness, const std::vector<uint64_t>& seeds, gen::InputShape shape);

// Synchronously generates recursive table values and one dense restriction
// matrix for an explicit, pre-sampled chunk of case seeds.
gen::GeneratedRestrictions TableRestrictionsForSeeds(uint16_t bitness, const std::vector<uint64_t>& seeds,
                                                     gen::InputShape shape);

}  // namespace func
