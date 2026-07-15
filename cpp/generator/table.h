#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string_view>
#include <vector>

#include "case.h"

inline constexpr uint16_t kMinTableBitness = 4;
inline constexpr uint16_t kMaxTableBitness = 256;  // technical limitation for a while
inline constexpr uint16_t kSolvableTableBitness = 12;

class TableCase : Case {
public:
    TableCase(uint16_t bitness, size_t case_id, uint64_t seed);

    bool Evaluate(std::string_view input) const;
    void FillValueTensor(size_t reps, uint64_t seed, float* out) const;
    void FillRestrictionsTensor(size_t reps, uint64_t seed, float* out) const;
    const std::vector<bool>& TruthTable() const;

private:
    std::vector<bool> truth_table_;
    uint64_t sparse_seed_ = 0;
};
