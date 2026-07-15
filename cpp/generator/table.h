#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string_view>
#include <vector>

#include "case.h"

namespace gen {

inline constexpr uint16_t kMinTableBitness = 4;
inline constexpr uint16_t kMaxTableBitness = 256;  // technical limitation for a while
inline constexpr uint16_t kSolvableTableBitness = 12;

class TableCase : Case {
public:
    TableCase(uint16_t bitness, size_t case_id, uint64_t seed);

    bool Evaluate(std::string_view input) const;
    void FillValueTensor(size_t reps, std::vector<bool>& out, size_t base);
    void FillRestrictionsTensor(size_t reps, std::vector<bool>& out, size_t base);
    const std::vector<bool>& TruthTable() const;

private:
    std::vector<bool> truth_table_;
    uint64_t sparse_seed_ = 0;
};

size_t SolveForDepth(uint16_t bitness, const std::vector<bool>& truth_table);
size_t SolveForSize(uint16_t bitness, const std::vector<bool>& truth_table);

}  // namespace gen

