#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string_view>
#include <vector>

inline constexpr uint16_t kMinTableBitness = 4;
inline constexpr uint16_t kMaxTableBitness = 256; // technical limitation for a while
inline constexpr uint16_t kSolvableTableBitness = 16;

class TableCase {
public:
    TableCase(uint16_t bitness, size_t case_id);

    bool Evaluate(std::string_view input) const;
    void FillValueTensor(size_t reps, uint64_t seed, float* out) const;
    void FillRestrictionsTensor(size_t reps, uint64_t seed, float* out) const;
    const std::vector<bool>& TruthTable() const;

private:
    uint16_t bitness_;
    size_t case_id_;
    std::vector<bool> truth_table_;
};
