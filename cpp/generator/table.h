#pragma once

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "case.h"

namespace gen {

inline constexpr uint16_t kMinTableBitness = 4;
inline constexpr uint16_t kMaxTableBitness = 256;  // technical limitation for a while
inline constexpr uint16_t kSolvableTableBitness = 12;

class TableCase : public Case {
public:
    TableCase(uint16_t bitness, size_t case_id);

    bool Evaluate(const std::vector<bool>& input) const override;
    const std::vector<bool>& TruthTable() const;

private:
    std::vector<bool> truth_table_;
    uint64_t sparse_seed_ = 0;
};

}  // namespace gen

