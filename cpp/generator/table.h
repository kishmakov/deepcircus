#pragma once

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "case.h"
#include "solver.h"

namespace gen {

inline constexpr uint16_t kMinTableBitness = 4;
// Technical limitation for a while.
inline constexpr uint16_t kMaxTableBitness = 256;
// Tables at or below this bitness get exact targets.
inline constexpr uint16_t kSolvableTableBitness = tools::kMaxSolvableBitness;

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

