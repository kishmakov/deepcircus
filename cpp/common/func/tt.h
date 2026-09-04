#pragma once

#include <stdint.h>

#include <vector>

#include "func/func.h"
#include "func/table.h"
#include "func/tree.h"

namespace func {

// The tree spends one of its variables on the table's value, so the function
// stops one bit short of what the tree can address.
inline constexpr uint16_t kMaxTTBitness = kMaxBitness - 1;

// The `(g, f)` pair the model of `docs/paper.tex` is trained on, built so that
// the answer comes with it.
class TTFunc : public func::Func {
public:
    // Splits `seed` into one stream for the table and one for the tree, so the
    // pair is deterministic in `(bitness, seed)` alone.
    TTFunc(uint16_t bitness, uint64_t seed);
    TTFunc(uint16_t bitness, TableFunc table, TreeFunc tree);
    TTFunc(uint16_t bitness, const std::vector<uint8_t>& bytes);

    // Keep the inherited batch overload visible.
    using Func::operator();
    bool operator()(const FuncInput& input) const override;

    // The two children's bytes, the table's behind a uint32_t length so the
    // split survives a change in either format.
    std::vector<uint8_t> serialize() const override;

    // Upper bounds on the witnessed tree complexity, not the minimum.
    uint32_t Size() const { return tree_.Size(); }
    uint32_t Depth() const { return tree_.Depth(); }

private:
    const TableFunc table_;
    const TreeFunc tree_;
};

}  // namespace func
