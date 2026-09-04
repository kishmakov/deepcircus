#pragma once

#include <stdint.h>

#include <vector>

#include "func/func.h"

namespace func {

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

}  // namespace func
