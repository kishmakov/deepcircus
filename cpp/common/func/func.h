#pragma once

#include <stdint.h>

#include <vector>

namespace func {

// Base interface for the functional object.
struct Func {
    using FuncInput = std::vector<bool>;

    Func(uint16_t bitness) : bitness_(bitness) {}

    virtual bool operator()(const FuncInput& input) const = 0;

    virtual std::vector<uint8_t> serialize() const = 0;

    virtual std::vector<bool> operator()(const std::vector<FuncInput>& inputs) const {
        std::vector<bool> values;
        values.reserve(inputs.size());

        for (const auto& input: inputs) {
            values.emplace_back((*this)(input));
        }

        return values;
    }

    virtual ~Func() = default;

protected:

    const uint16_t bitness_;
};

}  // namespace func
