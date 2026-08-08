#pragma once

#include <cstdint>

namespace tools {

// SplitMix64's output finalizer, without advancing a state.
uint64_t Mix64(uint64_t value);

// One SplitMix64 step without retaining the advanced state.
uint64_t Mix(uint64_t value);

// Advances `state` and returns the next SplitMix64 output.
uint64_t SplitMix64(uint64_t& state);

class Random {
public:
    explicit Random(uint64_t seed) : state_(seed) {}

    uint64_t Next();
    uint64_t Below(uint64_t bound);
    bool Bool();

private:
    uint64_t state_;
};

}  // namespace tools
