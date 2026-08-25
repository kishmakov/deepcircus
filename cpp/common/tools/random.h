#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

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

// Seed of one offline entry, keyed off its coordinates alone, so it never depends on the draws entries before it
// burned. Salt `seed` (`Mix(seed) ^ stream`) to keep one kind of entry's stream clear of another's.
uint64_t EntrySeed(uint64_t seed, uint16_t series, uint16_t bitness, uint32_t index);

// Seed of one generation task, derived from the run seed.
uint64_t TaskSeed(uint64_t seed, uint16_t bitness, uint64_t iteration);

// Seed of one sampling domain within a task, so that domains draw independently.
uint64_t DomainSeed(uint64_t seed, uint64_t domain, uint16_t bitness);

// Returns seeds for initialization of cases.
std::vector<uint64_t> SampleSeeds(size_t count, uint64_t task_seed);

}  // namespace tools
