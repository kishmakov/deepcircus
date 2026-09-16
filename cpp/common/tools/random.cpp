#include "tools/random.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tools {
namespace {

constexpr uint64_t kSplitMixIncrement = 0x9e3779b97f4a7c15ull;

constexpr uint64_t kFnvPrime = 0x100000001b3ull;

}  // namespace

uint64_t Mix64(uint64_t value) {
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

uint64_t Mix(uint64_t value) { return Mix64(value + kSplitMixIncrement); }

uint64_t SplitMix64(uint64_t& state) { return Mix64(state += kSplitMixIncrement); }

uint64_t Random::NextBits(unsigned count) {
    uint64_t result = 0;
    for (unsigned written = 0; written < count;) {
        if (used_ == 64) {
            word_ = SplitMix64(state_);
            used_ = 0;
        }
        const unsigned take = std::min(count - written, 64 - used_);
        const uint64_t bits = (word_ >> used_) & (UINT64_MAX >> (64 - take));
        result |= bits << written;
        used_ += take;
        written += take;
    }
    return result;
}

uint64_t Random::NextU64() { return NextBits(64); }

uint8_t Random::NextU8() { return static_cast<uint8_t>(NextBits(8)); }

bool Random::NextBool() { return NextBits(1) != 0; }

uint64_t Random::Below(uint64_t bound) {
    assert(bound > 0);
    const uint64_t threshold = static_cast<uint64_t>(-bound) % bound;
    while (true) {
        const uint64_t value = NextU64();
        if (value >= threshold) return value % bound;
    }
}

uint64_t EntrySeed(uint64_t seed, uint16_t series, uint16_t bitness, uint32_t index) {
    uint64_t state = Mix(seed);
    state = Mix(state ^ series);
    state = Mix(state ^ bitness);
    return Mix(state ^ index);
}

uint64_t DomainSeed(uint64_t seed, uint64_t domain, uint16_t bitness) {
    return Mix(seed ^ domain ^ (static_cast<uint64_t>(bitness) << 48));
}

bool RandomFuncValue(uint16_t bitness, uint64_t seed, const std::vector<bool>& bits) {
    assert(bits.size() == bitness);

    size_t used = bitness;
    while (used > 0 && !bits[used - 1]) {
        --used;
    }

    uint64_t value = seed;
    for (size_t bit_id = 0; bit_id < used; ++bit_id) {
        value ^= static_cast<uint64_t>(bits[bit_id]);
        value *= kFnvPrime;
    }
    return (Mix64(value) & 1ull) != 0;
}

}  // namespace tools
