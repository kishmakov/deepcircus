#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace gen {

namespace {

constexpr uint64_t kSplitMixIncrement = 0x9e3779b97f4a7c15ull;

class SplitMixGenerator {
public:
    explicit SplitMixGenerator(uint64_t seed) : state_(seed) {}

    uint64_t Generate() { return SplitMix64(state_); }

    uint64_t GenerateBelow(uint64_t bound) {
        assert(bound > 0);
        const uint64_t threshold = static_cast<uint64_t>(-bound) % bound;
        while (true) {
            const uint64_t value = Generate();
            if (value >= threshold) {
                return value % bound;
            }
        }
    }

private:
    uint64_t state_;
};

}  // namespace

uint64_t Mix64(uint64_t value) {
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

uint64_t SplitMix64(uint64_t& state) { return Mix64(state += kSplitMixIncrement); }

uint64_t TaskSeed(uint64_t seed, uint16_t bitness, uint64_t iteration) {
    uint64_t state = seed ^ (static_cast<uint64_t>(bitness) << 48) ^ iteration;
    return SplitMix64(state);
}

uint64_t DomainSeed(uint64_t seed, uint64_t domain, uint16_t bitness) {
    uint64_t state = seed ^ domain ^ (static_cast<uint64_t>(bitness) << 48);
    return SplitMix64(state);
}

std::vector<size_t> SampleCaseIds(size_t population, size_t cases, uint64_t seed) {
    assert(cases > 0);
    assert(cases <= population);
    SplitMixGenerator rng(seed);
    std::unordered_set<size_t> selected;
    selected.reserve(cases * 2);
    std::vector<size_t> result;
    result.reserve(cases);

    for (size_t current = population - cases; current < population; ++current) {
        const size_t candidate = rng.GenerateBelow(static_cast<uint64_t>(current) + 1);
        const size_t case_id = selected.contains(candidate) ? current : candidate;
        const bool inserted = selected.insert(case_id).second;
        assert(inserted);
        result.push_back(case_id);
    }
    return result;
}

size_t FullBitId(size_t bit_id, size_t fixed_id) { return bit_id < fixed_id ? bit_id : bit_id + 1; }

float SizeScore(uint16_t bitness, size_t tree_size) {
    const double max_size = std::exp2(static_cast<double>(bitness));
    assert(static_cast<double>(tree_size) < max_size);
    return static_cast<float>(std::log2(max_size - static_cast<double>(tree_size)));
}

}  // namespace gen
