#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "tools/random.h"

namespace gen {

uint64_t TaskSeed(uint64_t seed, uint16_t bitness, uint64_t iteration) {
    uint64_t state = seed ^ (static_cast<uint64_t>(bitness) << 48) ^ iteration;
    return tools::SplitMix64(state);
}

uint64_t DomainSeed(uint64_t seed, uint64_t domain, uint16_t bitness) {
    uint64_t state = seed ^ domain ^ (static_cast<uint64_t>(bitness) << 48);
    return tools::SplitMix64(state);
}

std::vector<size_t> SampleCaseIds(size_t population, size_t cases, uint64_t seed) {
    assert(cases > 0);
    assert(cases <= population);
    tools::Random random(seed);
    std::unordered_set<size_t> selected;
    selected.reserve(cases * 2);
    std::vector<size_t> result;
    result.reserve(cases);

    for (size_t current = population - cases; current < population; ++current) {
        const size_t candidate = random.Below(static_cast<uint64_t>(current) + 1);
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
