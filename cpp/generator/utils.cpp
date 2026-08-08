#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdint>
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

std::vector<uint64_t> SampleSeeds(size_t count, uint64_t task_seed) {
    assert(count > 0);
    tools::Random random(task_seed);
    std::vector<uint64_t> seeds;
    seeds.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        seeds.push_back(random.Next());
    }
    return seeds;
}

size_t FullBitId(size_t bit_id, size_t fixed_id) { return bit_id < fixed_id ? bit_id : bit_id + 1; }

float SizeScore(uint16_t bitness, size_t tree_size) {
    const double max_size = std::exp2(static_cast<double>(bitness));
    assert(static_cast<double>(tree_size) < max_size);
    return static_cast<float>(std::log2(max_size - static_cast<double>(tree_size)));
}

}  // namespace gen
