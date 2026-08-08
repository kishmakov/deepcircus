#pragma once

#include <stddef.h>
#include <stdint.h>

#include <vector>

namespace gen {

uint64_t TaskSeed(uint64_t seed, uint16_t bitness, uint64_t iteration);
uint64_t DomainSeed(uint64_t seed, uint64_t domain, uint16_t bitness);

// Deterministic, chunk-order-independent sample of `cases` distinct ids from
// [0, population).
std::vector<size_t> SampleCaseIds(size_t population, size_t cases, uint64_t seed);

size_t FullBitId(size_t bit_id, size_t fixed_id);

// Size training target: log2(2^bitness - tree_size), where tree_size counts
// internal nodes. A constant function scores bitness, a full tree scores 0.
float SizeScore(uint16_t bitness, size_t tree_size);

}  // namespace gen
