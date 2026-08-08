#pragma once

#include <stddef.h>
#include <stdint.h>

#include <vector>

namespace gen {

uint64_t TaskSeed(uint64_t seed, uint16_t bitness, uint64_t iteration);
uint64_t DomainSeed(uint64_t seed, uint64_t domain, uint16_t bitness);

// Per-case seeds: `count` independent draws off `task_seed`. Each seed depends
// only on its index, so splitting the result into contiguous chunks and
// generating each separately reproduces one call over the whole list.
std::vector<uint64_t> SampleSeeds(size_t count, uint64_t task_seed);

size_t FullBitId(size_t bit_id, size_t fixed_id);

// Size training target: log2(2^bitness - tree_size), where tree_size counts
// internal nodes. A constant function scores bitness, a full tree scores 0.
float SizeScore(uint16_t bitness, size_t tree_size);

}  // namespace gen
