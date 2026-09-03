#include "func/table.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "generator.h"
#include "tools/random.h"
#include "tools/solver.h"
#include "utils.h"

namespace func {

namespace {

constexpr uint64_t kTableSelectionDomain = 0x7461626c655f7365ull;

// Widest table worth materializing.
constexpr uint16_t kMaxMaterializedBitness = 16;

uint64_t DeserializeSeed(const std::vector<uint8_t>& bytes) {
    assert(bytes.size() == sizeof(uint64_t));

    uint64_t seed = 0;
    std::memcpy(&seed, bytes.data(), sizeof(seed));
    return seed;
}

// The seed's function at one point. The fold stops after the last set bit, so
// trailing zeros cost a point nothing and it keeps its value however wide the
// function around it is -- the narrow truth table stays the prefix of every
// wider one.
bool HashedValue(uint64_t seed, const std::vector<bool>& input) {
    size_t bits = input.size();
    while (bits > 0 && !input[bits - 1]) {
        --bits;
    }

    uint64_t value = seed;
    for (size_t bit_id = 0; bit_id < bits; ++bit_id) {
        value ^= static_cast<uint64_t>(input[bit_id]);
        value *= 0x100000001b3ull;
    }
    return (tools::Mix64(value) & 1ull) != 0;
}

}  // namespace

TableCase::TableCase(uint16_t bitness, uint64_t seed) : func::Func(bitness), seed_(seed) {
    assert(bitness_ >= kMinTableBitness && bitness_ <= kMaxTableBitness);
}

TableCase::TableCase(uint16_t bitness, std::vector<uint8_t> bytes)
    : func::Func(bitness), seed_(DeserializeSeed(bytes)) {
    assert(bitness_ >= kMinTableBitness && bitness_ <= kMaxTableBitness);
}

bool TableCase::operator()(const FuncInput& input) const {
    assert(input.size() == bitness_);
    return HashedValue(seed_, input);
}

std::vector<uint8_t> TableCase::serialize() const {
    std::vector<uint8_t> bytes(sizeof(seed_));
    std::memcpy(bytes.data(), &seed_, sizeof(seed_));
    return bytes;
}

std::vector<bool> TableCase::TruthTable() const {
    assert(bitness_ <= kMaxMaterializedBitness);

    std::vector<bool> truth_table(size_t{1} << bitness_);
    FuncInput input(bitness_);
    for (size_t id = 0; id < truth_table.size(); ++id) {
        for (uint16_t bit_id = 0; bit_id < bitness_; ++bit_id) {
            input[bit_id] = ((id >> bit_id) & 1u) != 0;
        }
        truth_table[id] = HashedValue(seed_, input);
    }
    return truth_table;
}

std::string TableValue(uint16_t bitness, uint64_t seed, const std::vector<bool>& input) {
    assert(bitness >= kMinTableBitness && bitness <= kMaxTableBitness);
    assert(input.size() >= bitness);

    const TableCase table(bitness, seed);
    const std::vector<bool> point(input.begin(), input.begin() + bitness);
    return table.SampledValueString(point);
}

std::vector<uint64_t> TableSampleSeeds(uint16_t bitness, size_t cases, uint64_t task_seed) {
    return tools::SampleSeeds(cases, tools::DomainSeed(task_seed, kTableSelectionDomain, bitness));
}

gen::GeneratedValues TableValuesForSeeds(uint16_t bitness, const std::vector<uint64_t>& seeds, gen::InputShape shape) {
    const size_t cases = seeds.size();
    assert(cases > 0);
    assert(shape.batches > 1);
    assert(std::has_single_bit(shape.batch_size));
    assert(bitness >= kMinTableBitness);
    assert(bitness <= tools::kMaxSolvableBitness);

    const size_t sample_size = 2 * bitness + 1;
    const size_t columns = static_cast<size_t>(shape.batches) * shape.batch_size * sample_size;
    std::vector<bool> values(cases * columns);
    std::vector<float> targets(gen::kTargetsPerCase * cases);

    for (size_t case_index = 0; case_index < cases; ++case_index) {
        TableCase table(bitness, seeds[case_index]);
        const std::vector<bool> samples = table.SampleValues(shape);
        assert(samples.size() == columns);
        std::copy(samples.begin(), samples.end(), values.begin() + case_index * columns);
        // Built once: the table is no longer kept on the case.
        const std::vector<bool> truth_table = table.TruthTable();
        const size_t depth = tools::SolveForDepth(bitness, truth_table);
        const size_t size = tools::SolveForSize(bitness, truth_table);
        targets[gen::kTargetsPerCase * case_index] = static_cast<float>(bitness - depth);
        targets[gen::kTargetsPerCase * case_index + 1] = gen::SizeScore(bitness, size);
    }

    return gen::GeneratedValues{gen::Values(cases, columns, std::move(values)), std::move(targets)};
}

gen::GeneratedRestrictions TableRestrictionsForSeeds(uint16_t bitness, const std::vector<uint64_t>& seeds,
                                                     gen::InputShape shape) {
    const size_t cases = seeds.size();
    assert(cases > 0);
    assert(shape.batches > 1);
    assert(std::has_single_bit(shape.batch_size));
    assert(bitness > tools::kMaxSolvableBitness);
    assert(bitness <= kMaxTableBitness);

    const size_t points = static_cast<size_t>(shape.batches) * shape.batch_size;
    const size_t columns = points * (2 * bitness + 1);
    const size_t restriction_size = 2 * bitness * points * (2 * (bitness - 1) + 1);
    std::vector<bool> values(cases * columns);
    std::vector<bool> restrictions(cases * restriction_size);

    for (size_t case_index = 0; case_index < cases; ++case_index) {
        TableCase table(bitness, seeds[case_index]);

        const std::vector<bool> case_values = table.SampleValues(shape);
        assert(case_values.size() == columns);
        std::copy(case_values.begin(), case_values.end(), values.begin() + case_index * columns);

        const std::vector<bool> case_restrictions = table.SampleRestrictions(shape);
        assert(case_restrictions.size() == restriction_size);
        std::copy(case_restrictions.begin(), case_restrictions.end(),
                  restrictions.begin() + case_index * restriction_size);
    }

    return gen::GeneratedRestrictions{gen::Values(cases, columns, std::move(values)),
                                      gen::Restrictions(cases, restriction_size, std::move(restrictions))};
}

}  // namespace func
