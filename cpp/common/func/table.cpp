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

uint64_t DeserializeSeed(const std::vector<uint8_t>& bytes) {
    assert(bytes.size() == sizeof(uint64_t));

    uint64_t seed = 0;
    std::memcpy(&seed, bytes.data(), sizeof(seed));
    return seed;
}

}  // namespace

TableFunc::TableFunc(uint16_t bitness, uint64_t seed) : func::Func(bitness), seed_(seed) {
    assert(bitness_ >= kMinBitness && bitness_ <= kMaxBitness);
}

TableFunc::TableFunc(uint16_t bitness, std::vector<uint8_t> bytes)
    : func::Func(bitness), seed_(DeserializeSeed(bytes)) {
    assert(bitness_ >= kMinBitness && bitness_ <= kMaxBitness);
}

bool TableFunc::operator()(const FuncInput& input) const {
    assert(input.size() == bitness_);
    return tools::RandomFuncValue(bitness_, seed_, input);
}

std::vector<uint8_t> TableFunc::serialize() const {
    std::vector<uint8_t> bytes(sizeof(seed_));
    std::memcpy(bytes.data(), &seed_, sizeof(seed_));
    return bytes;
}

std::string TableValue(uint16_t bitness, uint64_t seed, const std::vector<bool>& input) {
    assert(bitness >= kMinBitness && bitness <= kMaxBitness);
    assert(input.size() >= bitness);

    const TableFunc table(bitness, seed);
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
    assert(bitness >= kMinBitness);
    assert(bitness <= tools::kMaxSolvableBitness);

    const size_t sample_size = 2 * bitness + 1;
    const size_t columns = static_cast<size_t>(shape.batches) * shape.batch_size * sample_size;
    std::vector<bool> values(cases * columns);
    std::vector<float> targets(gen::kTargetsPerCase * cases);

    for (size_t case_index = 0; case_index < cases; ++case_index) {
        TableFunc table(bitness, seeds[case_index]);
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
    assert(bitness <= kMaxBitness);

    const size_t points = static_cast<size_t>(shape.batches) * shape.batch_size;
    const size_t columns = points * (2 * bitness + 1);
    const size_t restriction_size = 2 * bitness * points * (2 * (bitness - 1) + 1);
    std::vector<bool> values(cases * columns);
    std::vector<bool> restrictions(cases * restriction_size);

    for (size_t case_index = 0; case_index < cases; ++case_index) {
        TableFunc table(bitness, seeds[case_index]);

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
