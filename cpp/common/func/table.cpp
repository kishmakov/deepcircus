#include "func/table.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <random>
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

constexpr uint16_t kTruthTableBitness = 16;

// A uniformly random truth table, one rng word at a time.
std::vector<bool> TruthTable(uint16_t bitness, std::mt19937& rng) {
    assert(bitness >= kMinTableBitness && bitness <= kTruthTableBitness);

    std::vector<bool> truth_table(size_t{1} << bitness);
    size_t input_id = 0;
    while (input_id < truth_table.size()) {
        const uint32_t chunk = rng();
        for (size_t bit = 0; bit < 32 && input_id < truth_table.size(); ++bit) {
            truth_table[input_id] = ((chunk >> bit) & 1u) != 0;
            ++input_id;
        }
    }
    return truth_table;
}

size_t BitsToNum(const std::vector<bool>& input) {
    assert(input.size() <= kTruthTableBitness);

    size_t id = 0;
    for (size_t bit_id = 0; bit_id < input.size(); ++bit_id) {
        if (input[bit_id]) {
            id |= size_t{1} << bit_id;
        }
    }
    return id;
}

bool SparseTableValue(uint16_t bitness, uint64_t base_seed, const std::vector<bool>& input) {
    assert(input.size() == bitness);
    uint64_t value = base_seed;
    for (bool bit : input) {
        value ^= static_cast<uint64_t>(bit);
        value *= 0x100000001b3ull;
    }
    return (tools::Mix64(value) & 1ull) != 0;
}

}  // namespace

TableCase::TableCase(uint16_t bitness, uint64_t seed) : gen::Case(bitness, seed) {
    assert(bitness_ >= kMinTableBitness && bitness_ <= kMaxTableBitness);
    if (bitness_ <= kTruthTableBitness) {
        // Qualified: the member TruthTable() would otherwise hide the drawing
        // function at class scope.
        truth_table_ = func::TruthTable(bitness_, rng_);
    }
    // Above that the function is a hash keyed on the seed itself, so there is
    // nothing to draw and nothing to store.
}

bool TableCase::Evaluate(const std::vector<bool>& input) const {
    assert(input.size() == bitness_);
    if (truth_table_.has_value()) {
        return (*truth_table_)[BitsToNum(input)];
    }
    return SparseTableValue(bitness_, seed_, input);
}

const std::vector<bool>& TableCase::TruthTable() const {
    assert(truth_table_.has_value());
    return *truth_table_;
}

std::string TableValue(uint16_t bitness, uint64_t seed, const std::vector<bool>& input) {
    assert(bitness >= kMinTableBitness && bitness <= kMaxTableBitness);
    assert(input.size() >= bitness);

    const TableCase table(bitness, seed);
    const std::vector<bool> point(input.begin(), input.begin() + bitness);
    return table.SampledValueString(point);
}

std::vector<uint64_t> TableSampleSeeds(uint16_t bitness, size_t cases, uint64_t task_seed) {
    return gen::SampleSeeds(cases, gen::DomainSeed(task_seed, kTableSelectionDomain, bitness));
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
        const size_t depth = tools::SolveForDepth(bitness, table.TruthTable());
        const size_t size = tools::SolveForSize(bitness, table.TruthTable());
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
