#include "table.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "generator.h"
#include "tools/solver.h"
#include "tree.h"
#include "utils.h"

namespace gen {

namespace {

constexpr size_t kTableCasesNumber = size_t{1} << 32;

constexpr uint64_t kTableSelectionDomain = 0x7461626c655f7365ull;

constexpr uint16_t kMaxSmallBitness = 6;

bool IsSmallBitness(uint16_t bitness) { return kMinTableBitness <= bitness && bitness <= kMaxSmallBitness; }

size_t SmallBitnessCasesNumber(uint16_t bitness) {
    assert(IsSmallBitness(bitness));
    switch (bitness) {
        case 4:
            return 0x10000ull;
        case 5:
            return 0xffffffffull;
        case 6:
            return 0xffffffffull;
        default:
            assert(false);
            return 0;
    }
}

bool IsMediumBitness(uint16_t bitness) { return kMaxSmallBitness < bitness && bitness <= kSolvableTableBitness; }

std::vector<bool> MediumBitnessTruthTable(uint16_t bitness, std::mt19937& rng) {
    assert(IsMediumBitness(bitness));

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
    assert(input.size() <= kSolvableTableBitness);

    size_t id = 0;
    for (size_t bit_id = 0; bit_id < input.size(); ++bit_id) {
        if (input[bit_id]) {
            id |= size_t{1} << bit_id;
        }
    }
    return id;
}

std::vector<bool> SmallTableVector(uint16_t bitness, size_t case_id) {
    assert(IsSmallBitness(bitness));
    assert(case_id < SmallBitnessCasesNumber(bitness));

    std::vector<bool> table(size_t{1} << bitness);
    for (size_t input_id = 0; input_id < table.size(); ++input_id) {
        table[input_id] = ((case_id >> input_id) & 1ull) != 0;
    }
    return table;
}

std::vector<bool> SolvableTableVector(uint16_t bitness, size_t case_id, std::mt19937& rng) {
    assert(bitness >= kMinTableBitness && bitness <= kSolvableTableBitness);
    assert(case_id < gen::TableCasesNumber(bitness));

    if (IsSmallBitness(bitness)) {
        return SmallTableVector(bitness, case_id);
    }
    return MediumBitnessTruthTable(bitness, rng);
}

bool SparseTableValue(uint16_t bitness, uint64_t base_seed, const std::vector<bool>& input) {
    assert(input.size() == bitness);
    uint64_t value = base_seed;
    for (bool bit : input) {
        value ^= static_cast<uint64_t>(bit);
        value *= 0x100000001b3ull;
    }
    return (Mix64(value) & 1ull) != 0;
}

}  // namespace

TableCase::TableCase(uint16_t bitness, size_t case_id) : Case(bitness, case_id) {
    assert(bitness_ >= kMinTableBitness && bitness_ <= kMaxTableBitness);
    assert(case_id_ < gen::TableCasesNumber(bitness_));
    if (bitness_ <= kSolvableTableBitness) {
        truth_table_ = SolvableTableVector(bitness_, case_id_, rng_);
    } else {
        sparse_seed_ = static_cast<uint64_t>(rng_()) | (static_cast<uint64_t>(rng_()) << 32);
    }
}

bool TableCase::Evaluate(const std::vector<bool>& input) const {
    assert(input.size() == bitness_);
    if (bitness_ <= kSolvableTableBitness) {
        return truth_table_[BitsToNum(input)];
    }
    return SparseTableValue(bitness_, sparse_seed_, input);
}

const std::vector<bool>& TableCase::TruthTable() const {
    assert(bitness_ <= kSolvableTableBitness);
    return truth_table_;
}

size_t TableCasesNumber(uint16_t bitness) {
    assert(bitness >= kMinTableBitness && bitness <= kMaxTableBitness);
    if (IsSmallBitness(bitness)) {
        return SmallBitnessCasesNumber(bitness);
    }
    return kTableCasesNumber;
}

uint16_t TableSolvableBitness() { return kSolvableTableBitness; }

std::string TableValue(uint16_t bitness, size_t case_id, const std::vector<bool>& input) {
    assert(bitness >= kMinTableBitness && bitness <= kMaxTableBitness);
    assert(case_id < TableCasesNumber(bitness));
    assert(input.size() >= bitness);

    const TableCase table(bitness, case_id);
    const std::vector<bool> point(input.begin(), input.begin() + bitness);
    return table.SampledValueString(point);
}

std::vector<size_t> TableSampleCaseIds(uint16_t bitness, size_t cases, uint64_t seed) {
    return SampleCaseIds(TableCasesNumber(bitness), cases, DomainSeed(seed, kTableSelectionDomain, bitness));
}

GeneratedValues TableValuesForCases(uint16_t bitness, const std::vector<size_t>& case_ids, InputShape shape) {
    const size_t cases = case_ids.size();
    assert(cases > 0);
    assert(shape.batches > 1);
    assert(std::has_single_bit(shape.batch_size));
    assert(bitness >= kMinTableBitness);
    assert(bitness <= kSolvableTableBitness);

    const size_t sample_size = 2 * bitness + 1;
    const size_t columns = static_cast<size_t>(shape.batches) * shape.batch_size * sample_size;
    std::vector<bool> values(cases * columns);
    std::vector<float> targets(kTargetsPerCase * cases);

    for (size_t case_index = 0; case_index < cases; ++case_index) {
        const size_t case_id = case_ids[case_index];
        TableCase table(bitness, case_id);
        const std::vector<bool> samples = table.SampleValues(shape);
        assert(samples.size() == columns);
        std::copy(samples.begin(), samples.end(), values.begin() + case_index * columns);
        const size_t depth = tools::SolveForDepth(bitness, table.TruthTable());
        const size_t size = tools::SolveForSize(bitness, table.TruthTable());
        targets[gen::kTargetsPerCase * case_index] = static_cast<float>(bitness - depth);
        targets[gen::kTargetsPerCase * case_index + 1] = SizeScore(bitness, size);
    }

    return GeneratedValues{Values(cases, columns, std::move(values)), std::move(targets)};
}

GeneratedRestrictions TableRestrictionsForCases(uint16_t bitness, const std::vector<size_t>& case_ids, InputShape shape) {
    const size_t cases = case_ids.size();
    assert(cases > 0);
    assert(shape.batches > 1);
    assert(std::has_single_bit(shape.batch_size));
    assert(bitness > kSolvableTableBitness);
    assert(bitness <= kMaxTableBitness);

    const size_t points = static_cast<size_t>(shape.batches) * shape.batch_size;
    const size_t columns = points * (2 * bitness + 1);
    const size_t restriction_size = 2 * bitness * points * (2 * (bitness - 1) + 1);
    std::vector<bool> values(cases * columns);
    std::vector<bool> restrictions(cases * restriction_size);

    for (size_t case_index = 0; case_index < cases; ++case_index) {
        const size_t case_id = case_ids[case_index];
        TableCase table(bitness, case_id);

        const std::vector<bool> case_values = table.SampleValues(shape);
        assert(case_values.size() == columns);
        std::copy(case_values.begin(), case_values.end(), values.begin() + case_index * columns);

        const std::vector<bool> case_restrictions = table.SampleRestrictions(shape);
        assert(case_restrictions.size() == restriction_size);
        std::copy(case_restrictions.begin(), case_restrictions.end(),
                  restrictions.begin() + case_index * restriction_size);
    }

    return GeneratedRestrictions{Values(cases, columns, std::move(values)),
                                 Restrictions(cases, restriction_size, std::move(restrictions))};
}

}  // namespace gen
