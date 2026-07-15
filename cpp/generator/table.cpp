#include "table.h"

#include <cassert>
#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "generator.h"
#include "tree.h"
#include "utils.h"

namespace {

constexpr size_t kTableCasesNumber = size_t{1} << 32;

// Seed-domain tag for table value-input streams, shared by dataset
// generation and sparse table evaluation.
constexpr uint64_t kTableValueDomain = 0x7461626c655f7661ull;
constexpr uint64_t kTableSelectionDomain = 0x7461626c655f7365ull;
constexpr uint64_t kRestrictionDomain = 0x7265737472696374ull;
constexpr uint64_t kTableStructureDomain = 0x7461626c655f7374ull;

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

size_t CharsToNum(std::string_view input) {
    assert(input.size() <= kSolvableTableBitness);

    size_t id = 0;
    for (size_t bit_id = 0; bit_id < input.size(); ++bit_id) {
        assert(input[bit_id] == '0' || input[bit_id] == '1');
        if (input[bit_id] == '1') {
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

bool SparseTableValue(uint16_t bitness, uint64_t base_seed, std::string_view input) {
    assert(input.size() == bitness);
    uint64_t value = base_seed;
    for (char bit : input) {
        assert(bit == '0' || bit == '1');
        value ^= static_cast<uint64_t>(bit);
        value *= 0x100000001b3ull;
    }
    return (Mix64(value) & 1ull) != 0;
}

}  // namespace

TableCase::TableCase(uint16_t bitness, size_t case_id, uint64_t seed)
    : Case(bitness, case_id, DomainSeed(seed, kTableStructureDomain, bitness)) {
    assert(bitness_ >= kMinTableBitness && bitness_ <= kMaxTableBitness);
    assert(case_id_ < gen::TableCasesNumber(bitness_));
    if (bitness_ <= kSolvableTableBitness) {
        truth_table_ = SolvableTableVector(bitness_, case_id_, RNG());
    } else {
        sparse_seed_ = static_cast<uint64_t>(RNG()()) | (static_cast<uint64_t>(RNG()()) << 32);
    }
}

bool TableCase::Evaluate(std::string_view input) const {
    assert(input.size() == bitness_);
    if (bitness_ <= kSolvableTableBitness) {
        return truth_table_[CharsToNum(input)];
    }
    return SparseTableValue(bitness_, sparse_seed_, input);
}

void TableCase::FillValueTensor(size_t reps, uint64_t seed, float* out) const {
    const auto evaluate = [this](std::string_view input) { return Evaluate(input); };
    FillGeneratedValueTensor(bitness_, reps, seed, out, evaluate);
}

void TableCase::FillRestrictionsTensor(size_t reps, uint64_t seed, float* out) const {
    const auto evaluate = [this](std::string_view input) { return Evaluate(input); };
    FillGeneratedRestrictionsTensor(bitness_, reps, seed, out, evaluate);
}

const std::vector<bool>& TableCase::TruthTable() const {
    assert(bitness_ <= kSolvableTableBitness);
    return truth_table_;
}

namespace gen {

size_t TableCasesNumber(uint16_t bitness) {
    assert(bitness >= kMinTableBitness && bitness <= kMaxTableBitness);
    if (IsSmallBitness(bitness)) {
        return SmallBitnessCasesNumber(bitness);
    }
    return kTableCasesNumber;
}

uint16_t TableSolvableBitness() { return kSolvableTableBitness; }

std::string TableValue(uint16_t bitness, size_t case_id, uint64_t seed, std::string_view input) {
    assert(bitness >= kMinTableBitness && bitness <= kMaxTableBitness);
    assert(case_id < TableCasesNumber(bitness));
    assert(input.size() >= bitness);

    const TableCase table(bitness, case_id, seed);
    const auto evaluate = [&table](std::string_view point) { return table.Evaluate(point); };
    return SampledValueString(bitness, input.substr(0, bitness), evaluate);
}

std::vector<size_t> TableSampleCaseIds(uint16_t bitness, size_t cases, uint64_t seed) {
    return SampleCaseIds(TableCasesNumber(bitness), cases, DomainSeed(seed, kTableSelectionDomain, bitness));
}

GeneratedValues TableValuesForCases(uint16_t bitness, const std::vector<size_t>& case_ids, size_t reps, uint64_t seed) {
    const size_t cases = case_ids.size();
    assert(cases > 0);
    assert(reps > 0);
    assert(reps % 2 == 0);
    assert(bitness >= kMinTableBitness);
    assert(bitness <= kSolvableTableBitness);

    const size_t sample_size = 2 * bitness + 1;
    std::vector<std::vector<bool>> values(cases, std::vector<bool>(reps * sample_size));
    std::vector<float> targets(gen::kTargetsPerCase * cases);
    const uint64_t value_seed = DomainSeed(seed, kTableValueDomain, bitness);

    std::vector<float> case_values(reps * sample_size);
    for (size_t case_index = 0; case_index < cases; ++case_index) {
        const size_t case_id = case_ids[case_index];
        TableCase table(bitness, case_id, seed);
        table.FillValueTensor(reps, CaseInputSeed(value_seed, bitness, case_id), case_values.data());
        const size_t depth = SolveForDepth(bitness, table.TruthTable());
        const size_t size = SolveForSize(bitness, table.TruthTable());
        targets[gen::kTargetsPerCase * case_index] = static_cast<float>(bitness - depth);
        targets[gen::kTargetsPerCase * case_index + 1] = SizeScore(bitness, size);
        StoreBits(values[case_index], case_values);
    }

    return GeneratedValues{Values(std::move(values)), std::move(targets)};
}

GeneratedRestrictions TableRestrictionsForCases(uint16_t bitness, const std::vector<size_t>& case_ids, size_t reps,
                                                uint64_t seed) {
    const size_t cases = case_ids.size();
    assert(cases > 0);
    assert(reps > 0);
    assert(reps % 2 == 0);
    assert(bitness > kSolvableTableBitness);
    assert(bitness <= kMaxTableBitness);

    const size_t sample_size = 2 * bitness + 1;
    const size_t restriction_size = 2 * bitness * reps * (2 * bitness - 1);
    std::vector<std::vector<bool>> values(cases, std::vector<bool>(reps * sample_size));
    std::vector<std::vector<bool>> restrictions(cases, std::vector<bool>(restriction_size));
    const uint64_t value_seed = DomainSeed(seed, kTableValueDomain, bitness);
    const uint64_t restriction_seed = DomainSeed(seed, kRestrictionDomain, bitness);

    std::vector<float> case_values(reps * sample_size);
    std::vector<float> restriction_values(restriction_size);
    for (size_t case_index = 0; case_index < cases; ++case_index) {
        const size_t case_id = case_ids[case_index];
        TableCase table(bitness, case_id, seed);
        table.FillValueTensor(reps, CaseInputSeed(value_seed, bitness, case_id), case_values.data());
        table.FillRestrictionsTensor(reps, CaseInputSeed(restriction_seed, bitness, case_id),
                                     restriction_values.data());
        StoreBits(values[case_index], case_values);
        StoreBits(restrictions[case_index], restriction_values);
    }

    return GeneratedRestrictions{Values(std::move(values)), Restrictions(std::move(restrictions))};
}

}  // namespace gen
