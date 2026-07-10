#include "bool_bench.h"
#include "decision_tree.h"
#include "table.h"
#include "utils.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr size_t kTableCasesNumber = size_t{1} << 32;

constexpr uint16_t kMinSmallBitness = 4;
constexpr uint16_t kMaxSmallBitness = 6;

constexpr uint16_t kMinMediumBitness = 7;
constexpr uint16_t kMaxMediumBitness = 16;


bool IsSmallBitness(uint16_t bitness) {
    return kMinSmallBitness <= bitness && bitness <= kMaxSmallBitness;
}

size_t SmallBitnessCasesNumber(uint16_t bitness) {
    assert(IsSmallBitness(bitness));
    switch (bitness) {
        case 4: return 0x10000ull;
        case 5: return 0xffffffffull;
        case 6: return 0xffffffffull;
        default: assert(false); return 0;
    }
}

bool IsMediumBitness(uint16_t bitness) {
    return kMinMediumBitness <= bitness && bitness <= kMaxMediumBitness;
}

size_t MediumBitnessCasesNumber(uint16_t bitness) {
    assert(IsMediumBitness(bitness));
    return kTableCasesNumber;
}

std::vector<bool> MediumBitnessTruthTable(uint16_t bitness, size_t case_id) {
    assert(IsMediumBitness(bitness));
    assert(case_id < MediumBitnessCasesNumber(bitness));

    std::vector<bool> truth_table(size_t{1} << bitness);
    std::seed_seq seed{
        static_cast<uint32_t>(bitness),
        static_cast<uint32_t>(case_id),
    };
    std::mt19937_64 rng(seed);

    size_t input_id = 0;
    while (input_id < truth_table.size()) {
        const uint64_t chunk = rng();
        for (size_t bit = 0; bit < 64 && input_id < truth_table.size(); ++bit) {
            truth_table[input_id] = ((chunk >> bit) & 1ull) != 0;
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

std::vector<bool> SolvableTableVector(uint16_t bitness, size_t case_id) {
    assert(bitness >= kMinTableBitness && bitness <= kSolvableTableBitness);
    assert(case_id < bb_table_cases_number(bitness));

    if (IsSmallBitness(bitness)) {
        return SmallTableVector(bitness, case_id);
    }
    return MediumBitnessTruthTable(bitness, case_id);
}

uint64_t Mix64(uint64_t value) {
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

bool SparseTableValue(
    uint16_t bitness,
    size_t case_id,
    std::string_view input)
{
    assert(input.size() == bitness);
    uint64_t value = CaseInputSeed(0x7461626c655f7661ull, bitness, case_id);
    for (char bit : input) {
        assert(bit == '0' || bit == '1');
        value ^= static_cast<uint64_t>(bit);
        value *= 0x100000001b3ull;
    }
    return (Mix64(value) & 1ull) != 0;
}

}  // namespace

TableCase::TableCase(uint16_t bitness, size_t case_id)
    : bitness_(bitness)
    , case_id_(case_id)
{
    assert(bitness_ >= kMinTableBitness && bitness_ <= kMaxTableBitness);
    assert(case_id_ < bb_table_cases_number(bitness_));
    if (bitness_ <= kSolvableTableBitness) {
        truth_table_ = SolvableTableVector(bitness_, case_id_);
    }
}

bool TableCase::Evaluate(std::string_view input) const {
    assert(input.size() == bitness_);
    if (bitness_ <= kSolvableTableBitness) {
        return truth_table_[CharsToNum(input)];
    }
    return SparseTableValue(bitness_, case_id_, input);
}

void TableCase::FillValueTensor(size_t reps, uint64_t seed, float* out) const {
    const auto evaluate = [this](std::string_view input) {
        return Evaluate(input);
    };
    FillGeneratedValueTensor(bitness_, reps, seed, out, evaluate);
}

void TableCase::FillRestrictionsTensor(
    size_t reps,
    uint64_t seed,
    float* out) const
{
    const auto evaluate = [this](std::string_view input) {
        return Evaluate(input);
    };
    FillGeneratedRestrictionsTensor(bitness_, reps, seed, out, evaluate);
}

const std::vector<bool>& TableCase::TruthTable() const {
    assert(bitness_ <= kSolvableTableBitness);
    return truth_table_;
}

size_t bb_table_cases_number(uint16_t bitness) {
    assert(bitness >= kMinTableBitness && bitness <= kMaxTableBitness);
    if (IsSmallBitness(bitness)) {
        return SmallBitnessCasesNumber(bitness);
    }
    if (IsMediumBitness(bitness)) {
        return MediumBitnessCasesNumber(bitness);
    }
    return kTableCasesNumber;
}

uint16_t bb_table_solvable_bitness() {
    return kSolvableTableBitness;
}

const char* bb_table_value(uint16_t bitness, size_t case_id, const char* input) {
    assert(bitness >= kMinTableBitness && bitness <= kMaxTableBitness);
    assert(case_id < bb_table_cases_number(bitness));
    assert(input != nullptr);
    assert(std::strlen(input) >= bitness);

    thread_local std::string value;
    thread_local FlippingSampler sampler;

    value.assign(2 * bitness + 1, '0');
    sampler.Reset(bitness, {input, bitness});

    const TableCase table(bitness, case_id);
    const auto evaluate = [&table](std::string_view input) {
        return table.Evaluate(input);
    };
    sampler.Fill(
        value,
        /*sample_offset=*/0,
        bitness,
        evaluate);

    return value.c_str();
}
