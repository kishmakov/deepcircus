#include "table.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "generator.h"
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

// Each DP state (mask, values) is packed into a single byte:
//   bits 0-1: "seen"  - which output values (0 and/or 1) are still
//             reachable given the bits fixed so far.
//   bits 2-7: "depth" - optimal remaining query depth for that state
//             (valid only once seen == 3, i.e. the value isn't forced yet).
// This halves per-state storage vs. two parallel uint8_t arrays and,
// combined with flattening below, avoids ~2 * 2^bitness separate small
// heap allocations that the original vector<vector<uint8_t>> design paid
// for one per mask.
constexpr uint8_t kSeenMask = 0b11;
constexpr int kDepthShift = 2;

uint8_t MakeDepthCell(uint8_t seen, uint8_t depth) { return seen | (depth << kDepthShift); }
uint8_t DepthOf(uint8_t cell) { return cell >> kDepthShift; }

uint32_t MakeSizeCell(uint8_t seen, uint32_t size) { return static_cast<uint32_t>(seen) | (size << kDepthShift); }
uint32_t SizeOf(uint32_t cell) { return cell >> kDepthShift; }

template <typename Cell>
uint8_t SeenOf(Cell cell) {
    return static_cast<uint8_t>(cell & kSeenMask);
}

// Number of bits already fixed in `mask` below position `bit_id` - i.e.
// the position `bit_id` occupies once fixed bits are stripped out.
uint16_t FixedBitsBefore(uint32_t mask, uint16_t bit_id) {
    return static_cast<uint16_t>(std::popcount(mask & ((uint32_t{1} << bit_id) - 1)));
}

// Inserts `bit_value` as a new bit at `bit_pos` within the packed
// "free bits only" value `values`, shifting higher bits up by one.
size_t InsertFixedBit(size_t values, uint16_t bit_pos, bool bit_value) {
    const size_t lower_mask = (size_t{1} << bit_pos) - 1;
    const size_t lower = values & lower_mask;
    const size_t upper = values & ~lower_mask;
    return lower | (static_cast<size_t>(bit_value) << bit_pos) | (upper << 1);
}

// Everything about a "query this bit next" transition that does NOT
// depend on `values`, hoisted out of the inner loop and computed once
// per mask instead of once per (mask, values) pair.
template <typename Cell>
struct FreeBit {
    uint16_t bit_pos;   // position of this bit among mask's free bits
    const Cell* child;  // pointer to the child mask's state row
};

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
        const size_t depth = SolveForDepth(bitness, table.TruthTable());
        const size_t size = SolveForSize(bitness, table.TruthTable());
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

size_t SolveForDepth(uint16_t bitness, const std::vector<bool>& truth_table) {
    assert(bitness <= kSolvableTableBitness);
    assert(truth_table.size() == (size_t{1} << bitness));

    const uint32_t masks_number = uint32_t{1} << bitness;
    const uint32_t all_bits_mask = masks_number - 1;

    // Bucket masks by popcount once, so each DP level below is a plain
    // scan instead of an O(masks_number) pass with a popcount+skip filter.
    std::vector<std::vector<uint32_t>> masks_by_popcount(bitness + 1);
    for (uint32_t mask = 0; mask < masks_number; ++mask) {
        masks_by_popcount[std::popcount(mask)].push_back(mask);
    }

    // Flatten all (mask, values) states into one contiguous buffer of size
    // exactly sum_mask 2^popcount(mask) == 3^bitness (one slot per ternary
    // assignment: each variable is unqueried / fixed-0 / fixed-1).
    std::vector<size_t> offset(masks_number);
    size_t total_states = 0;
    for (uint32_t mask = 0; mask < masks_number; ++mask) {
        offset[mask] = total_states;
        total_states += size_t{1} << std::popcount(mask);
    }

    std::vector<uint8_t> table(total_states);

    // Base case: every bit fixed, state maps 1:1 onto a truth-table entry.
    for (size_t values = 0; values < truth_table.size(); ++values) {
        table[offset[all_bits_mask] + values] = MakeDepthCell(1u << truth_table[values], 0);
    }

    // Reused across masks to avoid a heap allocation per mask.
    std::vector<FreeBit<uint8_t>> free_bits;
    free_bits.reserve(bitness);

    for (uint16_t fixed_count = bitness; fixed_count-- > 0;) {
        for (uint32_t mask : masks_by_popcount[fixed_count]) {
            const size_t states_number = size_t{1} << fixed_count;
            uint8_t* row = table.data() + offset[mask];

            free_bits.clear();
            for (uint16_t bit_id = 0; bit_id < bitness; ++bit_id) {
                const uint32_t bit = uint32_t{1} << bit_id;
                if (mask & bit) continue;
                const uint32_t child_mask = mask | bit;
                free_bits.push_back({FixedBitsBefore(mask, bit_id), table.data() + offset[child_mask]});
            }

            // Smallest free bit id, used solely to propagate "seen" up from the next level;
            // which free bit is used here doesn't affect correctness since the recursion
            // already folds in every other free bit by induction.
            const uint8_t* seen_child = free_bits.front().child;
            const uint16_t seen_bit_pos = free_bits.front().bit_pos;

            for (size_t values = 0; values < states_number; ++values) {
                const uint8_t state_seen =
                    static_cast<uint8_t>(SeenOf(seen_child[InsertFixedBit(values, seen_bit_pos, false)]) |
                                         SeenOf(seen_child[InsertFixedBit(values, seen_bit_pos, true)]));

                if (state_seen != 3) {
                    // Output already forced given the fixed bits - no more
                    // queries needed.
                    row[values] = MakeDepthCell(state_seen, 0);
                    continue;
                }

                uint8_t best_depth = static_cast<uint8_t>(bitness);
                for (const FreeBit<uint8_t>& fb : free_bits) {
                    const uint8_t child_depth =
                        static_cast<uint8_t>(1 + std::max(DepthOf(fb.child[InsertFixedBit(values, fb.bit_pos, false)]),
                                                          DepthOf(fb.child[InsertFixedBit(values, fb.bit_pos, true)])));
                    best_depth = std::min(best_depth, child_depth);
                    if (best_depth == 1) break;  // depth 1 is the global minimum
                }
                row[values] = MakeDepthCell(3, best_depth);
            }
        }
    }

    return DepthOf(table[offset[0] + 0]);
}

size_t SolveForSize(uint16_t bitness, const std::vector<bool>& truth_table) {
    assert(bitness <= kSolvableTableBitness);
    assert(truth_table.size() == (size_t{1} << bitness));

    const uint32_t masks_number = uint32_t{1} << bitness;
    const uint32_t all_bits_mask = masks_number - 1;

    std::vector<std::vector<uint32_t>> masks_by_popcount(bitness + 1);
    for (uint32_t mask = 0; mask < masks_number; ++mask) {
        masks_by_popcount[std::popcount(mask)].push_back(mask);
    }

    std::vector<size_t> offset(masks_number);
    size_t total_states = 0;
    for (uint32_t mask = 0; mask < masks_number; ++mask) {
        offset[mask] = total_states;
        total_states += size_t{1} << std::popcount(mask);
    }

    std::vector<uint32_t> table(total_states);

    for (size_t values = 0; values < truth_table.size(); ++values) {
        table[offset[all_bits_mask] + values] = MakeSizeCell(1u << truth_table[values], 0);
    }

    std::vector<FreeBit<uint32_t>> free_bits;
    free_bits.reserve(bitness);

    for (uint16_t fixed_count = bitness; fixed_count-- > 0;) {
        for (uint32_t mask : masks_by_popcount[fixed_count]) {
            const size_t states_number = size_t{1} << fixed_count;
            uint32_t* row = table.data() + offset[mask];

            free_bits.clear();
            for (uint16_t bit_id = 0; bit_id < bitness; ++bit_id) {
                const uint32_t bit = uint32_t{1} << bit_id;
                if (mask & bit) continue;
                const uint32_t child_mask = mask | bit;
                free_bits.push_back({FixedBitsBefore(mask, bit_id), table.data() + offset[child_mask]});
            }

            const uint32_t* seen_child = free_bits.front().child;
            const uint16_t seen_bit_pos = free_bits.front().bit_pos;

            for (size_t values = 0; values < states_number; ++values) {
                const uint8_t state_seen =
                    static_cast<uint8_t>(SeenOf(seen_child[InsertFixedBit(values, seen_bit_pos, false)]) |
                                         SeenOf(seen_child[InsertFixedBit(values, seen_bit_pos, true)]));

                if (state_seen != 3) {
                    row[values] = MakeSizeCell(state_seen, 0);
                    continue;
                }

                uint32_t best_size = std::numeric_limits<uint32_t>::max();
                for (const FreeBit<uint32_t>& fb : free_bits) {
                    const uint32_t child_size = 1 + SizeOf(fb.child[InsertFixedBit(values, fb.bit_pos, false)]) +
                                                SizeOf(fb.child[InsertFixedBit(values, fb.bit_pos, true)]);
                    best_size = std::min(best_size, child_size);
                    if (best_size == 1) break;
                }
                row[values] = MakeSizeCell(3, best_size);
            }
        }
    }

    return SizeOf(table[offset[0] + 0]);
}


}  // namespace gen
