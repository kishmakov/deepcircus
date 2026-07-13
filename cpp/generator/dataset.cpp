#include "decision_tree.h"
#include "generator.h"
#include "table.h"
#include "tree.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

    constexpr uint64_t kTreeSelectionDomain = 0x747265655f73656cull;
    constexpr uint64_t kTableSelectionDomain = 0x7461626c655f7365ull;
    constexpr uint64_t kTreeValueDomain = 0x747265655f76616cull;
    constexpr uint64_t kTableValueDomain = 0x7461626c655f7661ull;
    constexpr uint64_t kRestrictionDomain = 0x7265737472696374ull;

    enum class GeneratorKind {
        Tree,
        Table,
    };

    uint64_t DomainSeed(uint64_t seed, uint64_t domain, uint16_t bitness) {
        uint64_t state = seed ^ domain ^ (static_cast<uint64_t>(bitness) << 48);
        return SplitMix64(state);
    }

    class SplitMixGenerator {
    public:
        explicit SplitMixGenerator(uint64_t seed) : state_(seed) {}

        uint64_t Generate() { return SplitMix64(state_); }

        uint64_t GenerateBelow(uint64_t bound) {
            assert(bound > 0);
            const uint64_t threshold = static_cast<uint64_t>(-bound) % bound;
            while (true) {
                const uint64_t value = Generate();
                if (value >= threshold) {
                    return value % bound;
                }
            }
        }

    private:
        uint64_t state_;
    };

    std::vector<size_t> SampleCaseIds(size_t population, size_t cases, uint64_t seed) {
        assert(cases > 0);
        assert(cases <= population);
        SplitMixGenerator rng(seed);
        std::unordered_set<size_t> selected;
        selected.reserve(cases * 2);
        std::vector<size_t> result;
        result.reserve(cases);

        for (size_t current = population - cases; current < population; ++current) {
            const size_t candidate = static_cast<size_t>(rng.GenerateBelow(static_cast<uint64_t>(current) + 1));
            const size_t case_id = selected.contains(candidate) ? current : candidate;
            const bool inserted = selected.insert(case_id).second;
            assert(inserted);
            result.push_back(case_id);
        }
        return result;
    }

    // Generated +/-1 values stay bit-packed until a caller requests float output.
    void StoreBits(std::vector<bool> &destination, const std::vector<float> &source) {
        assert(destination.size() == source.size());
        for (size_t index = 0; index < source.size(); ++index) {
            assert(source[index] == -1.0f || source[index] == 1.0f);
            destination[index] = source[index] > 0.0f;
        }
    }

    void WriteBits(const std::vector<std::vector<bool>> &data, float *output) {
        assert(output != nullptr);
        size_t offset = 0;
        for (const std::vector<bool> &row: data) {
            for (bool value: row) {
                output[offset++] = value ? 1.0f : -1.0f;
            }
        }
    }

    gen::GeneratedValues MakeValues(GeneratorKind kind, uint16_t bitness, const std::vector<size_t> &case_ids,
                                    size_t reps, uint64_t seed) {
        const size_t cases = case_ids.size();
        assert(cases > 0);
        assert(reps > 0);
        assert(reps % 2 == 0);
        if (kind == GeneratorKind::Tree) {
            assert(bitness >= kMinTreeBitness && bitness <= kMaxTreeBitness);
        } else {
            assert(bitness >= kMinTableBitness);
            assert(bitness <= kSolvableTableBitness);
        }

        const size_t sample_size = 2 * bitness + 1;
        std::vector<std::vector<bool>> values(cases, std::vector<bool>(reps * sample_size));
        std::vector<float> targets(cases);

        const uint64_t value_seed =
                DomainSeed(seed, kind == GeneratorKind::Tree ? kTreeValueDomain : kTableValueDomain, bitness);

        std::vector<float> case_values(reps * sample_size);
        for (size_t case_index = 0; case_index < cases; ++case_index) {
            const size_t case_id = case_ids[case_index];
            if (kind == GeneratorKind::Tree) {
                DecisionTree tree = BuildTreeCase(bitness, case_id);
                tree.FillValueTensor(reps, CaseInputSeed(value_seed, bitness, case_id), case_values.data());
                targets[case_index] = static_cast<float>(bitness - tree.depth);
            } else {
                TableCase table(bitness, case_id);
                table.FillValueTensor(reps, CaseInputSeed(value_seed, bitness, case_id), case_values.data());
                const size_t depth = SolveForDepth(bitness, table.TruthTable());
                targets[case_index] = static_cast<float>(bitness - depth);
            }
            StoreBits(values[case_index], case_values);
        }

        return gen::GeneratedValues{gen::Values(std::move(values)), std::move(targets)};
    }

    gen::GeneratedRestrictions MakeRestrictions(uint16_t bitness, const std::vector<size_t> &case_ids, size_t reps,
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
            TableCase table(bitness, case_id);
            table.FillValueTensor(reps, CaseInputSeed(value_seed, bitness, case_id), case_values.data());
            table.FillRestrictionsTensor(reps, CaseInputSeed(restriction_seed, bitness, case_id),
                                         restriction_values.data());
            StoreBits(values[case_index], case_values);
            StoreBits(restrictions[case_index], restriction_values);
        }

        return gen::GeneratedRestrictions{gen::Values(std::move(values)), gen::Restrictions(std::move(restrictions))};
    }

} // namespace

namespace gen {

Values::Values(std::vector<std::vector<bool>> data) : data_(std::move(data)) {
    assert(!data_.empty());
    assert(!data_.front().empty());
    for (const std::vector<bool> &row: data_) {
        assert(row.size() == data_.front().size());
    }
}

Values Values::Concat(std::vector<Values> chunks) {
    assert(!chunks.empty());
    const size_t columns = chunks.front().Columns();
    size_t rows = 0;
    for (const Values &chunk: chunks) {
        assert(chunk.Columns() == columns);
        rows += chunk.Rows();
    }

    std::vector<std::vector<bool>> data;
    data.reserve(rows);
    for (Values &chunk: chunks) {
        for (std::vector<bool> &row: chunk.data_) {
            data.push_back(std::move(row));
        }
    }
    return Values(std::move(data));
}

void Values::WriteValues(float *output) const { WriteBits(data_, output); }

Restrictions::Restrictions(std::vector<std::vector<bool>> data) : data_(std::move(data)) {
    assert(!data_.empty());
    assert(!data_.front().empty());
    for (const std::vector<bool> &row: data_) {
        assert(row.size() == data_.front().size());
    }
}

Restrictions Restrictions::Concat(std::vector<Restrictions> chunks) {
    assert(!chunks.empty());
    const size_t columns = chunks.front().Columns();
    size_t rows = 0;
    for (const Restrictions &chunk: chunks) {
        assert(chunk.Columns() == columns);
        rows += chunk.Rows();
    }

    std::vector<std::vector<bool>> data;
    data.reserve(rows);
    for (Restrictions &chunk: chunks) {
        for (std::vector<bool> &row: chunk.data_) {
            data.push_back(std::move(row));
        }
    }
    return Restrictions(std::move(data));
}

void Restrictions::WriteValues(float *output) const { WriteBits(data_, output); }

std::vector<size_t> TreeSampleCaseIds(uint16_t bitness, size_t cases, uint64_t seed) {
    return SampleCaseIds(TreeCasesNumber(bitness), cases, DomainSeed(seed, kTreeSelectionDomain, bitness));
}

GeneratedValues TreeValuesForCases(uint16_t bitness, const std::vector<size_t> &case_ids, size_t reps,
                                   uint64_t seed) {
    return MakeValues(GeneratorKind::Tree, bitness, case_ids, reps, seed);
}

std::vector<size_t> TableSampleCaseIds(uint16_t bitness, size_t cases, uint64_t seed) {
    return SampleCaseIds(TableCasesNumber(bitness), cases, DomainSeed(seed, kTableSelectionDomain, bitness));
}

GeneratedValues TableValuesForCases(uint16_t bitness, const std::vector<size_t> &case_ids, size_t reps,
                                    uint64_t seed) {
    return MakeValues(GeneratorKind::Table, bitness, case_ids, reps, seed);
}

GeneratedRestrictions TableRestrictionsForCases(uint16_t bitness, const std::vector<size_t> &case_ids, size_t reps,
                                                 uint64_t seed) {
    return MakeRestrictions(bitness, case_ids, reps, seed);
}

} // namespace gen
