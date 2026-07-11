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

    constexpr uint64_t kSplitMixIncrement = 0x9e3779b97f4a7c15ull;
    constexpr uint64_t kTreeSelectionDomain = 0x747265655f73656cull;
    constexpr uint64_t kTableSelectionDomain = 0x7461626c655f7365ull;
    constexpr uint64_t kTreeValueDomain = 0x747265655f76616cull;
    constexpr uint64_t kTableValueDomain = 0x7461626c655f7661ull;
    constexpr uint64_t kRestrictionDomain = 0x7265737472696374ull;

    enum class DataKind {
        Tree,
        Table,
    };

    uint64_t SplitMix64(uint64_t &state) {
        uint64_t value = (state += kSplitMixIncrement);
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
        return value ^ (value >> 31);
    }

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
    void StoreBits(std::vector<bool> &destination, size_t offset, const std::vector<float> &source) {
        assert(offset + source.size() <= destination.size());
        for (size_t index = 0; index < source.size(); ++index) {
            assert(source[index] == -1.0f || source[index] == 1.0f);
            destination[offset + index] = source[index] > 0.0f;
        }
    }

    void WriteBits(const std::vector<bool> &values, float *output) {
        assert(output != nullptr);
        for (size_t index = 0; index < values.size(); ++index) {
            output[index] = values[index] ? 1.0f : -1.0f;
        }
    }

    // This entire batch is generated synchronously on the calling thread.
    gen::Data MakeData(DataKind kind, uint16_t bitness, size_t cases, size_t reps, size_t restriction_chunk_cases,
                       uint64_t seed) {
        assert(cases > 0);
        assert(reps > 0);
        assert(reps % 2 == 0);
        if (kind == DataKind::Tree) {
            assert(bitness >= kMinTreeBitness && bitness <= kMaxTreeBitness);
            assert(restriction_chunk_cases == 0);
        } else if (bitness <= kSolvableTableBitness) {
            assert(bitness >= kMinTableBitness);
            assert(restriction_chunk_cases == 0);
        } else {
            assert(bitness <= kMaxTableBitness);
            assert(restriction_chunk_cases > 0);
            assert(restriction_chunk_cases <= cases);
        }

        const bool recursive = kind == DataKind::Table && bitness > kSolvableTableBitness;
        const size_t sample_size = 2 * bitness + 1;
        std::vector<bool> values(cases * reps * sample_size);
        std::vector<float> targets;
        if (!recursive) {
            targets.resize(cases);
        }

        const size_t restriction_size = recursive ? 2 * bitness * reps * (2 * bitness - 1) : 0;
        std::vector<gen::Tensor> restrictions;
        std::vector<std::vector<bool>> restriction_bits;
        if (recursive) {
            for (size_t first_case = 0; first_case < cases; first_case += restriction_chunk_cases) {
                const size_t chunk_cases = std::min(restriction_chunk_cases, cases - first_case);
                restriction_bits.emplace_back(chunk_cases * restriction_size);
            }
        }

        const uint64_t value_seed =
                DomainSeed(seed, kind == DataKind::Tree ? kTreeValueDomain : kTableValueDomain, bitness);
        const uint64_t restriction_seed = DomainSeed(seed, kRestrictionDomain, bitness);
        const size_t population = kind == DataKind::Tree ? gen::TreeCasesNumber(bitness) : gen::TableCasesNumber(bitness);
        const std::vector<size_t> case_ids = SampleCaseIds(
                population, cases,
                DomainSeed(seed, kind == DataKind::Tree ? kTreeSelectionDomain : kTableSelectionDomain, bitness));

        std::vector<float> case_values(reps * sample_size);
        std::vector<float> restriction_values(restriction_size);
        for (size_t case_index = 0; case_index < cases; ++case_index) {
            const size_t case_id = case_ids[case_index];
            if (kind == DataKind::Tree) {
                DecisionTree tree = BuildTreeCase(bitness, case_id);
                tree.FillValueTensor(reps, CaseInputSeed(value_seed, bitness, case_id), case_values.data());
                targets[case_index] = static_cast<float>(bitness - tree.depth);
            } else {
                TableCase table(bitness, case_id);
                table.FillValueTensor(reps, CaseInputSeed(value_seed, bitness, case_id), case_values.data());
                if (!recursive) {
                    const size_t depth = SolveForDepth(bitness, table.TruthTable());
                    targets[case_index] = static_cast<float>(bitness - depth);
                } else {
                    const size_t chunk_id = case_index / restriction_chunk_cases;
                    const size_t chunk_case = case_index % restriction_chunk_cases;
                    table.FillRestrictionsTensor(reps, CaseInputSeed(restriction_seed, bitness, case_id),
                                                 restriction_values.data());
                    StoreBits(restriction_bits[chunk_id], chunk_case * restriction_size, restriction_values);
                }
            }
            StoreBits(values, case_index * case_values.size(), case_values);
        }

        if (recursive) {
            restrictions.reserve(restriction_bits.size());
            for (size_t chunk = 0; chunk < restriction_bits.size(); ++chunk) {
                const size_t chunk_cases = restriction_bits[chunk].size() / restriction_size;
                restrictions.emplace_back(bitness, chunk_cases, reps, std::move(restriction_bits[chunk]));
            }
        }

        return gen::Data(bitness, cases, reps, std::move(values), std::move(targets), std::move(restrictions));
    }

} // namespace

namespace gen {

Tensor::Tensor(uint16_t bitness, size_t cases, size_t reps, std::vector<bool> values)
    : bitness_(bitness), cases_(cases), reps_(reps), values_(std::move(values)) {}

void Tensor::WriteValues(float *output) const { WriteBits(values_, output); }

Data::Data(uint16_t bitness, size_t cases, size_t reps, std::vector<bool> values, std::vector<float> targets,
           std::vector<Tensor> restrictions)
    : bitness_(bitness), cases_(cases), reps_(reps), values_(std::move(values)), targets_(std::move(targets)),
      restrictions_(std::move(restrictions)) {}

void Data::WriteValues(float *output) const { WriteBits(values_, output); }

void Data::WriteTargets(float *output) const {
    assert(output != nullptr);
    std::copy(targets_.begin(), targets_.end(), output);
}

Data TreeValueTensor(uint16_t bitness, size_t cases, size_t reps, uint64_t seed) {
    return MakeData(DataKind::Tree, bitness, cases, reps, 0, seed);
}

Data TableValueTensor(uint16_t bitness, size_t cases, size_t reps, size_t restriction_chunk_cases, uint64_t seed) {
    return MakeData(DataKind::Table, bitness, cases, reps, restriction_chunk_cases, seed);
}

} // namespace gen
