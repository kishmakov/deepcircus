#include "decision_tree.h"
#include "generator.h"
#include "table.h"
#include "tree.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
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

} // namespace

struct gen_tensor {
    uint16_t bitness;
    size_t cases;
    size_t reps;
    std::vector<bool> values;
};

struct gen_data {
    DataKind kind;
    uint16_t bitness;
    size_t cases;
    size_t reps;
    std::vector<bool> values;
    std::vector<float> targets;
    // Recursive table restrictions are complete and owned by the parent result.
    std::vector<std::unique_ptr<gen_tensor>> restrictions;
};

namespace {

    // This entire batch is generated synchronously on the calling thread.
    std::unique_ptr<gen_data> MakeData(DataKind kind, uint16_t bitness, size_t cases, size_t reps,
                                       size_t restriction_chunk_cases, uint64_t seed) {
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

        auto data = std::make_unique<gen_data>();
        data->kind = kind;
        data->bitness = bitness;
        data->cases = cases;
        data->reps = reps;
        const size_t sample_size = 2 * bitness + 1;
        data->values.resize(cases * reps * sample_size);
        if (kind == DataKind::Tree || bitness <= kSolvableTableBitness) {
            data->targets.resize(cases);
        }

        if (kind == DataKind::Table && bitness > kSolvableTableBitness) {
            for (size_t first_case = 0; first_case < cases; first_case += restriction_chunk_cases) {
                auto tensor = std::make_unique<gen_tensor>();
                tensor->bitness = bitness;
                tensor->cases = std::min(restriction_chunk_cases, cases - first_case);
                tensor->reps = reps;
                tensor->values.resize(tensor->cases * 2 * bitness * reps * (2 * bitness - 1));
                data->restrictions.push_back(std::move(tensor));
            }
        }

        const uint64_t value_seed =
                DomainSeed(seed, kind == DataKind::Tree ? kTreeValueDomain : kTableValueDomain, bitness);
        const uint64_t restriction_seed = DomainSeed(seed, kRestrictionDomain, bitness);
        const size_t population =
                kind == DataKind::Tree ? gen_tree_cases_number(bitness) : gen_table_cases_number(bitness);
        const std::vector<size_t> case_ids = SampleCaseIds(
                population, cases,
                DomainSeed(seed, kind == DataKind::Tree ? kTreeSelectionDomain : kTableSelectionDomain, bitness));

        std::vector<float> case_values(reps * sample_size);
        const size_t restriction_size =
                kind == DataKind::Table && bitness > kSolvableTableBitness ? 2 * bitness * reps * (2 * bitness - 1) : 0;
        std::vector<float> restriction_values(restriction_size);
        for (size_t case_index = 0; case_index < cases; ++case_index) {
            const size_t case_id = case_ids[case_index];
            if (kind == DataKind::Tree) {
                DecisionTree tree = BuildTreeCase(bitness, case_id);
                tree.FillValueTensor(reps, CaseInputSeed(value_seed, bitness, case_id), case_values.data());
                data->targets[case_index] = static_cast<float>(bitness - tree.depth);
            } else {
                TableCase table(bitness, case_id);
                table.FillValueTensor(reps, CaseInputSeed(value_seed, bitness, case_id), case_values.data());
                if (bitness <= kSolvableTableBitness) {
                    const size_t depth = SolveForDepth(bitness, table.TruthTable());
                    data->targets[case_index] = static_cast<float>(bitness - depth);
                } else {
                    const size_t chunk_id = case_index / restriction_chunk_cases;
                    const size_t chunk_case = case_index % restriction_chunk_cases;
                    gen_tensor *tensor = data->restrictions[chunk_id].get();
                    table.FillRestrictionsTensor(reps, CaseInputSeed(restriction_seed, bitness, case_id),
                                                 restriction_values.data());
                    StoreBits(tensor->values, chunk_case * restriction_size, restriction_values);
                }
            }
            StoreBits(data->values, case_index * case_values.size(), case_values);
        }
        return data;
    }

} // namespace

gen_data *gen_tree_value_tensor(uint16_t bitness, size_t cases, size_t reps, uint64_t seed) {
    return MakeData(DataKind::Tree, bitness, cases, reps, 0, seed).release();
}

gen_data *gen_table_value_tensor(uint16_t bitness, size_t cases, size_t reps, size_t restriction_chunk_cases,
                                 uint64_t seed) {
    return MakeData(DataKind::Table, bitness, cases, reps, restriction_chunk_cases, seed).release();
}

uint16_t gen_data_bitness(const gen_data *data) {
    assert(data != nullptr);
    return data->bitness;
}

size_t gen_data_cases(const gen_data *data) {
    assert(data != nullptr);
    return data->cases;
}

size_t gen_data_reps(const gen_data *data) {
    assert(data != nullptr);
    return data->reps;
}

size_t gen_data_value_count(const gen_data *data) {
    assert(data != nullptr);
    return data->values.size();
}

size_t gen_data_target_count(const gen_data *data) {
    assert(data != nullptr);
    return data->targets.size();
}

void gen_data_write_values(const gen_data *data, float *output) {
    assert(data != nullptr);
    WriteBits(data->values, output);
}

void gen_data_write_targets(const gen_data *data, float *output) {
    assert(data != nullptr);
    assert(output != nullptr);
    std::copy(data->targets.begin(), data->targets.end(), output);
}

size_t gen_data_restriction_count(const gen_data *data) {
    assert(data != nullptr);
    return data->restrictions.size();
}

const gen_tensor *gen_data_restriction(const gen_data *data, size_t index) {
    assert(data != nullptr);
    assert(index < data->restrictions.size());
    return data->restrictions[index].get();
}

void gen_data_destroy(gen_data *data) { delete data; }

uint16_t gen_tensor_bitness(const gen_tensor *tensor) {
    assert(tensor != nullptr);
    return tensor->bitness;
}

size_t gen_tensor_cases(const gen_tensor *tensor) {
    assert(tensor != nullptr);
    return tensor->cases;
}

size_t gen_tensor_reps(const gen_tensor *tensor) {
    assert(tensor != nullptr);
    return tensor->reps;
}

size_t gen_tensor_value_count(const gen_tensor *tensor) {
    assert(tensor != nullptr);
    return tensor->values.size();
}

void gen_tensor_write_values(const gen_tensor *tensor, float *output) {
    assert(tensor != nullptr);
    WriteBits(tensor->values, output);
}
