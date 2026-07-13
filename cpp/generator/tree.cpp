#include "tree.h"
#include "decision_tree.h"
#include "generator.h"
#include "utils.h"

#include <cassert>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

    constexpr uint64_t kTreeSelectionDomain = 0x747265655f73656cull;
    constexpr uint64_t kTreeValueDomain = 0x747265655f76616cull;

} // namespace

DecisionTree BuildTreeCase(uint16_t bitness, size_t case_id) {
    assert(bitness >= kMinTreeBitness && bitness <= kMaxTreeBitness);
    assert(case_id < kTreeCasesNumber);
    RandomBoolGenerator rng(PrepRNG(bitness, case_id));
    DecisionTree tree(bitness);
    std::vector<bool> path_used_bits(bitness, false);
    tree.BuildSubtree(case_id, path_used_bits,
                      /*path_used_count=*/0, rng.Generate(), rng);
    tree.Finalize();
    return tree;
}

namespace gen {

uint16_t MinTreeBitness() { return kMinTreeBitness; }

size_t TreeCasesNumber(uint16_t bitness) {
    assert(bitness >= kMinTreeBitness && bitness <= kMaxTreeBitness);
    return kTreeCasesNumber;
}

std::string TreeValue(uint16_t bitness, size_t case_id, std::string_view input) {
    assert(case_id < TreeCasesNumber(bitness));
    assert(input.size() == bitness);

    const DecisionTree tree = BuildTreeCase(bitness, case_id);
    const auto evaluate = [&tree](std::string_view point) { return tree.Evaluate(point); };
    return SampledValueString(bitness, input, evaluate);
}

std::vector<size_t> TreeSampleCaseIds(uint16_t bitness, size_t cases, uint64_t seed) {
    return SampleCaseIds(TreeCasesNumber(bitness), cases, DomainSeed(seed, kTreeSelectionDomain, bitness));
}

GeneratedValues TreeValuesForCases(uint16_t bitness, const std::vector<size_t> &case_ids, size_t reps,
                                   uint64_t seed) {
    const size_t cases = case_ids.size();
    assert(cases > 0);
    assert(reps > 0);
    assert(reps % 2 == 0);
    assert(bitness >= kMinTreeBitness && bitness <= kMaxTreeBitness);

    const size_t sample_size = 2 * bitness + 1;
    std::vector<std::vector<bool>> values(cases, std::vector<bool>(reps * sample_size));
    std::vector<float> targets(cases);
    const uint64_t value_seed = DomainSeed(seed, kTreeValueDomain, bitness);

    std::vector<float> case_values(reps * sample_size);
    for (size_t case_index = 0; case_index < cases; ++case_index) {
        const size_t case_id = case_ids[case_index];
        DecisionTree tree = BuildTreeCase(bitness, case_id);
        tree.FillValueTensor(reps, CaseInputSeed(value_seed, bitness, case_id), case_values.data());
        targets[case_index] = static_cast<float>(bitness - tree.depth);
        StoreBits(values[case_index], case_values);
    }

    return GeneratedValues{Values(std::move(values)), std::move(targets)};
}

} // namespace gen
