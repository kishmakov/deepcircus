#include "tree.h"
#include "decision_tree.h"
#include "generator.h"
#include "utils.h"

#include <cassert>
#include <string>
#include <string_view>
#include <vector>

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

    std::string value(2 * bitness + 1, '0');
    FlippingSampler sampler(bitness, input);

    const DecisionTree tree = BuildTreeCase(bitness, case_id);
    const auto evaluate = [&tree](std::string_view point) {
        return tree.Evaluate(point);
    };
    sampler.Fill(value,
                 /*sample_offset=*/0, bitness, evaluate);

    return value;
}

} // namespace gen
