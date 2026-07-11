#include "tree.h"
#include "decision_tree.h"
#include "generator.h"
#include "utils.h"

#include <cassert>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

    using CaseKey = std::pair<uint16_t, size_t>;

    std::map<CaseKey, DecisionTree> g_decision_trees;
    std::mutex g_decision_trees_mutex;

    const DecisionTree &GetDecisionTree(uint16_t bitness, size_t case_id) {
        assert(case_id < gen::TreeCasesNumber(bitness));
        std::lock_guard<std::mutex> lock(g_decision_trees_mutex);

        const CaseKey key{bitness, case_id};
        auto it = g_decision_trees.find(key);
        if (it == g_decision_trees.end()) {
            it = g_decision_trees.emplace(key, BuildTreeCase(bitness, case_id)).first;
        }
        return it->second;
    }

    bool EvaluateTreeCase(uint16_t bitness, size_t case_id, std::string_view input) {
        assert(input.size() == bitness);
        return GetDecisionTree(bitness, case_id).Evaluate(input);
    }

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

    std::string value(2 * bitness + 1, '0');
    FlippingSampler sampler(bitness, input);

    const auto evaluate = [bitness, case_id](std::string_view point) {
        return EvaluateTreeCase(bitness, case_id, point);
    };
    sampler.Fill(value,
                 /*sample_offset=*/0, bitness, evaluate);

    return value;
}

} // namespace gen
