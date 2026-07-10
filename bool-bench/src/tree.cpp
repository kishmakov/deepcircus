#include "bool_bench.h"
#include "decision_tree.h"
#include "tree.h"
#include "utils.h"

#include <cassert>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using CaseKey = std::pair<uint16_t, size_t>;

std::map<CaseKey, DecisionTree> g_decision_trees;
std::mutex g_decision_trees_mutex;

const DecisionTree& GetDecisionTree(uint16_t bitness, size_t case_id) {
    assert(case_id < bb_tree_cases_number(bitness));
    std::lock_guard<std::mutex> lock(g_decision_trees_mutex);

    const CaseKey key{bitness, case_id};
    auto it = g_decision_trees.find(key);
    if (it == g_decision_trees.end()) {
        it = g_decision_trees.emplace(key, BuildTreeCase(bitness, case_id)).first;
    }
    return it->second;
}

bool EvaluateTreeCase(
    uint16_t bitness,
    size_t case_id,
    std::string_view input)
{
    assert(input.size() == bitness);
    return GetDecisionTree(bitness, case_id).Evaluate(input);
}

}  // namespace

DecisionTree BuildTreeCase(uint16_t bitness, size_t case_id) {
    assert(bitness >= kMinTreeBitness && bitness <= kMaxTreeBitness);
    assert(case_id < kTreeCasesNumber);
    RandomBoolGenerator rng(PrepRNG(bitness, case_id));
    DecisionTree tree(bitness);
    std::vector<bool> path_used_bits(bitness, false);
    tree.BuildSubtree(
        case_id,
        path_used_bits,
        /*path_used_count=*/0,
        rng.Generate(),
        rng);
    tree.Finalize();
    return tree;
}

uint16_t bb_min_tree_bitness() {
    return kMinTreeBitness;
}

size_t bb_tree_cases_number(uint16_t bitness) {
    assert(bitness >= kMinTreeBitness && bitness <= kMaxTreeBitness);
    return kTreeCasesNumber;
}

const char* bb_tree_value(uint16_t bitness, size_t case_id, const char* input) {
    assert(case_id < bb_tree_cases_number(bitness));
    assert(input != nullptr);
    assert(std::strlen(input) == bitness);

    thread_local std::string value;
    thread_local FlippingSampler sampler;

    value.assign(2 * bitness + 1, '0');
    sampler.Reset(bitness, {input, bitness});

    const auto evaluate = [bitness, case_id](std::string_view point) {
        return EvaluateTreeCase(bitness, case_id, point);
    };
    sampler.Fill(
        value,
        /*sample_offset=*/0,
        bitness,
        evaluate);

    return value.c_str();
}
