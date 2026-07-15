#include "tree.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "generator.h"
#include "table.h"
#include "utils.h"

namespace gen {

namespace {

constexpr uint64_t kTreeSelectionDomain = 0x747265655f73656cull;
constexpr uint64_t kTreeValueDomain = 0x747265655f76616cull;
constexpr uint64_t kTreeStructureDomain = 0x747265655f737472ull;

size_t RandomUnusedBit(const std::vector<bool>& path_used_bits, size_t free_bits, std::mt19937& rng) {
    assert(free_bits > 0);

    const size_t selected = std::uniform_int_distribution<size_t>(0, free_bits - 1)(rng);
    size_t seen = 0;
    for (size_t bit = 0; bit < path_used_bits.size(); ++bit) {
        if (path_used_bits[bit]) continue;
        if (seen == selected) return bit;
        ++seen;
    }
    assert(false);
    return 0;
}

std::pair<size_t, size_t> SplitBudget(size_t n, std::mt19937& rng) {
    assert(n >= 1);
    const size_t remaining = n - 1;
    if (remaining == 0) return {0, 0};
    const size_t left = std::uniform_int_distribution<size_t>(0, remaining)(rng);
    return {left, remaining - left};
}

size_t MaxInternalNodes(size_t free_bits) {
    if (free_bits >= std::numeric_limits<size_t>::digits) {
        return std::numeric_limits<size_t>::max();
    }
    return (size_t{1} << free_bits) - 1;
}

size_t ComputeDepth(const std::vector<Node>& nodes, size_t node_id) {
    const Node& node = nodes[node_id];
    const Div* division = std::get_if<Div>(&node);
    if (division == nullptr) {
        return 0;
    }
    return 1 + std::max(ComputeDepth(nodes, division->child0), ComputeDepth(nodes, division->child1));
}

}  // namespace

TreeCase::TreeCase(uint16_t bitness, size_t case_id, uint64_t seed)
    : Case(bitness, case_id, DomainSeed(seed, kTreeStructureDomain, bitness)), used_bits(bitness, false) {
    assert(bitness >= kMinTreeBitness && bitness <= kMaxTreeBitness);
    assert(case_id < kTreeCasesNumber);
    std::vector<bool> path_used_bits(bitness, false);
    const bool root_required_value = Generate();
    BuildSubtree(case_id, path_used_bits, /*path_used_count=*/0, root_required_value);
    depth = ComputeDepth(nodes, 0);
}

size_t TreeCase::AddLeaf(bool value) {
    const size_t node_id = nodes.size();
    nodes.push_back(value);
    ++num_leafs;
    return node_id;
}

size_t TreeCase::BuildSubtree(size_t budget, std::vector<bool>& path_used_bits, size_t path_used_count,
                                  bool required_value) {
    assert(path_used_bits.size() == bitness_);
    assert(path_used_count <= bitness_);

    const size_t free_bits = bitness_ - path_used_count;
    if (budget == 0 || free_bits == 0) {
        return AddLeaf(required_value);
    }

    const size_t node_id = nodes.size();
    nodes.push_back(false);

    const size_t bit_id = RandomUnusedBit(path_used_bits, free_bits, RNG());
    used_bits[bit_id] = true;
    path_used_bits[bit_id] = true;

    auto [left_budget, right_budget] = SplitBudget(budget, RNG());

    const size_t max_child_nodes = MaxInternalNodes(free_bits - 1);
    left_budget = std::min(left_budget, max_child_nodes);
    right_budget = std::min(right_budget, max_child_nodes);

    const bool child0_required_value = Generate();
    const bool child1_required_value = !child0_required_value;

    const size_t child0 = BuildSubtree(left_budget, path_used_bits, path_used_count + 1, child0_required_value);
    const size_t child1 = BuildSubtree(right_budget, path_used_bits, path_used_count + 1, child1_required_value);
    path_used_bits[bit_id] = false;
    nodes[node_id] = Div{bit_id, child0, child1};
    return node_id;
}

bool TreeCase::Evaluate(std::string_view input) const {
    assert(input.size() == bitness_);

    size_t node_id = 0;
    while (true) {
        const Node& node = nodes[node_id];
        const Div* division = std::get_if<Div>(&node);
        if (division == nullptr) {
            return std::get<bool>(node);
        }
        assert(division->bitId < input.size());
        node_id = input[division->bitId] == '1' ? division->child1 : division->child0;
    }
}

void TreeCase::FillValueTensor(size_t reps, uint64_t seed, std::vector<bool>& out, size_t base) const {
    const auto evaluate = [this](std::string_view input) { return Evaluate(input); };
    FillGeneratedValueTensor(bitness_, reps, seed, out, base, evaluate);
}

size_t TreeCasesNumber(uint16_t bitness) {
    assert(bitness >= kMinTreeBitness && bitness <= kMaxTreeBitness);
    return kTreeCasesNumber;
}

std::string TreeValue(uint16_t bitness, size_t case_id, uint64_t seed, std::string_view input) {
    assert(case_id < TreeCasesNumber(bitness));
    assert(input.size() == bitness);

    const TreeCase tree(bitness, case_id, seed);
    const auto evaluate = [&tree](std::string_view point) { return tree.Evaluate(point); };
    return SampledValueString(bitness, input, evaluate);
}

std::vector<size_t> TreeSampleCaseIds(uint16_t bitness, size_t cases, uint64_t seed) {
    return SampleCaseIds(TreeCasesNumber(bitness), cases, DomainSeed(seed, kTreeSelectionDomain, bitness));
}

GeneratedValues TreeValuesForCases(uint16_t bitness, const std::vector<size_t>& case_ids, size_t reps, uint64_t seed) {
    const size_t cases = case_ids.size();
    assert(cases > 0);
    assert(reps > 0);
    assert(reps % 2 == 0);
    assert(bitness >= kMinTreeBitness && bitness <= kMaxTreeBitness);

    const size_t sample_size = 2 * bitness + 1;
    const size_t columns = reps * sample_size;
    std::vector<bool> values(cases * columns);
    std::vector<float> targets(gen::kTargetsPerCase * cases);
    const uint64_t value_seed = DomainSeed(seed, kTreeValueDomain, bitness);

    for (size_t case_index = 0; case_index < cases; ++case_index) {
        const size_t case_id = case_ids[case_index];
        TreeCase tree(bitness, case_id, seed);
        tree.FillValueTensor(reps, CaseInputSeed(value_seed, bitness, case_id), values, case_index * columns);
        const size_t size = tree.nodes.size() - tree.num_leafs;
        targets[gen::kTargetsPerCase * case_index] = static_cast<float>(bitness - tree.depth);
        targets[gen::kTargetsPerCase * case_index + 1] = SizeScore(bitness, size);
    }

    return GeneratedValues{Values(cases, columns, std::move(values)), std::move(targets)};
}

}  // namespace gen

