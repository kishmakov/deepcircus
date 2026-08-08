#include "tree.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "generator.h"
#include "utils.h"

namespace gen {

namespace {

constexpr uint64_t kTreeSelectionDomain = 0x747265655f73656cull;

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

TreeCase::TreeCase(uint16_t bitness, uint64_t seed) : Case(bitness, seed), used_bits(bitness, false) {
    assert(bitness >= kMinTreeBitness && bitness <= kMaxTreeBitness);
    // Internal-node budget the builder spends. Most draws are more nodes than
    // this bitness can place, so what the tree ends up being is mostly how
    // BuildSubtree's clamps and random splits spend it.
    const size_t budget = std::uniform_int_distribution<size_t>(0, kMaxTreeSize)(rng_);
    std::vector<bool> path_used_bits(bitness, false);
    const bool root_required_value = GenerateBool();
    BuildSubtree(budget, path_used_bits, /*path_used_count=*/0, root_required_value);
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

    const size_t bit_id = RandomUnusedBit(path_used_bits, free_bits, rng_);
    used_bits[bit_id] = true;
    path_used_bits[bit_id] = true;

    auto [left_budget, right_budget] = SplitBudget(budget, rng_);

    const size_t max_child_nodes = MaxInternalNodes(free_bits - 1);
    left_budget = std::min(left_budget, max_child_nodes);
    right_budget = std::min(right_budget, max_child_nodes);

    const bool child0_required_value = GenerateBool();
    const bool child1_required_value = !child0_required_value;

    const size_t child0 = BuildSubtree(left_budget, path_used_bits, path_used_count + 1, child0_required_value);
    const size_t child1 = BuildSubtree(right_budget, path_used_bits, path_used_count + 1, child1_required_value);
    path_used_bits[bit_id] = false;
    nodes[node_id] = Div{bit_id, child0, child1};
    return node_id;
}

bool TreeCase::Evaluate(const std::vector<bool>& input) const {
    assert(input.size() == bitness_);

    size_t node_id = 0;
    while (true) {
        const Node& node = nodes[node_id];
        const Div* division = std::get_if<Div>(&node);
        if (division == nullptr) {
            return std::get<bool>(node);
        }
        assert(division->bitId < input.size());
        node_id = input[division->bitId] ? division->child1 : division->child0;
    }
}

std::string TreeValue(uint16_t bitness, uint64_t seed, const std::vector<bool>& input) {
    assert(input.size() == bitness);

    const TreeCase tree(bitness, seed);
    return tree.SampledValueString(input);
}

std::vector<uint64_t> TreeSampleSeeds(uint16_t bitness, size_t cases, uint64_t task_seed) {
    return SampleSeeds(cases, DomainSeed(task_seed, kTreeSelectionDomain, bitness));
}

GeneratedValues TreeValuesForSeeds(uint16_t bitness, const std::vector<uint64_t>& seeds, InputShape shape) {
    const size_t cases = seeds.size();
    assert(cases > 0);
    assert(shape.batches > 1);
    assert(std::has_single_bit(shape.batch_size));
    assert(bitness >= kMinTreeBitness && bitness <= kMaxTreeBitness);

    const size_t sample_size = 2 * bitness + 1;
    const size_t columns = static_cast<size_t>(shape.batches) * shape.batch_size * sample_size;
    std::vector<bool> values(cases * columns);
    std::vector<float> targets(gen::kTargetsPerCase * cases);

    for (size_t case_index = 0; case_index < cases; ++case_index) {
        TreeCase tree(bitness, seeds[case_index]);
        const std::vector<bool> samples = tree.SampleValues(shape);
        assert(samples.size() == columns);
        std::copy(samples.begin(), samples.end(), values.begin() + case_index * columns);
        const size_t size = tree.nodes.size() - tree.num_leafs;
        targets[gen::kTargetsPerCase * case_index] = static_cast<float>(bitness - tree.depth);
        targets[gen::kTargetsPerCase * case_index + 1] = SizeScore(bitness, size);
    }

    return GeneratedValues{Values(cases, columns, std::move(values)), std::move(targets)};
}

}  // namespace gen
